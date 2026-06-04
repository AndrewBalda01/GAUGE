"""
Engine abstraction layer.

Concrete implementations:
  - LlamaCppEngine  — wraps llama-cpp-python (GGUF, CPU/GPU)
  - VllmEngine      — wraps vLLM (GPU, AWQ/GPTQ)
  - MockEngine      — deterministic fake for testing without hardware

All engines share the same async interface so the benchmark and API
never depend on which backend is loaded.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GenerateRequest:
    prompt: str
    system: str = ""
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    stop: list[str] = field(default_factory=list)


@dataclass
class GenerateResponse:
    text: str
    tokens_prompt: int
    tokens_generated: int
    ttft_ms: float          # time-to-first-token
    total_ms: float
    tokens_per_second: float

    @property
    def tokens_total(self) -> int:
        return self.tokens_prompt + self.tokens_generated


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseEngine(ABC):
    name: str = "base"

    @abstractmethod
    async def load(self) -> None:
        """Load / warm-up the model. Called once at startup."""

    @abstractmethod
    async def unload(self) -> None:
        """Release GPU/CPU memory."""

    @abstractmethod
    async def generate(self, req: GenerateRequest) -> GenerateResponse:
        """Single non-streaming generation."""

    @abstractmethod
    async def generate_stream(
        self, req: GenerateRequest
    ) -> AsyncIterator[str]:
        """Token-by-token streaming generation (yields text chunks)."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

    @property
    @abstractmethod
    def vram_mb(self) -> float:
        """Current VRAM usage in MB (0 if CPU-only or unavailable)."""


# ---------------------------------------------------------------------------
# llama.cpp engine
# ---------------------------------------------------------------------------

class LlamaCppEngine(BaseEngine):
    """
    Wraps llama-cpp-python.
    Install: pip install llama-cpp-python
    GPU:     CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        n_threads: int | None = None,
        name: str = "llama.cpp",
    ) -> None:
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.name = name
        self._model = None

    async def load(self) -> None:
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python not installed. "
                "Run: pip install llama-cpp-python"
            ) from exc
        self._model = await asyncio.to_thread(
            Llama,
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            n_threads=self.n_threads,
            verbose=False,
        )

    async def unload(self) -> None:
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def vram_mb(self) -> float:
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True,
            )
            return float(out.strip().split("\n")[0])
        except Exception:
            return 0.0

    async def generate(self, req: GenerateRequest) -> GenerateResponse:
        if not self.is_loaded:
            raise RuntimeError("Engine not loaded. Call load() first.")

        prompt = req.prompt
        if req.system:
            prompt = f"<|system|>\n{req.system}\n<|user|>\n{req.prompt}\n<|assistant|>\n"

        t0 = time.perf_counter()
        first_token_time: float | None = None

        tokens_generated = 0
        output_text = ""

        def _run():
            nonlocal first_token_time, tokens_generated, output_text
            result = self._model(
                prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                stop=req.stop or [],
                echo=False,
            )
            first_token_time = time.perf_counter()
            output_text = result["choices"][0]["text"]
            tokens_generated = result["usage"]["completion_tokens"]
            return result

        result = await asyncio.to_thread(_run)
        total_ms = (time.perf_counter() - t0) * 1000
        ttft_ms = ((first_token_time or time.perf_counter()) - t0) * 1000
        tokens_prompt = result["usage"]["prompt_tokens"]
        tps = tokens_generated / (total_ms / 1000) if total_ms > 0 else 0.0

        return GenerateResponse(
            text=output_text,
            tokens_prompt=tokens_prompt,
            tokens_generated=tokens_generated,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            tokens_per_second=tps,
        )

    async def generate_stream(self, req: GenerateRequest) -> AsyncIterator[str]:
        if not self.is_loaded:
            raise RuntimeError("Engine not loaded.")
        prompt = req.prompt
        if req.system:
            prompt = f"<|system|>\n{req.system}\n<|user|>\n{req.prompt}\n<|assistant|>\n"

        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def _stream():
            for chunk in self._model(
                prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                stop=req.stop or [],
                stream=True,
                echo=False,
            ):
                text = chunk["choices"][0].get("text", "")
                if text:
                    queue.put_nowait(text)
            queue.put_nowait(None)

        asyncio.get_event_loop().run_in_executor(None, _stream)
        while True:
            token = await queue.get()
            if token is None:
                break
            yield token


# ---------------------------------------------------------------------------
# vLLM engine
# ---------------------------------------------------------------------------

class VllmEngine(BaseEngine):
    """
    Wraps vLLM (requires GPU + pip install vllm).
    Supports AWQ and GPTQ quantization via model_kwargs.
    """

    def __init__(
        self,
        model_id: str,
        quantization: str | None = None,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 4096,
        name: str = "vllm",
    ) -> None:
        self.model_id = model_id
        self.quantization = quantization
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.name = name
        self._llm = None
        self._sampling_params_cls = None

    async def load(self) -> None:
        try:
            from vllm import LLM, SamplingParams  # type: ignore
        except ImportError as exc:
            raise RuntimeError("vllm not installed. Run: pip install vllm") from exc

        kwargs: dict = dict(
            model=self.model_id,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
        )
        if self.quantization:
            kwargs["quantization"] = self.quantization

        self._llm = await asyncio.to_thread(LLM, **kwargs)
        self._sampling_params_cls = SamplingParams

    async def unload(self) -> None:
        self._llm = None

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    @property
    def vram_mb(self) -> float:
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True,
            )
            return float(out.strip().split("\n")[0])
        except Exception:
            return 0.0

    async def generate(self, req: GenerateRequest) -> GenerateResponse:
        if not self.is_loaded:
            raise RuntimeError("Engine not loaded.")

        prompt = req.prompt
        if req.system:
            prompt = f"System: {req.system}\n\nUser: {req.prompt}\n\nAssistant:"

        sp = self._sampling_params_cls(
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
            stop=req.stop or [],
        )
        t0 = time.perf_counter()
        outputs = await asyncio.to_thread(self._llm.generate, [prompt], sp)
        total_ms = (time.perf_counter() - t0) * 1000

        out = outputs[0].outputs[0]
        tokens_generated = len(out.token_ids)
        tokens_prompt = len(outputs[0].prompt_token_ids)
        tps = tokens_generated / (total_ms / 1000) if total_ms > 0 else 0.0

        return GenerateResponse(
            text=out.text,
            tokens_prompt=tokens_prompt,
            tokens_generated=tokens_generated,
            ttft_ms=total_ms,  # vLLM batch mode: ttft ≈ total
            total_ms=total_ms,
            tokens_per_second=tps,
        )

    async def generate_stream(self, req: GenerateRequest) -> AsyncIterator[str]:
        resp = await self.generate(req)
        yield resp.text


# ---------------------------------------------------------------------------
# Mock engine (testing / CI without hardware)
# ---------------------------------------------------------------------------

class MockEngine(BaseEngine):
    """
    Deterministic fake engine for tests.
    Returns a fixed response, simulates latency proportional to max_tokens.
    """

    def __init__(
        self,
        name: str = "mock",
        tokens_per_second: float = 50.0,
        ttft_ms: float = 80.0,
        response_template: str = "Mock response for: {prompt}",
    ) -> None:
        self.name = name
        self._tokens_per_second = tokens_per_second
        self._ttft_ms = ttft_ms
        self._response_template = response_template
        self._loaded = False

    async def load(self) -> None:
        self._loaded = True

    async def unload(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def vram_mb(self) -> float:
        return 0.0

    async def generate(self, req: GenerateRequest) -> GenerateResponse:
        tokens_out = min(req.max_tokens, 32)
        simulated_ms = self._ttft_ms + (tokens_out / self._tokens_per_second) * 1000
        await asyncio.sleep(simulated_ms / 1000)

        text = self._response_template.format(prompt=req.prompt[:60])
        prompt_tokens = len(req.prompt.split())

        return GenerateResponse(
            text=text,
            tokens_prompt=prompt_tokens,
            tokens_generated=tokens_out,
            ttft_ms=self._ttft_ms,
            total_ms=simulated_ms,
            tokens_per_second=self._tokens_per_second,
        )

    async def generate_stream(self, req: GenerateRequest) -> AsyncIterator[str]:
        resp = await self.generate(req)
        for word in resp.text.split():
            await asyncio.sleep(0.005)
            yield word + " "


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_engine(config: dict) -> BaseEngine:
    """
    Build an engine from a config dict (loaded from YAML).
    config keys: backend, name, model_path/model_id, n_gpu_layers, quantization, ...
    """
    backend = config.get("backend", "mock")
    if backend == "llama.cpp":
        return LlamaCppEngine(
            model_path=config["model_path"],
            n_ctx=config.get("n_ctx", 4096),
            n_gpu_layers=config.get("n_gpu_layers", 0),
            n_threads=config.get("n_threads"),
            name=config.get("name", "llama.cpp"),
        )
    if backend == "vllm":
        return VllmEngine(
            model_id=config["model_id"],
            quantization=config.get("quantization"),
            gpu_memory_utilization=config.get("gpu_memory_utilization", 0.90),
            max_model_len=config.get("max_model_len", 4096),
            name=config.get("name", "vllm"),
        )
    if backend == "mock":
        return MockEngine(
            name=config.get("name", "mock"),
            tokens_per_second=config.get("tokens_per_second", 50.0),
            ttft_ms=config.get("ttft_ms", 80.0),
            response_template=config.get("response_template", "Mock: {prompt}"),
        )
    raise ValueError(f"Unknown engine backend: {backend!r}")
