# Progetto 02 — Inferenza Locale Ottimizzata + Router Costo/Qualità

> **Ruolo nel portfolio:** il pezzo model-level. Coerente col tuo approccio local-first.
> **Skill segnalata:** comprensione del modello sotto il livello dell'API. Ottimizzazione di una funzione costo/qualità/latenza reale — AI engineering puro.

---

## 1. Obiettivo

Prendere un modello open, farlo girare in locale ottimizzato, e **documentare i trade-off** tra quantizzazione, qualità, throughput e VRAM. Sopra, un router che instrada le query semplici su modello piccolo e quelle complesse su modello grande, ottimizzando costo/qualità.

Il valore **non** è "ho fatto girare un modello". È la **tabella di trade-off misurata** e la decisione di routing motivata dai dati.

## 2. Perché segnala (lettura recruiter)

- La maggior parte dei candidati sa solo chiamare API. Qui dimostri di capire cosa succede *dentro*: quantizzazione, KV cache, throughput, batching.
- Il router costo/qualità è esattamente il tipo di ottimizzazione che si fa in produzione per tagliare i costi senza perdere qualità.
- Local-first / privacy-conscious è un posizionamento credibile e on-brand per te.

## 3. Scope — cosa fa

- Serving locale di un modello open (es. Qwen3, Llama, Mistral) con un engine serio.
- Benchmark sistematico: per ogni livello di quantizzazione → latency (p50/p95), throughput (tok/s), VRAM, **qualità** (misurata con l'harness del progetto 01).
- **Router** che classifica la difficoltà della query e sceglie il modello/config target.
- Report con tabelle e grafici dei trade-off.

## 4. Scope — cosa NON fa

- Non addestra né fine-tuna (quello è eventualmente un altro progetto).
- Non è una UI di chat — è un endpoint + benchmark + router.

## 5. Stack consigliato

| Layer | Scelta | Note |
|---|---|---|
| Engine serving | **vLLM** (se GPU) *oppure* **llama.cpp** (CPU/GPU consumer, GGUF) | scegli in base all'hardware che hai |
| Modelli | 1 piccolo + 1 grande della stessa famiglia | es. Qwen3 in due taglie |
| Quantizzazione | GGUF a vari bit (Q4, Q5, Q8) o AWQ/GPTQ con vLLM | il cuore del benchmark |
| Benchmark qualità | **riusa il Progetto 01** | qui si chiude il loop |
| Benchmark perf | script tuo + eventualmente strumenti dell'engine | misura tok/s, TTFT, VRAM |
| Router | classifier leggero (euristica → poi modello piccolo come router) | parti semplice |
| API | FastAPI | coerente col tuo stack |

## 6. Architettura

```
serving/
  engine.py             # wrapper sull'engine (vLLM/llama.cpp)
  configs/              # una config per (modello, quant, params)
benchmark/
  perf.py               # TTFT, tok/s, p50/p95 latency, VRAM peak
  quality.py            # invoca eval-harness su ogni config
  sweep.py              # gira tutte le combinazioni → raccoglie risultati
router/
  classifier.py         # stima difficoltà query → sceglie target
  policy.py             # regole costo/qualità (soglie, fallback)
api/
  main.py               # endpoint OpenAI-compatible che usa il router
report/
  tradeoffs.py          # tabella + grafici quant × qualità × tok/s × VRAM
```

## 7. Il benchmark — cuore del progetto

Per **ogni configurazione** (modello × livello di quant) misura:

| Metrica | Come |
|---|---|
| Qualità | eval-harness (Progetto 01) → score medio sul task |
| Throughput | tok/s in generazione, a batch 1 e batch N |
| TTFT | time-to-first-token |
| Latency | p50 / p95 su N richieste |
| VRAM / RAM | picco di memoria |
| Costo equivalente | $/1M token stimato (se confronti con API cloud) |

Il deliverable è il grafico **qualità vs costo/latenza**: si vede a colpo d'occhio dove sta il punto di Pareto. Quello è il pezzo che fa pensare "questo sa cosa fa".

## 8. Il router — il pezzo "engineering"

- **Input:** query dell'utente.
- **Decisione:** difficoltà stimata → modello piccolo (veloce/economico) o grande (qualità).
- **Policy:** soglie configurabili; fallback al modello grande se il piccolo ha bassa confidenza.
- **Misura:** quanto risparmi (% query servite dal piccolo) a parità di qualità misurata. **Questo numero va nel README.**

Progressione consigliata:
1. Router euristico (lunghezza, keyword, presenza di codice/matematica).
2. Router come classifier addestrato/promptato su un set etichettato.
3. Confronto: quanto è meglio del puramente euristico (di nuovo, misurato con l'harness).

## 9. Fasi di sviluppo

- **Fase 0:** scelta hardware/engine, serving di un modello, endpoint funzionante.
- **Fase 1:** benchmark perf (tok/s, latency, VRAM) su una config.
- **Fase 2:** sweep su tutti i livelli di quant + qualità via harness → tabella trade-off.
- **Fase 3:** router euristico + misura del risparmio.
- **Fase 4:** router intelligente + confronto.
- **Fase 5:** report con grafici Pareto + scrittura README.

## 10. Deliverable concreti

- README con la **tabella trade-off** e il **grafico qualità/costo** (il pezzo forte).
- Endpoint OpenAI-compatible che gira in locale.
- Numero chiaro: *"il router serve l'X% delle query dal modello piccolo, risparmiando Y% di costo/latenza con < Z% di calo qualità"*.
- Sezione hardware: cosa hai usato, cosa serve per riprodurre.

## 11. Metriche di successo del progetto

- Aggiungere una nuova config = aggiungere un file, lo sweep la include.
- Il grafico di Pareto è leggibile e motiva una scelta.
- Il router dà un risparmio quantificato e onesto.

## 12. Collegamenti con gli altri progetti

- **← 01 Eval Harness:** è lo strumento che misura la qualità di ogni config. Senza, il benchmark non ha l'asse "qualità".
- **→ 03 Osservabilità:** l'endpoint locale è instrumentato col tracing (token, latenza, quale modello ha scelto il router).

## 13. Estensioni (se vuoi spingere)

- **Speculative decoding** (modello draft piccolo + verifica col grande): tecnicamente impressionante.
- KV-cache / continuous batching: misura il guadagno di throughput.
- Confronto locale vs API cloud sullo stesso task: costo, latenza, privacy.
