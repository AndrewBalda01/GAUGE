# Progetto 01 — LLM Evaluation Harness

> **Ruolo nel portfolio:** progetto fondante. È quello che ti qualifica come AI engineer e non come prompt-tinkerer. Gli altri tre progetti lo richiamano.
> **Skill segnalata:** rigore ingegneristico, pensiero da produzione, capacità di *misurare* la qualità invece di affidarsi al "sembra che funzioni".

---

## 1. Obiettivo

Un framework per valutare modelli e prompt in modo rigoroso, riproducibile e automatizzato. Deve rispondere alla domanda: *"questa modifica al prompt/modello ha migliorato o peggiorato la qualità, e di quanto?"* — con un numero, non con un'impressione.

Il deliverable che fa scena: **CI integration** che fa girare la eval suite a ogni push e blocca il merge se la qualità regredisce oltre una soglia. È esattamente il pensiero che i recruiter cercano in un AI engineer.

## 2. Perché segnala (lettura recruiter)

- Pochissimi junior sanno fare eval serio. È la cosa che separa "ci gioco" da "lo metto in produzione".
- Mostra che capisci il ciclo di vita di un sistema LLM: non finisce quando "risponde", finisce quando è *misurabile e non regredisce*.
- È la skill più rara e più richiesta del profilo.

## 3. Scope — cosa fa

- Dataset di test **versionati** (no dataset volanti: ogni run è riproducibile).
- Esecuzione di una test suite contro uno o più modelli/configurazioni di prompt.
- Metriche multiple:
  - **LLM-as-judge** con rubrica strutturata (output JSON validato).
  - Metriche deterministiche dove possibile (exact match, regex, JSON schema validity, latency, costo).
  - `pass@k` per task con varianza.
  - **Regression detection**: confronto tra run, flag su peggioramenti.
- Persistenza di tutti i run (input, output, voti, metadata, costo, latenza).
- Dashboard / report comparativo tra run.
- Hook CI che fallisce la build su regressione.

## 4. Scope — cosa NON fa (per non disperdersi)

- Non è un playground interattivo (no UI di chat).
- Non addestra modelli.
- Non gestisce serving — consuma endpoint (API o locale).

## 5. Stack consigliato

| Layer | Scelta | Note |
|---|---|---|
| Linguaggio | Python 3.11+ | coerente col tuo stack (FastAPI/SQLite) |
| Validazione output | **Pydantic** | judge restituisce JSON tipizzato e validato |
| Storage run | SQLite (start) → Postgres (scale) | parti semplice |
| Tracing/eval store | **Langfuse self-hosted** *oppure* layer tuo su SQLite | scegli "tuo" se vuoi mostrare che sai costruirlo |
| Orchestrazione LLM | client async diretto (httpx) | evita astrazioni pesanti, fa vedere che capisci cosa succede |
| CI | **GitHub Actions** | il pezzo che fa la differenza |
| Report | Markdown auto-generato + grafici (matplotlib/plotly) | report in PR commentato dal bot CI = ottimo effetto |

## 6. Architettura

```
datasets/                 # test set versionati (jsonl + manifest con hash)
  legal_qa.v1.jsonl
  legal_qa.manifest.json
evals/
  cases.py                # definizione test case (input, expected, metric refs)
  metrics/
    judge.py              # LLM-as-judge + rubrica
    deterministic.py      # exact match, schema validity, ecc.
    aggregate.py          # pass@k, medie, percentili
runners/
  runner.py               # esegue suite su una config (modello+prompt+params)
store/
  schema.py               # Pydantic models + tabelle
  db.py                   # persistenza run
report/
  compare.py              # diff tra due run → regressione sì/no
  render.py               # genera report markdown + grafici
.github/workflows/
  eval.yml                # gira suite, posta report in PR, blocca merge su regressione
```

### Flusso di un run
1. Carica dataset versionato (verifica hash → riproducibilità).
2. Per ogni case × config: chiama il modello, raccoglie output + latenza + costo.
3. Applica metriche (deterministiche + judge).
4. Persisti run con tutti i metadata.
5. Confronta con baseline (ultimo run su `main`).
6. Genera report; se regressione > soglia → exit code ≠ 0.

## 7. Rubrica LLM-as-judge — punto chiave

Il judge non deve dare "un voto" generico. Deve valutare **dimensioni separate** con criteri espliciti, restituendo JSON:

```json
{
  "correctness": {"score": 1-5, "reasoning": "..."},
  "completeness": {"score": 1-5, "reasoning": "..."},
  "format_adherence": {"score": 1-5, "reasoning": "..."},
  "hallucination": {"detected": true/false, "evidence": "..."}
}
```

Accorgimenti che dimostrano maturità:
- Usa un modello diverso (e più forte) come judge rispetto a quello valutato.
- **Calibra il judge**: un piccolo set annotato a mano per misurare quanto il judge concorda con te (agreement / Cohen's kappa). Citarlo nel README ti fa sembrare uno che sa che "LLM-as-judge è rumoroso e va validato".
- Temperatura 0 sul judge, output forzato in JSON.

## 8. Fasi di sviluppo

- **Fase 0 — Scaffolding:** repo, schema Pydantic, dataset di esempio (50-100 case su un dominio verticale concreto, non generico).
- **Fase 1 — Runner + metriche deterministiche:** esegui suite, salva run, metriche oggettive.
- **Fase 2 — LLM-as-judge + rubrica:** judge tipizzato, calibrazione su set annotato.
- **Fase 3 — Comparison & regression:** diff tra run, soglie, exit code.
- **Fase 4 — CI:** GitHub Action, report commentato in PR, blocco merge.
- **Fase 5 — Report/dashboard:** grafici costo/qualità/latenza per run, trend storico.

## 9. Deliverable concreti (cosa vede chi guarda il repo)

- README con: problema, architettura, **esempio di report con grafici**, istruzioni run.
- Una PR di esempio dove la CI ha **bloccato un merge** per regressione (screenshot nel README → fortissimo).
- Report comparativo tra ≥2 modelli sullo stesso task.
- Sezione "limiti": rumorosità del judge, costo dell'eval, cosa non copre. La consapevolezza dei limiti segnala seniority.

## 10. Metriche di successo del progetto

- Aggiungere un nuovo test case = modificare 1 file jsonl, zero codice.
- Un run completo è riproducibile da hash del dataset + config.
- La CI gira in < qualche minuto e dà verdetto chiaro pass/fail.
- Agreement judge-umano documentato (anche solo su 30 esempi).

## 11. Collegamenti con gli altri progetti

- **→ 02 Inferenza locale:** l'harness misura la qualità ai vari livelli di quantizzazione. La tabella quant × qualità nasce da qui.
- **→ 03 Osservabilità:** l'harness è uno dei sistemi che instrumenti col tracing.
- **← 04 Synthetic data:** i dati sintetici alimentano/espandono i dataset di eval.

## 12. Estensioni (se vuoi spingere)

- Eval adversariale: jailbreak / prompt injection come test case di sicurezza.
- Multi-judge con voto di maggioranza per ridurre la varianza.
- Costo cumulato per run mostrato in dashboard (lega bene al profilo "costo-conscious").
