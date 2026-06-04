# Progetto 03 — Osservabilità per App LLM

> **Ruolo nel portfolio:** il pezzo production-thinking. Un mini-LangSmith fatto da te.
> **Skill segnalata:** sai strumentare un sistema reale in produzione, non un giocattolo. Ti posiziona come AI engineer e non come "uno che fa demo".

---

## 1. Obiettivo

Tracciare le chiamate LLM di un'applicazione reale: token, costo per request, latenza (p50/p95/p99), replay delle conversazioni, alert su drift e anomalie. Un layer di osservabilità che si mette *sopra* un sistema esistente.

**Mossa chiave:** non costruirlo standalone. Instrumentalo **sopra i Progetti 01 e 02**. Così il portfolio mostra che sai osservare un sistema vero, non un toy isolato.

## 2. Perché segnala (lettura recruiter)

- "Production thinking" è la differenza tra chi fa demo e chi manda roba in produzione. In azienda *nessuno* mette un LLM in prod senza osservabilità.
- Mostra che pensi a costo, latenza, debugging e affidabilità — i problemi veri di chi gestisce LLM su scala.
- Usare **OpenTelemetry** come backbone ti fa segnare punti: è lo standard reale che usano le aziende, non un'invenzione tua.

## 3. Scope — cosa fa

- Intercetta ogni chiamata LLM (decorator/middleware) e cattura: prompt, output, modello, token in/out, costo, latenza, timestamp, esito.
- Aggrega metriche: costo per periodo, latenza p50/p95/p99, error rate, token medi.
- **Replay**: ricostruire e ri-eseguire una conversazione/chiamata passata.
- **Alert**: anomalie su costo, latenza, error rate, drift dei prompt (cambio distribuzione input/output).
- Dashboard di consultazione (anche semplice ma leggibile — stile control-room, on-brand per te).

## 4. Scope — cosa NON fa

- Non è un APM generico — è specifico per LLM (token, costo, prompt, qualità).
- Non rimpiazza l'eval (Progetto 01); semmai lo complementa con segnali di produzione.

## 5. Stack consigliato

| Layer | Scelta | Note |
|---|---|---|
| Strumentazione | **OpenTelemetry** (semantic conventions per GenAI) | il backbone che fa colpo |
| Capture | decorator/middleware Python + context manager | bassa frizione per chi integra |
| Storage trace | SQLite/Postgres + (opz.) ClickHouse per volumi | parti semplice |
| Metriche/alert | regole su query aggregate; soglie configurabili | drift = confronto finestre temporali |
| Dashboard | FastAPI + frontend leggero *oppure* Grafana | Grafana fa molto "production" |
| Standard costi | tabella prezzi per modello → costo calcolato per request | |

## 6. Architettura

```
otel/
  instrumentation.py     # decorator @trace_llm + OTel spans con attributi GenAI
  attributes.py          # mapping a semantic conventions (model, tokens, cost...)
collector/
  ingest.py              # riceve span, normalizza, persiste
  cost.py                # calcolo costo da token + tabella prezzi
store/
  schema.py
  queries.py             # aggregazioni: costo/periodo, percentili, error rate
analysis/
  drift.py               # confronto distribuzioni tra finestre temporali
  anomaly.py             # soglie + regole di alert
  replay.py              # ricostruisce ed (opz.) ri-esegue una chiamata
dashboard/
  app.py                 # viste: timeline, costo, latenza, replay, alert
```

## 7. Cosa cattura ogni span (il modello dati)

```json
{
  "trace_id": "...",
  "timestamp": "...",
  "model": "qwen3-...",
  "prompt": "...",
  "completion": "...",
  "tokens_in": 1234,
  "tokens_out": 567,
  "cost_usd": 0.0021,
  "latency_ms": 842,
  "ttft_ms": 120,
  "status": "ok | error | timeout",
  "metadata": { "app": "eval-harness", "route": "small-model", ... }
}
```

Il campo `metadata.route` chiude il loop col Progetto 02: vedi *quale modello ha scelto il router* e quanto ti costa nel tempo.

## 8. I pezzi che fanno la differenza

- **Drift detection:** confronta distribuzione di lunghezze/token/costo/score tra finestre (es. oggi vs settimana scorsa). Un alert "il costo medio per request è salito del 40%" è esattamente ciò che salva i conti in azienda.
- **Replay:** poter ripescare una chiamata problematica e rieseguirla è il debugging reale degli LLM. Pochi lo implementano.
- **p99, non solo media:** mostrare i percentili (non la media) segnala che capisci come si guarda la latenza in produzione.

## 9. Fasi di sviluppo

- **Fase 0:** decorator di capture + persistenza span base.
- **Fase 1:** calcolo costo + metriche aggregate (costo/periodo, percentili, error rate).
- **Fase 2:** integrazione OpenTelemetry con semantic conventions GenAI.
- **Fase 3:** dashboard con timeline, costo, latenza.
- **Fase 4:** drift + anomaly detection + alert.
- **Fase 5:** replay delle chiamate.
- **Integrazione:** instrumenta Progetto 01 e 02; mostra i loro dati nella dashboard.

## 10. Deliverable concreti

- README con screenshot della dashboard (timeline costi, percentili latenza).
- Esempio di **alert scattato** (drift di costo o latenza) — screenshot nel README.
- Integrazione live con uno degli altri progetti, mostrata.
- Spiegazione di *perché* OpenTelemetry e non un formato custom (segnala consapevolezza degli standard).

## 11. Metriche di successo del progetto

- Strumentare una nuova funzione LLM = aggiungere un decorator, zero altro.
- Overhead di tracing trascurabile (misuralo e dichiaralo).
- Dashboard leggibile: in 10 secondi capisci quanto spendi e dov'è il collo di bottiglia.

## 12. Collegamenti con gli altri progetti

- **← 01 / 02:** sono i sistemi che osservi. L'osservabilità senza un sistema sotto è un guscio vuoto; questo accoppiamento è il punto di forza.
- È il progetto che lega tutto: rende visibile il comportamento in "produzione" di ciò che hai costruito.

## 13. Estensioni (se vuoi spingere)

- Tracciamento di catene/agenti multi-step (span gerarchici padre-figlio).
- Budget enforcement: blocco/alert quando una request supera un costo soglia.
- Export verso Grafana/Prometheus per il look "infra vera".
