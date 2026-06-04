# Progetto 04 — Synthetic Data Generation Pipeline

> **Ruolo nel portfolio:** il jolly. Opzionale: i primi tre coprono già il profilo AI engineer. Aggiungilo se hai tempo o vuoi distinguerti su un tema caldo.
> **Skill segnalata:** data engineering applicato all'IA. Argomento tecnicamente sottile e molto richiesto.

---

## 1. Obiettivo

Generare dataset sintetici di **qualità** per training ed eval, con controllo di diversità, deduplica, filtering automatico della qualità e validazione strutturata. Non "chiedo a un LLM 1000 esempi e li butto in un file" — ma una pipeline che produce dati *controllati, diversificati e validati*.

**Collegamento:** i dati sintetici alimentano ed espandono i dataset di eval del **Progetto 01**. Lì si chiude il loop.

## 2. Perché segnala (lettura recruiter)

- I dati sintetici sono uno dei temi più caldi (scarsità di dati reali, privacy, costo dell'annotazione umana).
- La parte difficile non è generare — è **garantire qualità e diversità**. Mostrare che lo sai è raro e segnalante.
- È data engineering vero: dedup, filtering, validazione, metriche di distribuzione.

## 3. Scope — cosa fa

- Generazione guidata da **specifica** (schema target, distribuzione voluta, edge case richiesti).
- **Controllo diversità:** evita il collasso su pochi pattern ripetitivi.
- **Deduplica:** esatta + semantica (near-duplicate).
- **Quality filtering automatico:** scarta esempi malformati/banali/errati.
- **Validazione strutturata:** ogni esempio rispetta uno schema (Pydantic/JSON Schema).
- Report sulla distribuzione del dataset prodotto.

## 4. Scope — cosa NON fa

- Non addestra modelli (consuma i dati a valle, es. nell'eval o in un eventuale fine-tuning).
- Non è un tool generico "scrivi testo" — è orientato a *dataset strutturati e misurabili*.

## 5. Stack consigliato

| Layer | Scelta | Note |
|---|---|---|
| Generazione | LLM via API o **modello locale (Progetto 02)** | usare il tuo serving chiude un altro loop |
| Validazione | **Pydantic / JSON Schema** | ogni esempio tipizzato |
| Dedup esatta | hash | banale ma necessaria |
| Dedup semantica | **MinHash/LSH** *o* embedding + soglia coseno | il pezzo tecnico |
| Diversità | clustering su embedding + copertura | misura, non a occhio |
| Quality filter | LLM-as-judge (riusa rubrica del Prog. 01) + euristiche | |
| Report | distribuzioni, cluster, tasso di scarto | grafici |

## 6. Architettura

```
spec/
  schema.py              # schema target dell'esempio (Pydantic)
  plan.py                # distribuzione voluta, edge case, quote per categoria
generate/
  generator.py           # prompting strutturato → esempi grezzi
  seed.py                # seed/diversificatori (personas, parametri, scenari)
quality/
  validate.py            # schema validity
  dedup.py               # esatta + semantica (MinHash/embedding)
  filter.py              # quality filtering (judge + euristiche)
  diversity.py           # clustering, copertura, metriche
report/
  stats.py               # distribuzione, tasso di scarto, cluster
pipeline.py              # orchestrazione end-to-end
```

## 7. Il flusso (e dove sta la difficoltà)

1. **Spec & plan:** definisci schema e *distribuzione voluta* (quante per categoria, quali edge case). Generare a caso produce dataset sbilanciati: il piano è il pezzo intelligente.
2. **Generazione diversificata:** semina con personas/scenari/parametri variabili per evitare il mode collapse (l'LLM lasciato libero ripete pochi pattern).
3. **Validazione strutturale:** scarta ciò che non rispetta lo schema.
4. **Deduplica:** esatta + semantica. La near-dedup è dove dimostri competenza (MinHash/LSH o embedding).
5. **Quality filtering:** judge + euristiche scartano banalità/errori.
6. **Misura diversità:** clustering sugli embedding → copertura dello spazio. **Questo numero va nel README.**
7. **Report:** distribuzione finale vs voluta, tasso di scarto per stage.

## 8. I pezzi che fanno la differenza

- **Mode collapse esplicitato:** mostrare il *prima/dopo* della diversificazione (cluster prima compressi, poi distribuiti) è una dimostrazione visiva potentissima di competenza.
- **Near-dedup semantica:** non basta l'hash; due esempi "uguali ma riformulati" vanno presi. MinHash/LSH o embedding.
- **Tasso di scarto per stage:** quanto butti a validazione/dedup/filter → racconta l'onestà della pipeline.
- **Closing the loop:** i dati prodotti vengono *davvero usati* nell'eval del Progetto 01.

## 9. Fasi di sviluppo

- **Fase 0:** schema target + generazione grezza funzionante.
- **Fase 1:** validazione strutturale + dedup esatta.
- **Fase 2:** dedup semantica (MinHash/embedding).
- **Fase 3:** quality filtering (judge + euristiche).
- **Fase 4:** diversificazione (seed/personas) + metriche di diversità.
- **Fase 5:** report distribuzione + integrazione con dataset di eval (Prog. 01).

## 10. Deliverable concreti

- README con: schema, pipeline, **grafico diversità prima/dopo**, tasso di scarto per stage.
- Un dataset sintetico finito, validato, usato nel Progetto 01.
- Numero chiaro: *"da N esempi grezzi a M validati, con copertura di K cluster e tasso duplicati < X%"*.
- Sezione limiti: bias del modello generatore, rischio di distillare gli errori del modello, costo.

## 11. Metriche di successo del progetto

- Cambiare dominio = cambiare schema + plan, pipeline invariata.
- Metriche di diversità/dedup quantificate, non "a occhio".
- I dati prodotti migliorano o estendono concretamente l'eval suite.

## 12. Collegamenti con gli altri progetti

- **→ 01 Eval Harness:** alimenta/espande i dataset di test. È il legame principale.
- **← 02 Inferenza locale:** puoi usare il tuo modello locale come generatore (privacy + costo zero).
- **→ 03 Osservabilità:** la generazione massiva è un buon caso da tracciare (costo/token della pipeline).

## 13. Estensioni (se vuoi spingere)

- Generazione di **edge case adversariali** mirati a far fallire un modello (utili per l'eval di sicurezza del Prog. 01).
- Self-instruct / evol-instruct: complessità crescente guidata.
- Confronto qualità di un modellino addestrato/valutato su dati sintetici vs reali.
