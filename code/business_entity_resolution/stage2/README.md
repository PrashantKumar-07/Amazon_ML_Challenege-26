# Stage-2 (v5–v8; final file `output/matching_results.tsv`, public LB 0.990)

Prerequisite: a stage-1 pipeline run (`python -m src.pipeline --mode full`,
which provides caches, candidates, 89 features and stage-1 test/OOF predictions).
Stage-2 re-scores the hard pairs only. Design and every validation number are in
`work/EXPERIMENT_LOG.md` (not in git; kept locally).

- **Stage 1 (src, filter only):** XGBoost on 89 features. Pairs with stage-1
  p >= 0.003 go to stage-2 (keeps 99.986% of true pairs in-country,
  99.76% on the leave-one-country-out proxy). Stage-1 p is never a feature.
- **Stage-2 features (113):** the 89 stage-1 features (IDF capped at log 1e5) +
  18 edit-operation name features (`ops.txt`: inserted/deleted/substituted token
  counts typed by country-invariant frequency percentiles, positions, novelty) +
  6 cross-encoder features: `multilingual-e5-small` (MIT, 118M) and
  `xlm-roberta-base` (MIT, 278M) logits with record-side gap/rank context.
  Each CE is fine-tuned on 2 S1-hash folds for out-of-fold features
  (1 epoch); test uses the mean of the two fold logits. Total neural params
  ~0.4B (limit 8B); XGBoost is Apache-2.0.
- **Stage-2 models:** 5-fold XGBoost grouped by S1 (same hyper-params as
  stage-1). `capop` = no-CE (89+18); `capxr` = full (89+18+6).
- **Final v8 decision (`combine8b.py`):**
  - training-seen countries: 0.5 logit blend of capce+capxr (`v8inus`) @ 0.75;
  - unseen countries (France): 0.6·capop + 0.4·capxr (`blend64`) with a
    house-number gate — shared number token → 0.65, both numbered but
    disjoint → 0.95, else 0.8; same-number pairs whose record inserts "and"
    (from `&`, see `src/normalize.py`) → 0.5.
  - Command: `python combine8b.py OUT v8inus 0.75 blend64 0.8 0.65 0.95 0.5`
- **Lineage (public LB):** v4 0.979 → v5 0.983 (hybrid CE-seen / no-CE-unseen)
  → v6 0.985 (unseen logit blend) → v7 0.984 (rejected: XLM-R for seen too) →
  **v8 0.990** (seen blend + unseen house-number gate). v9 0.984 (stronger gate,
  rejected — beyond the LOCO optimum); v10 = v8 + 1 pair (ownership offset has
  no effect).
- **Reproduce:** `./run_v7.sh` rebuilds the v6/v7 predictions from the stage-1
  caches (~6–8 h on 2× L40S, CE steps dominate); then run the v8 blend + gate
  (`combine8b.py`). CE checkpoints (~6 GB) and `work/` tables are not in git;
  retrain from the HF bases (see `ce/ce.py`, `final.py`, `exp.py`).
- **Known limitations:** absolute paths; CE folds (2) are not nested inside the
  stage-2 folds (5); bf16 CE logits saturate near 10; the gate thresholds were
  set on the LOCO proxy + LB moves, not on plain OOF.
