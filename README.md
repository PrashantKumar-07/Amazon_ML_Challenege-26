# Amazon ML Challenge 2026 — Business Entity Resolution

Final file: `output/matching_results.tsv` (public LB **0.990**).
`output/candidate_pairs.tsv` (2.3 GB) is excluded from git (GitHub 100 MB
limit); the pipeline regenerates it (`src/blocking.py`).

```
├── output/
│   └── matching_results.tsv        # final matches (leaderboard file, v8)
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # stage-1 pipeline (blocking → 89 feats → XGBoost → thr)
│       ├── stage2/                 # stage-2 rescoring + v8 decision (113 feats, CE, hn-gate)
│       ├── artifacts/              # stage-1 models + token map + decision (inference w/o retraining)
│       ├── tests/                  # pytest suite (unit + end-to-end with extreme cases)
│       ├── README.md               # setup, quick start (inference), full retraining, hardware
│       └── requirements.txt        # pinned dependencies
├── Documentation_template.md       # methodology write-up
└── ML_problem_statement.pdf        # challenge problem statement
```

**Approach:** learned cross-script token map → exact char-3gram TF-IDF blocking
on GPU (record → top-10 S1, same country label) → 89 string/context/token
features → XGBoost stage-1 (5-fold OOF, early stopping, filter p ≥ 0.003) →
stage-2 XGBoost on 113 features (89 capped + 18 edit-operation + 6
cross-encoder: e5-small + xlm-roberta-base, both MIT) → record assignment:
seen countries 0.5·(capce+capxr) @ 0.75; unseen countries 0.6·capop+0.4·capxr
with a house-number gate (shared → 0.65, disjoint → 0.95, else 0.8; inserted
"and" at same number → 0.5).

**Validation (train macro F0.5):** stage-1 OOF **0.9872**, test-like density
**0.9863**, unseen country **0.9521**; stage-2 in-country **0.9904**
(+0.0002 with the seen blend). Blocking recall 98.9% (ceiling 0.9966).
Selection on density simulation + leave-one-country-out, not plain OOF.

**LB lineage:** v1 0.969 → v4 0.979 → v5 0.983 → v6 0.985 → v7 0.984
(rejected) → **v8 0.990** (shipped); v9 0.984 (rejected), v10 = v8 + 1 pair.

**Teammates:** dataset (2.4 GB) and `models_hf/` cache are not in git — get the
dataset from the challenge portal, then follow the *Quick start* in
`code/business_entity_resolution/README.md` (stage-1 inference from shipped
models; `stage2/README.md` for the full v8 rebuild).

See `code/business_entity_resolution/README.md` to reproduce and
`Documentation_template.md` for details.
