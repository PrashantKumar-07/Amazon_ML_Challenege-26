#!/bin/bash
# Rebuilds stage-2 predictions from the stage-1 pipeline caches, then writes the outputs.
# Stage 1 (src.pipeline) is only a candidate filter: pairs with stage-1 p >= 0.003 go to stage 2.
# Portable paths (all overridable):
#   BER_ROOT : repository root            (default: derived from this script's location)
#   BER_W4   : stage-1 work dir (cache/, train/, test/ from src.pipeline)
#   BER_W5   : stage-2 work dir (reduced tables, CE scores, stage-2 models)
#   BER_PY   : python interpreter (default: <root>/.venv/bin/python)
#   OUT      : first arg, output dir for the submission TSVs (default: <root>/output_stage2)
set -e
ROOT="${BER_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
PY="${BER_PY:-$ROOT/.venv/bin/python}"
W4="${BER_W4:-$ROOT/work/stage1}"
W5="${BER_W5:-$ROOT/work/stage2}"
OUT="${1:-$ROOT/output_stage2}"
export BER_ROOT="$ROOT" BER_W4="$W4" BER_W5="$W5"
export HF_HOME="${HF_HOME:-$ROOT/models_hf}" TOKENIZERS_PARALLELISM=false
S2="$ROOT/code/business_entity_resolution/stage2"
mkdir -p "$W5/data" "$W5/train" "$W5/test"
cd "$ROOT"
$PY "$S2/build_reduced.py"                                  # reduced hard-pair tables (needs BER_HARNESS* for the LOCO union)
$PY "$S2/prep_labels.py"                                    # per-record owner array (train only)
$PY "$S2/opfeats.py" train && $PY "$S2/opfeats.py" test     # 18 edit-operation name features
$PY "$S2/xfeat_cap.py" train && $PY "$S2/xfeat_cap.py" test # token features with capped IDF (full candidate tables)
$PY "$S2/extract_cap.py" train && $PY "$S2/extract_cap.py" test
for base in "intfloat/multilingual-e5-small:ce_f" "xlm-roberta-base:xr_f"; do
  M=${base%%:*}; T=${base##*:}; ARGS=""; [ "$M" = "xlm-roberta-base" ] && ARGS="--lr 3e-5 --bs 128"
  for k in 0 1; do o=$((1-k))
    CUDA_VISIBLE_DEVICES=$k CE_BASE=$M $PY "$S2/ce/ce.py" train --sel "fold2==$k" --out $W5/ce/models/$T$k $ARGS
    CUDA_VISIBLE_DEVICES=$k $PY "$S2/ce/ce.py" predict --split train --sel "fold2==$o" --model $W5/ce/models/$T$k --out $W5/data/${T}${k}_train.npy
    CUDA_VISIBLE_DEVICES=$k $PY "$S2/ce/ce.py" predict --split test --sel all --model $W5/ce/models/$T$k --out $W5/data/${T}${k}_test.npy
  done
done
$PY "$S2/ce_feats.py" train && $PY "$S2/ce_feats.py" test && $PY "$S2/xr_feats.py" train && $PY "$S2/xr_feats.py" test
OPS=$(cat "$S2/ops.txt")
CUDA_VISIBLE_DEVICES=0 $PY "$S2/final.py" --tag capop --xname Xcap --extra $OPS --thr 0.8 --dev cuda:0 --out $W5/out_capop
CUDA_VISIBLE_DEVICES=0 $PY "$S2/final.py" --tag capxr --xname Xcap --extra $OPS,ce,rg_ce,rr_ce,xr,rg_xr,rr_xr --thr 0.75 --dev cuda:0 --out $W5/out_capxr
$PY - <<PY
import numpy as np, pandas as pd
from pathlib import Path
import os
D = Path(os.environ["BER_W5"]) / "data"
a = pd.read_parquet(f"{D}/test_pred_capop.parquet"); b = pd.read_parquet(f"{D}/test_pred_capxr.parquet")
assert np.array_equal(a.ri.values, b.ri.values) and np.array_equal(a.si.values, b.si.values)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
p = 1 / (1 + np.exp(-(0.6 * lg(a.p.values) + 0.4 * lg(b.p.values))))
pd.DataFrame({"ri": a.ri.values, "si": a.si.values, "p": p.astype(np.float32)}).to_parquet(f"{D}/test_pred_blendxr.parquet", index=False)
PY
# countries seen in training -> capxr @0.75; countries absent from training -> 0.6*capop + 0.4*capxr @0.8
$PY "$S2/combine7.py" "$OUT" 0.8 blendxr capxr
# Final v8 file: seen-country blend + unseen house-number gate (see README.md):
# $PY "$S2/combine8b.py" "$OUT" v8inus 0.75 blend64 0.8 0.65 0.95 0.5
