#!/bin/bash
# Reproduces output_v7 from the v4 pipeline caches (work_v4: preprocess -> blocking -> features -> xfeatures -> v4 stage-1).
# Stage 1 (v4, src.pipeline) is only a candidate filter: pairs with v4 p >= 0.003 go to stage 2. Paths are absolute (see README_v7.md).
set -e
cd "$(dirname "$0")"; PY=/data/nishant/Nishant/Prashant/AmazonML26/.venv/bin/python; W=/data/nishant/Nishant/Prashant/AmazonML26/work_v5
export HF_HOME=/data/nishant/Nishant/Prashant/AmazonML26/models_hf TOKENIZERS_PARALLELISM=false
mkdir -p $W/data $W/train $W/test
$PY build_reduced.py                                  # reduced hard-pair tables (train 10.47M, test 7.75M)
$PY prep_labels.py                                    # per-record owner array (train only)
$PY opfeats.py train && $PY opfeats.py test           # 18 edit-operation name features
$PY xfeat_cap.py train && $PY xfeat_cap.py test       # token features with capped IDF (full candidate tables)
$PY extract_cap.py train && $PY extract_cap.py test
for base in "intfloat/multilingual-e5-small:ce_f" "xlm-roberta-base:xr_f"; do
  M=${base%%:*}; T=${base##*:}; ARGS=""; [ "$M" = "xlm-roberta-base" ] && ARGS="--lr 3e-5 --bs 128"
  for k in 0 1; do o=$((1-k))
    CUDA_VISIBLE_DEVICES=$k CE_BASE=$M $PY ce/ce.py train --sel "fold2==$k" --out $W/ce/models/$T$k $ARGS
    CUDA_VISIBLE_DEVICES=$k $PY ce/ce.py predict --split train --sel "fold2==$o" --model $W/ce/models/$T$k --out $W/data/${T}${k}_train.npy
    CUDA_VISIBLE_DEVICES=$k $PY ce/ce.py predict --split test --sel all --model $W/ce/models/$T$k --out $W/data/${T}${k}_test.npy
  done
done
$PY ce_feats.py train && $PY ce_feats.py test && $PY xr_feats.py train && $PY xr_feats.py test
OPS=$(cat ops.txt)
CUDA_VISIBLE_DEVICES=0 $PY final.py --tag capop --xname Xcap --extra $OPS --thr 0.8 --dev cuda:0 --out $W/out_capop
CUDA_VISIBLE_DEVICES=0 $PY final.py --tag capxr --xname Xcap --extra $OPS,ce,rg_ce,rr_ce,xr,rg_xr,rr_xr --thr 0.75 --dev cuda:0 --out $W/out_capxr
$PY - <<'PY'
import numpy as np, pandas as pd
D = "/data/nishant/Nishant/Prashant/AmazonML26/work_v5/data"
a = pd.read_parquet(f"{D}/test_pred_capop.parquet"); b = pd.read_parquet(f"{D}/test_pred_capxr.parquet")
assert np.array_equal(a.ri.values, b.ri.values) and np.array_equal(a.si.values, b.si.values)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
p = 1 / (1 + np.exp(-(0.6 * lg(a.p.values) + 0.4 * lg(b.p.values))))
pd.DataFrame({"ri": a.ri.values, "si": a.si.values, "p": p.astype(np.float32)}).to_parquet(f"{D}/test_pred_blendxr.parquet", index=False)
PY
# countries seen in training -> capxr @0.75; countries absent from training -> 0.6*capop + 0.4*capxr @0.8
$PY combine7.py /data/nishant/Nishant/Prashant/AmazonML26/output_v7 0.8 blendxr capxr
