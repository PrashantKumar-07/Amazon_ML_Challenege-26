"""Build the reduced hard-pair tables (stage-2 universe) for train and test.
train rows: v4 OOF p >= T  OR  v4-config LOCO p >= T  (LOCO preds come from the harness table, joined on (ri, si)).
test rows : v4 test p >= T.  Stage-1 (v4) probability is used ONLY as a filter, never as a feature."""
import sys, numpy as np, polars as pl, pyarrow.parquet as pq, json
sys.path.insert(0, "code/business_entity_resolution")
from src.model import aligned_batches, fold_of
T = 0.003
W, O = "work_v4", "work_v5/data"
cols = json.load(open("work_v5/data/cols.json")) if False else None
import xgboost as xgb
FEATS = xgb.Booster(model_file=f"{W}/models/xgb_fold0.ubj").feature_names
json.dump(FEATS, open(f"{O}/feat_cols.json", "w"))

def extract(split, mask):
    rows = np.flatnonzero(mask); out = np.empty((len(rows), len(FEATS)), np.float32)
    ri = np.empty(len(rows), np.int32); si = np.empty(len(rows), np.int32); y = np.zeros(len(rows), np.int8)
    pos = 0; start = 0
    need = set(FEATS) | {"y"}
    for df in aligned_batches([f"{W}/{split}/feats.parquet", f"{W}/{split}/xfeats.parquet"], columns=need):
        m = mask[start:start + len(df)]; k = int(m.sum())
        if k:
            d = df[m]
            out[pos:pos + k] = d[FEATS].to_numpy(np.float32)
            ri[pos:pos + k] = d.ri.values; si[pos:pos + k] = d.si.values
            if "y" in d: y[pos:pos + k] = d.y.values
            pos += k
        start += len(df); print(split, start, pos, flush=True)
    assert pos == len(rows)
    return rows, out, ri, si, y

# train
o = pl.read_parquet(f"{W}/train/oof.parquet", columns=["ri", "si", "p"]).with_row_index("row")
TD = "work/agents/tuner/data"
lp = np.load("work/exp2/preds/loco_v2_es_base.npy")
h = pl.DataFrame({"ri": np.load(f"{TD}/ri_train.npy"), "si": np.load(f"{TD}/si_train.npy"), "p_loco": lp})
o = o.join(h, on=["ri", "si"], how="left")
print("v4 rows without a harness LOCO p:", o["p_loco"].null_count())
o = o.sort("row")
p4 = o["p"].to_numpy(); pl_ = o["p_loco"].fill_null(1.0).to_numpy()   # unmatched keys (near-ties) kept conservatively
mask = (p4 >= T) | (pl_ >= T)
print(f"train reduced rows: {mask.sum():,} (in-country {int((p4>=T).sum()):,}, loco {int((pl_>=T).sum()):,})")
rows, X, ri, si, y = extract("train", mask)
cty = pl.read_parquet(f"{W}/cache/train_s1.parquet", columns=["country"])["country"].to_numpy()
np.savez(f"{O}/train_meta.npz", rows=rows, ri=ri, si=si, y=y, p4=p4[rows].astype(np.float32), ploco=pl_[rows].astype(np.float32),
         cty=cty[si], fold2=fold_of(si, 2), fold5=fold_of(si, 5))
np.save(f"{O}/train_X.npy", X); del X
# test
t = pl.read_parquet(f"{W}/test/pred.parquet", columns=["p"])["p"].to_numpy()
mask = t >= T; print(f"test reduced rows: {mask.sum():,}")
rows, X, ri, si, _ = extract("test", mask)
cty = pl.read_parquet(f"{W}/cache/test_s1.parquet", columns=["country"])["country"].to_numpy()
np.savez(f"{O}/test_meta.npz", rows=rows, ri=ri, si=si, p4=t[rows].astype(np.float32), cty=cty[si])
np.save(f"{O}/test_X.npy", X)
print("done")
