import os
from pathlib import Path
import numpy as np, polars as pl, pandas as pd
ROOT = Path(os.environ.get("BER_ROOT", Path(__file__).resolve().parent.parent.parent.parent))
W4 = Path(os.environ.get("BER_W4", ROOT / "work" / "stage1"))
W5 = Path(os.environ.get("BER_W5", ROOT / "work" / "stage2"))
D = W5 / "data"
s1 = pl.read_parquet(W4 / "cache/train_s1.parquet", columns=["id", "country"]).to_pandas()
rec = pl.concat([pl.read_parquet(W4 / f"cache/train_s{k}.parquet", columns=["id"]) for k in (2, 3)])["id"].to_list()
rid = {x: i for i, x in enumerate(rec)}; sid = {x: i for i, x in enumerate(s1.id)}
gt = pd.read_csv(ROOT / "student_resource/dataset/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
ts = np.full(len(rec), -1, np.int32)
for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids):
    if m:
        for r in m.split(","): ts[rid[r]] = sid[s]
np.save(D / "true_si.npy", ts); np.save(D / "s1_cty.npy", s1.country.values.astype(str))
print(len(ts), (ts >= 0).mean())
