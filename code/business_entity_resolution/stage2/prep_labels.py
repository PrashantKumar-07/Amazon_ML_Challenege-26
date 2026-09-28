import numpy as np, polars as pl, pandas as pd
W = "work_v4"
s1 = pl.read_parquet(f"{W}/cache/train_s1.parquet", columns=["id", "country"]).to_pandas()
rec = pl.concat([pl.read_parquet(f"{W}/cache/train_s{k}.parquet", columns=["id"]) for k in (2, 3)])["id"].to_list()
rid = {x: i for i, x in enumerate(rec)}; sid = {x: i for i, x in enumerate(s1.id)}
gt = pd.read_csv("student_resource/dataset/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
ts = np.full(len(rec), -1, np.int32)
for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids):
    if m:
        for r in m.split(","): ts[rid[r]] = sid[s]
np.save("work_v5/data/true_si.npy", ts); np.save("work_v5/data/s1_cty.npy", s1.country.values.astype(str))
print(len(ts), (ts >= 0).mean())
