"""v5: stage-2 XGBoost on the v4-filtered hard pairs (v4 p>=0.003; v4 p is a filter only, never a feature),
5 folds grouped by S1, test = mean of the fold models; decision + writer from the production pipeline.
python final.py --extra a,b,... --thr 0.8 --out /data/nishant/Nishant/Prashant/AmazonML26/output_v5"""
import argparse, json, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0, "/data/nishant/Nishant/Prashant/AmazonML26/code/business_entity_resolution")
sys.path.insert(0, "/data/nishant/Nishant/Prashant/AmazonML26/work_v5")
from src.pipeline import write_outputs
from exp import load, fit, score, T, log

D = Path("/data/nishant/Nishant/Prashant/AmazonML26/work_v5/data")
W4 = Path("/data/nishant/Nishant/Prashant/AmazonML26/work_v4")

ap = argparse.ArgumentParser()
ap.add_argument("--extra", default=""); ap.add_argument("--tag", default="v5"); ap.add_argument("--drop", default="")
ap.add_argument("--thr", type=float, default=0.8); ap.add_argument("--out", required=True)
ap.add_argument("--dev", default="cuda:0"); ap.add_argument("--depth", type=int, default=10)
ap.add_argument("--mcw", type=float, default=5); ap.add_argument("--xname", default="X")
a = ap.parse_args()
extra = [e for e in a.extra.split(",") if e]
m, X, names = load(extra, set(a.drop.split(",")), a.xname)
ri, si, y, p4, f5 = m["ri"], m["si"], m["y"].astype(np.float32), m["p4"], m["fold5"]
params = dict(max_depth=a.depth, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=a.mcw, reg_lambda=1.0)
in_rows = p4 >= T
mt = np.load(D / "test_meta.npz", allow_pickle=True)
Xt = np.load(D / f"test_{a.xname}.npy", mmap_mode="r")
cols = json.load(open(D / "feat_cols.json"))
keep = [i for i, c in enumerate(cols) if c not in set(a.drop.split(","))]
Xt = np.asarray(Xt[:, keep])
if extra:
    Xt = np.hstack([Xt] + [np.load(D / f"extra_{e}_test.npy").astype(np.float32)[:, None] for e in extra])
assert Xt.shape[1] == X.shape[1]
oof = np.zeros(len(y), np.float32); pt = np.zeros(len(Xt), np.float64); its = []
mdir = Path("/data/nishant/Nishant/Prashant/AmazonML26/work_v5/models_final") / a.tag; mdir.mkdir(parents=True, exist_ok=True)
for k in range(5):
    b, it = fit(X, y, si, np.flatnonzero((f5 != k) & in_rows), params, a.dev); its.append(it)
    ev = np.flatnonzero((f5 == k) & in_rows)
    oof[ev] = b.inplace_predict(X[ev]); pt += b.inplace_predict(Xt) / 5
    b.save_model(str(mdir / f"xgb_s2_fold{k}.ubj"))
    log(f"fold {k}: {it} iterations")
json.dump({"features": names, "params": params, "iterations": its, "thr": a.thr, "filter_T": T},
          open(mdir / "config.json", "w"), indent=1)
true_si = np.load(D / "true_si.npy"); cty_s1 = np.load(D / "s1_cty.npy", allow_pickle=True)
print(score(ri, si, oof, true_si, cty_s1).round(5).to_string(), flush=True)
pred = pd.DataFrame({"ri": mt["ri"], "si": mt["si"], "p": pt.astype(np.float32)})
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
pp = D / f"test_pred_{a.tag}.parquet"; pred.to_parquet(pp, index=False)
write_outputs(W4 / "cache", W4 / "test" / "cands.parquet", pp, {"rule": "threshold", "t": a.thr}, out)
r = subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", str(out / "matching_results.tsv"),
                    "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", "dataset/test"],
                   cwd="/data/nishant/Nishant/Prashant/AmazonML26/student_resource", capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-2000:])
