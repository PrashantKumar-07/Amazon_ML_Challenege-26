"""v5 writer: S1 whose country label occurs in the training data -> CE-stacked model (capce, thr 0.75);
S1 of a country absent from training -> no-CE model (capop, thr 0.7; selected on leave-one-country-out).
The seen-country set is read from the train S1 table (no country names in code)."""
import shutil, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, polars as pl
sys.path.insert(0, "/data/nishant/Nishant/Prashant/AmazonML26/code/business_entity_resolution")
from src.decide import best_per_record, assign_threshold
from src.io_utils import MATCH_HEADER, write_id_lists

W4 = Path("/data/nishant/Nishant/Prashant/AmazonML26/work_v4"); D = Path("/data/nishant/Nishant/Prashant/AmazonML26/work_v5/data")
OUT = Path(sys.argv[1]); SEEN_T = 0.75; UNSEEN_T = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
UNSEEN_TAG = sys.argv[3] if len(sys.argv) > 3 else "capop"
SEEN_TAG = sys.argv[4] if len(sys.argv) > 4 else "capce"
OUT.mkdir(parents=True, exist_ok=True)
train_countries = set(pl.read_parquet(W4 / "cache/train_s1.parquet", columns=["country"])["country"].unique().to_list())
s1 = pd.read_parquet(W4 / "cache/test_s1.parquet", columns=["id", "country"])
rec_ids = np.concatenate([pd.read_parquet(W4 / f"cache/test_s{s}.parquet", columns=["id"]).id.values for s in (2, 3)])
seen_s1 = s1.country.isin(train_countries).values
print("train countries:", sorted(train_countries), "| test S1 from unseen countries:", int((~seen_s1).sum()))


def assign(tag, t, want_seen):
    pr = pd.read_parquet(D / f"test_pred_{tag}.parquet")
    pr = pr[(seen_s1[pr.si.values] == want_seen) & (pr.p.values >= t)]
    return assign_threshold(best_per_record(pr), t)


a = pd.concat([assign(SEEN_TAG, SEEN_T, True), assign(UNSEEN_TAG, UNSEEN_T, False)], ignore_index=True)
assert not a.ri.duplicated().any()
order = np.argsort(a.si.values, kind="stable"); si, ri = a.si.values[order], a.ri.values[order]
cut = np.flatnonzero(np.r_[True, si[1:] != si[:-1]])
groups = {s1.id.values[si[x]]: rec_ids[ri[x:y]] for x, y in zip(cut, np.r_[cut[1:], len(si)])}
write_id_lists(OUT / "matching_results.tsv", MATCH_HEADER, s1.id.values, groups)
cp = OUT / "candidate_pairs.tsv"
if not cp.exists():
    shutil.copy2("/data/nishant/Nishant/Prashant/AmazonML26/output_v6/candidate_pairs.tsv", cp)
nm = np.bincount(a.si.values, minlength=len(s1))
for c in sorted(s1.country.unique()):
    m = s1.country.values == c
    print(f"{c}: matches/S1 {nm[m].mean():.3f}, empty {np.mean(nm[m] == 0):.2%}")
print(f"wrote {len(a):,} matches for {len(s1):,} S1")
r = subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", str(OUT / "matching_results.tsv"),
                    "--candidate", str(cp), "--test-dir", "dataset/test", "--check-ids"],
                   cwd="/data/nishant/Nishant/Prashant/AmazonML26/student_resource", capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-1500:])
