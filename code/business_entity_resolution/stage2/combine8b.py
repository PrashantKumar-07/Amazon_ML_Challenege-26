"""v8 writer. Seen countries: SEEN_TAG @ SEEN_T (record-best assignment).
Unseen countries: UNSEEN_TAG, record-best S1 by p, then a house-number-gated threshold:
  shared number token (both addresses have digits and share >=1 number) -> T_EQ
  both have digits but share none                                        -> T_NE
  otherwise (a side without digits)                                      -> UNSEEN_T
Motivation (EXPERIMENT_LOG, LB-grounded): both LB moves on the unseen country agree that same-number pairs with a
varied name are true and different-number pairs are the distractors (v5->v6 +, v6->v7 -).
python combine8b.py OUT SEEN_TAG SEEN_T UNSEEN_TAG UNSEEN_T T_EQ T_NE"""
import shutil, subprocess, sys
import os
from pathlib import Path
import numpy as np, pandas as pd, polars as pl
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.decide import best_per_record, assign_threshold
from src.io_utils import MATCH_HEADER, write_id_lists

ROOT = Path(os.environ.get("BER_ROOT", Path(__file__).resolve().parent.parent.parent.parent))
W4 = Path(os.environ.get("BER_W4", ROOT / "work" / "stage1"))
W5 = Path(os.environ.get("BER_W5", ROOT / "work" / "stage2"))
D = W5 / "data"
OUT = Path(sys.argv[1]); SEEN_TAG, SEEN_T, UNSEEN_TAG = sys.argv[2], float(sys.argv[3]), sys.argv[4]
UNSEEN_T, T_EQ, T_NE = float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7])
T_AND = float(sys.argv[8]) if len(sys.argv) > 8 and not sys.argv[8].startswith("--") else None
OUT.mkdir(parents=True, exist_ok=True)
train_countries = set(pl.read_parquet(W4 / "cache/train_s1.parquet", columns=["country"])["country"].unique().to_list())
s1 = pd.read_parquet(W4 / "cache/test_s1.parquet", columns=["id", "country", "addr_digits", "name"])
recs = pd.concat([pd.read_parquet(W4 / f"cache/test_s{s}.parquet", columns=["id", "addr_digits", "name"]) for s in (2, 3)], ignore_index=True)
rec_ids = recs.id.values
seen_s1 = s1.country.isin(train_countries).values
print("train countries:", sorted(train_countries), "| test S1 from unseen countries:", int((~seen_s1).sum()))
numset = lambda s: frozenset(t.lstrip("0") or "0" for t in str(s).split())


def assign(tag, t, want_seen):
    pr = pd.read_parquet(D / f"test_pred_{tag}.parquet")
    pr = pr[seen_s1[pr.si.values.astype(np.int64)] == want_seen]
    return pr


seen = assign(SEEN_TAG, SEEN_T, True)
seen = assign_threshold(best_per_record(seen[seen.p.values >= SEEN_T]), SEEN_T)
un = assign(UNSEEN_TAG, 0, False)
tmin = min(UNSEEN_T, T_EQ, T_NE, T_AND or 1.0)
b = best_per_record(un)          # record-best over ALL candidates (ownership unchanged by the gate)
b = b[b.p.values >= tmin].reset_index(drop=True)
S = [numset(x) for x in s1.addr_digits.values[b.si.values.astype(np.int64)]]
R = [numset(x) for x in recs.addr_digits.values[b.ri.values.astype(np.int64)]]
both = np.array([bool(x) and bool(y) for x, y in zip(S, R)])
share = np.array([bool(x & y) for x, y in zip(S, R)])
thr = np.where(both & share, T_EQ, np.where(both & ~share, T_NE, UNSEEN_T))
if T_AND is not None:
    # '&' normalises to 'and' (normalize.py:27); an inserted 'and' is a train distractor marker, which the
    # unseen country's '& Fils'/'& Associes' suffixes inherit. Same-number pairs with that insertion only -> T_AND.
    sn = s1.name.values[b.si.values.astype(np.int64)]; rn = recs.name.values[b.ri.values.astype(np.int64)]
    ins_and = np.array([(("and" in str(r).split()) and ("and" not in str(q).split())) or (("et" in str(r).split()) and ("and" in str(q).split()) and ("and" not in str(r).split())) for r, q in zip(rn, sn)])
    fx = both & share & ins_and
    thr = np.where(fx, np.minimum(thr, T_AND), thr)
    print(f"and-fix: {int(fx.sum()):,} same-number pairs with an inserted 'and'; newly accepted {int((fx & (b.p.values >= T_AND) & (b.p.values < T_EQ)).sum()):,}")
un_a = b[b.p.values >= thr]
base = b[b.p.values >= UNSEEN_T]
print(f"unseen: base@{UNSEEN_T} {len(base):,} -> gated {len(un_a):,} (same-number {int((both & share).sum()):,}, "
      f"diff-number {int((both & ~share).sum()):,} in candidates >= {tmin})")
a = pd.concat([seen, un_a], ignore_index=True)
assert not a.ri.duplicated().any()
order = np.argsort(a.si.values, kind="stable"); si, ri = a.si.values[order].astype(np.int64), a.ri.values[order].astype(np.int64)
cut = np.flatnonzero(np.r_[True, si[1:] != si[:-1]])
groups = {s1.id.values[si[x]]: rec_ids[ri[x:y]] for x, y in zip(cut, np.r_[cut[1:], len(si)])}
write_id_lists(OUT / "matching_results.tsv", MATCH_HEADER, s1.id.values, groups)
cp = OUT / "candidate_pairs.tsv"
if not cp.exists():
    fb = ROOT / "output" / "candidate_pairs.tsv"
    if fb.exists():
        shutil.copy2(fb, cp)
    else:
        print(f"note: {cp} not written; copy the stage-1 candidate file here before validating")
nm = np.bincount(si, minlength=len(s1))
for c in sorted(s1.country.unique()):
    m = s1.country.values == c
    print(f"{c}: matches/S1 {nm[m].mean():.4f}, empty {np.mean(nm[m] == 0):.2%}")
print(f"wrote {len(a):,} matches for {len(s1):,} S1", flush=True)
if "--novalidate" not in sys.argv:
    r = subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", str(OUT / "matching_results.tsv"),
                        "--candidate", str(cp), "--test-dir", "dataset/test", "--check-ids"],
                       cwd=str(ROOT / "student_resource"), capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-1500:])
