"""Edit-operation name features for the reduced pair tables.
Record-name tokens are aligned to S1-name tokens (exact > fuzzy > abbreviation). Inserted / deleted tokens are
described by country-invariant, label-free statistics learned per split and per country from S1 names only:
  q     : frequency-rank percentile of the token among the country's S1 vocabulary (novel token -> -1)
  last  : share of the token's S1 occurrences in the final position   (legal-form-like)
  first : share of the token's S1 occurrences in the first position   (prefix-like)
python opfeats.py train|test"""
import sys, numpy as np, polars as pl
import os
from pathlib import Path
from collections import Counter
from multiprocessing import Pool
from rapidfuzz.distance import Levenshtein

ROOT = Path(os.environ.get("BER_ROOT", Path(__file__).resolve().parent.parent.parent.parent))
W4 = Path(os.environ.get("BER_W4", ROOT / "work" / "stage1"))
W5 = Path(os.environ.get("BER_W5", ROOT / "work" / "stage2"))
W, D = str(W4), str(W5 / "data")
COLS = ["o_nins", "o_ndel", "o_nsub", "o_ins_novel", "o_ins_qmax", "o_ins_qmin", "o_ins_lastmax", "o_ins_firstmax",
        "o_del_qmax", "o_del_qmin", "o_del_lastmax", "o_del_firstmax", "o_ins_at_end", "o_ins_at_start",
        "o_del_s1last", "o_ins_frac", "o_del_frac", "o_order_swap"]
G = {}


def toks(s):
    return [t for t in (s or "").split() if not t.isdigit()]


def country_stats(names):
    df, last, first = Counter(), Counter(), Counter()
    for nm in names:
        t = toks(nm)
        if not t:
            continue
        for x in set(t):
            df[x] += 1
        last[t[-1]] += 1; first[t[0]] += 1
    vocab = sorted(df, key=lambda x: df[x])
    n = max(len(vocab) - 1, 1)
    # frequency-rank percentile (ties share the lowest rank)
    q, prev, rank = {}, None, 0
    for i, x in enumerate(vocab):
        if df[x] != prev:
            rank, prev = i, df[x]
        q[x] = rank / n
    return {x: (q[x], last[x] / df[x], first[x] / df[x]) for x in df}


def match(a, b):
    if a == b:
        return 3
    if len(a) >= 3 and len(b) >= 3 and Levenshtein.normalized_similarity(a, b) >= 0.75:
        return 2
    s, l = (a, b) if len(a) <= len(b) else (b, a)
    if len(s) >= 2 and s[0] == l[0]:
        it = iter(l)
        if all(ch in it for ch in s):
            return 1
    return 0


def pair(sn, rn, st):
    S, R = toks(sn), toks(rn)
    out = np.full(len(COLS), np.nan, np.float32)
    if not S or not R:
        return out
    used = [False] * len(S); ralign = [-1] * len(R)
    for lvl in (3, 2, 1):
        for i, r in enumerate(R):
            if ralign[i] >= 0:
                continue
            for j, s in enumerate(S):
                if not used[j] and match(r, s) == lvl:
                    used[j] = True; ralign[i] = j; break
    ins = [R[i] for i in range(len(R)) if ralign[i] < 0]
    dele = [S[j] for j in range(len(S)) if not used[j]]
    al = [j for j in ralign if j >= 0]
    swaps = sum(1 for k in range(1, len(al)) if al[k] < al[k - 1])
    nsub = min(len(ins), len(dele))
    def stats(ts):
        v = [st.get(t) for t in ts]
        qs = [x[0] if x else -1.0 for x in v]
        la = [x[1] if x else 0.0 for x in v]
        fi = [x[2] if x else 0.0 for x in v]
        return (max(qs), min(qs), max(la), max(fi)) if ts else (np.nan,) * 4
    iq = stats(ins); dq = stats(dele)
    out[:] = [len(ins), len(dele), nsub, sum(1 for t in ins if t not in st), *iq, *dq,
              float(ralign[-1] < 0), float(ralign[0] < 0), float(not used[-1]),
              len(ins) / len(R), len(dele) / len(S), swaps]
    return out


def work(args):
    lo, hi = args
    si, ri, cty = G["si"][lo:hi], G["ri"][lo:hi], G["cty"][lo:hi]
    s1n, recn, stats = G["s1n"], G["recn"], G["stats"]
    return np.stack([pair(s1n[s], recn[r], stats[c]) for s, r, c in zip(si, ri, cty)])


NREF = 0


def main(split, frac=1.0, tag=""):
    m = np.load(f"{D}/{split}_meta.npz", allow_pickle=True)
    s1 = pl.read_parquet(f"{W}/cache/{split}_s1.parquet", columns=["name", "country"])
    rec = pl.concat([pl.read_parquet(f"{W}/cache/{split}_{t}.parquet", columns=["name"]) for t in ("s2", "s3")])
    G["s1n"] = s1["name"].fill_null("").to_list(); G["recn"] = rec["name"].fill_null("").to_list()
    h = ((np.arange(s1.height, dtype=np.uint64) * np.uint64(2246822519)) % np.uint64(1000000)).astype(np.int64)
    cnt = s1.group_by("country").len().to_pandas().set_index("country")["len"]
    cfrac = np.array([min(1.0, NREF / cnt[c]) if NREF else frac for c in s1["country"].to_list()])
    s1k = s1.filter(pl.Series(h < (cfrac * 1000000)))
    G["stats"] = {c: country_stats(s1k.filter(pl.col("country") == c)["name"].fill_null("").to_list())
                  for c in s1["country"].unique().to_list()}
    G["si"], G["ri"] = m["si"], m["ri"]; G["cty"] = s1["country"].to_numpy()[m["si"]]
    n = len(m["si"]); step = 50_000
    with Pool(96) as p:
        parts = p.map(work, [(i, min(n, i + step)) for i in range(0, n, step)], chunksize=1)
    F = np.vstack(parts)
    for j, c in enumerate(COLS):
        np.save(f"{D}/extra_{c}{tag}_{split}.npy", F[:, j])
    print(split, F.shape, np.nanmean(F, 0).round(3))


if __name__ == "__main__":
    if len(sys.argv) > 4:
        NREF = int(sys.argv[4])
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 1.0, sys.argv[3] if len(sys.argv) > 3 else "")
