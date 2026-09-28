"""Stage-2 experiments on the reduced hard-pair table (v4 p>=0.003 filter; v4 p is never a feature).
Views (all scored with the exact macro F0.5 over ALL train S1; pairs outside the table have p=0):
  in    : 5-fold by S1 hash, rows with v4 OOF p>=T, out-of-fold predictions
  clone : 'in' predictions + each unmatched record duplicated w.p. 0.9 (test-like distractor density)
  loco  : train on one country (its in-country rows), predict the other on rows with LOCO stage-1 p>=T
python exp.py --tag T [--drop feat,...] [--extra ce_a,...] [--views in,loco] [--dev cuda:0]"""
import argparse, json, sys, time
import numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0, "/data/nishant/Nishant/Prashant/AmazonML26/code/business_entity_resolution")
from src.decide import best_per_record, assign_threshold, score_rows

D = "/data/nishant/Nishant/Prashant/AmazonML26/work_v5/data"
T = 0.003
THRS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
T0 = time.time()


def log(*a):
    print(*a, f"[{time.time() - T0:.0f}s]", flush=True)


def load(extra=(), drop=(), xname="X"):
    m = dict(np.load(f"{D}/train_meta.npz", allow_pickle=True))
    cols = json.load(open(f"{D}/feat_cols.json"))
    X = np.load(f"{D}/train_{xname}.npy", mmap_mode="r")
    keep = [i for i, c in enumerate(cols) if c not in drop]
    names = [cols[i] for i in keep]
    X = np.asarray(X[:, keep]) if len(keep) < len(cols) else np.asarray(X)
    ex = [np.load(f"{D}/extra_{e}_train.npy").astype(np.float32) for e in extra]
    if ex:
        X = np.hstack([X] + [e[:, None] for e in ex]); names += list(extra)
    return m, X, names


def score(ri, si, p, true_si, cty_s1, thrs=THRS):
    n = len(cty_s1)
    single = np.bincount(true_si[true_si >= 0], minlength=n) == 0
    m = p >= min(thrs)
    best = best_per_record(pd.DataFrame({"ri": ri[m], "si": si[m], "p": p[m]}))
    out = []
    for t in thrs:
        a = assign_threshold(best, t)
        f = score_rows(a.ri.values, a.si.values, true_si, n)
        r = {"thr": t, "F": f.mean(), "single": f[single].mean(), "matched": f[~single].mean()}
        for c in np.unique(cty_s1):
            r[c] = f[cty_s1 == c].mean()
        out.append(r)
    return pd.DataFrame(out)


def fit(X, y, si, rows, params, dev, rounds=3000):
    h = ((si[rows].astype(np.uint64) * np.uint64(40503)) % np.uint64(1000)).astype(np.int32)
    va = h < 50
    tr, vr = rows[~va], rows[va]
    dtr = xgb.QuantileDMatrix(X[tr], label=y[tr], max_bin=256)
    dva = xgb.QuantileDMatrix(X[vr], label=y[vr], ref=dtr)
    p = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device=dev, max_bin=256, **params)
    b = xgb.train(p, dtr, num_boost_round=rounds, evals=[(dva, "es")], early_stopping_rounds=50, verbose_eval=False)
    return b[: b.best_iteration + 1], b.best_iteration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--drop", default=""); ap.add_argument("--extra", default="")
    ap.add_argument("--views", default="in,loco"); ap.add_argument("--xname", default="X"); ap.add_argument("--only0", action="store_true"); ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--depth", type=int, default=10); ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--mcw", type=float, default=5); ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--colsample", type=float, default=0.8)
    ap.add_argument("--train_rows", default="in", help="in: v4-filtered rows; union: + LOCO-only hard rows")
    ap.add_argument("--qnorm", default="", help="replace|both: per-country rank-normalised features")
    ap.add_argument("--mono", default="", help="json file {feature: +1/-1}")
    ap.add_argument("--loco_extra", default="", help="comma list of extra cols to REPLACE for the evaluated country in loco view, e.g. ce_a:ce_loco")
    a = ap.parse_args()
    extra = [e for e in a.extra.split(",") if e]
    m, X, names = load(extra, set(a.drop.split(",")), a.xname)
    log(f"{a.tag}: X {X.shape}")
    ri, si, y, p4, pl_, cty, f5 = m["ri"], m["si"], m["y"].astype(np.float32), m["p4"], m["ploco"], m["cty"], m["fold5"]
    true_si = np.load(f"{D}/true_si.npy"); cty_s1 = np.load(f"{D}/s1_cty.npy", allow_pickle=True)
    params = dict(max_depth=a.depth, learning_rate=a.lr, subsample=0.8, colsample_bytree=a.colsample,
                  min_child_weight=a.mcw, reg_lambda=a.lam)
    if a.mono:
        mono = json.load(open(a.mono))
        params["monotone_constraints"] = "(" + ",".join(str(int(mono.get(c, 0))) for c in names) + ")"
    in_rows = p4 >= T
    res = []
    Xl = X
    if a.qnorm:
        from concurrent.futures import ThreadPoolExecutor
        def qn(rowsets):
            Q = np.full(X.shape, np.nan, np.float32)
            def col(j):
                for rows in rowsets:
                    v = X[rows, j]; ok = ~np.isnan(v); r = rows[ok]; vv = v[ok]
                    if len(vv) == 0: continue
                    o = np.argsort(vv, kind="stable"); rk = np.empty(len(vv), np.float32); rk[o] = np.arange(len(vv), dtype=np.float32)
                    # ties -> mean rank via unique inverse
                    u, inv = np.unique(vv, return_inverse=True)
                    mr = np.bincount(inv, weights=rk) / np.bincount(inv)
                    Q[r, j] = (mr[inv] / max(len(vv) - 1, 1)).astype(np.float32)
            with ThreadPoolExecutor(16) as ex: list(ex.map(col, range(X.shape[1])))
            return Q
        cs = np.unique(cty)
        Qin = qn([np.flatnonzero(in_rows & (cty == c)) for c in cs])
        Qlo = qn([np.flatnonzero((pl_ >= T) & (cty == c)) for c in cs])
        if a.qnorm == "both":
            X = np.hstack([X, Qin]); Xl = np.hstack([Xl, Qlo]); names = names + ["q_" + n for n in names]
        else:
            X, Xl = Qin, Qlo
        log("qnorm done", X.shape)
    if "in" in a.views or "clone" in a.views:
        p = np.zeros(len(y), np.float32); its = []
        for k in ([0] if a.only0 else range(5)):
            tr = np.flatnonzero((f5 != k) & (in_rows if a.train_rows == "in" else np.ones(len(y), bool)))
            b, it = fit(X, y, si, tr, params, a.dev); its.append(it)
            import os; os.makedirs(f"{D}/models/{a.tag}", exist_ok=True); b.save_model(f"{D}/models/{a.tag}/in_fold{k}.ubj")
            ev = np.flatnonzero((f5 == k) & in_rows)
            p[ev] = b.inplace_predict(X[ev])
        log("in: iterations", its)
        np.save(f"{D}/pred_{a.tag}_in.npy", p)
        r = score(ri, si, p, true_si, cty_s1); r.insert(0, "view", "in"); res.append(r)
        # clone09: every unmatched record duplicated w.p. 0.9 (same rows, fresh record id)
        rng = np.random.default_rng(12345)
        nrec = int(ri.max()) + 1
        unm = true_si < 0
        cl = unm & (rng.random(len(true_si)) < 0.9)
        dup = np.flatnonzero(cl[ri] & (p >= 0.5))
        ri2 = np.r_[ri, ri[dup] + nrec]; si2 = np.r_[si, si[dup]]; p2 = np.r_[p, p[dup]]
        ts2 = np.r_[true_si, np.full(nrec, -1, true_si.dtype)]
        r = score(ri2, si2, p2, ts2, cty_s1); r.insert(0, "view", "clone"); res.append(r)
    if "loco" in a.views:
        p = np.zeros(len(y), np.float32)
        if a.loco_extra:
            Xl = X.copy()
            for pair in a.loco_extra.split(","):
                c, rep = pair.split(":")
                Xl[:, names.index(c)] = np.load(f"{D}/extra_{rep}_train.npy")
        for c in ["India", "US"]:
            tr = np.flatnonzero((cty != c) & (in_rows if a.train_rows == "in" else np.ones(len(y), bool)))
            b, it = fit(X, y, si, tr, params, a.dev)
            ev = np.flatnonzero((cty == c) & (pl_ >= T))
            p[ev] = b.inplace_predict(Xl[ev]); log(f"loco -> {c}: {it} iterations")
        np.save(f"{D}/pred_{a.tag}_loco.npy", p)
        r = score(ri, si, p, true_si, cty_s1); r.insert(0, "view", "loco"); res.append(r)
    res = pd.concat(res); res.insert(0, "tag", a.tag)
    print(res.round(5).to_string(), flush=True)
    res.to_csv(f"/data/nishant/Nishant/Prashant/AmazonML26/work_v5/res_{a.tag}.csv", index=False)


if __name__ == "__main__":
    main()
