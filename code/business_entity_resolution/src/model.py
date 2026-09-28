"""Pairwise GBDT matcher (XGBoost, GPU) with K-fold out-of-fold predictions on train.

Folds are grouped by S1 entity (all pairs of one S1 share a fold). Feature files (feats.parquet,
xfeats.parquet) are row-aligned and streamed in batches, so memory is bounded by the sampled training rows
rather than the full pair table. Boosters store their feature names; prediction selects columns by name."""
import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

NON_FEATURES = {"ri", "si", "y", "fold", "p"}


class StaleModels(RuntimeError):
    pass

PARAMS = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", max_depth=10,
              learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
              reg_lambda=1.0, max_bin=256)


def fold_of(si: np.ndarray, n_folds: int) -> np.ndarray:
    return ((si.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(2**32) % np.uint64(n_folds)).astype(np.int8)


def feature_cols(df: pd.DataFrame):
    return [c for c in df.columns if c not in NON_FEATURES]


KEYS = ("ri", "si")


def aligned_batches(paths, batch_rows: int = 5_000_000, columns=None):
    """Yield DataFrames of exactly batch_rows rows (last one shorter) with the columns of all row-aligned
    parquet files side by side. Every file must carry the pair keys (ri, si); they are compared batch by
    batch, so files whose rows are permuted or come from a different candidate set fail loudly.
    columns: optional set of names to read (the keys are always read)."""
    files = [pq.ParquetFile(p) for p in paths]
    n = files[0].metadata.num_rows
    if any(f.metadata.num_rows != n for f in files):
        raise ValueError(f"feature files are not row-aligned: {[f.metadata.num_rows for f in files]}")
    for p, f in zip(paths, files):
        if not set(KEYS) <= set(f.schema_arrow.names):
            raise ValueError(f"{p}: missing pair keys {KEYS}; rebuild it with the current code")
    seen = set(KEYS)
    cols = []
    for f in files:
        c = [x for x in f.schema_arrow.names if x not in seen and (columns is None or x in columns)]
        seen.update(c)
        cols.append(list(KEYS) + c)
    its = [f.iter_batches(batch_size=batch_rows, columns=c) for f, c in zip(files, cols)]
    bufs = [[] for _ in files]
    have = [0] * len(files)
    for start in range(0, n, batch_rows):
        need = min(batch_rows, n - start)
        parts = []
        for j, it in enumerate(its):
            while have[j] < need:
                b = next(it)
                bufs[j].append(b)
                have[j] += b.num_rows
            t = pa.Table.from_batches(bufs[j]).combine_chunks()
            parts.append(t.slice(0, need).to_pandas())
            rest = t.slice(need)
            bufs[j] = rest.to_batches() if rest.num_rows else []
            have[j] -= need
        for j in range(1, len(parts)):
            for k in KEYS:
                if not np.array_equal(parts[0][k].values, parts[j][k].values):
                    raise ValueError(f"{paths[j]}: pair keys differ from {paths[0]} in rows "
                                     f"{start}..{start + need} — feature files are not row-aligned")
            parts[j] = parts[j].drop(columns=list(KEYS))
        yield pd.concat(parts, axis=1)


def select_rows(paths, keep: np.ndarray, columns=None, batch_rows: int = 5_000_000) -> pd.DataFrame:
    out, a = [], 0
    for df in aligned_batches(paths, batch_rows, columns):
        m = keep[a:a + len(df)]
        if m.any():
            out.append(df[m])
        a += len(df)
    return pd.concat(out, ignore_index=True)


def feature_names(paths):
    """Feature columns of the row-aligned files, in file order (keys, labels and duplicates excluded)."""
    seen, cols = set(), []
    for p in paths:
        for c in pq.ParquetFile(p).schema_arrow.names:
            if c not in NON_FEATURES and c not in seen:
                seen.add(c)
                cols.append(c)
    return cols


def select_matrices(paths, masks, batch_rows: int = 5_000_000):
    """One pass over the feature files; for each boolean row mask, a preallocated float32 matrix of the selected
    rows. Memory ~ the selected rows only (no DataFrame copies or concatenation)."""
    cols = feature_names(paths)
    mats = [np.empty((int(m.sum()), len(cols)), dtype=np.float32) for m in masks]
    pos = [0] * len(masks)
    a = 0
    for df in aligned_batches(paths, batch_rows, set(cols)):
        for j, m in enumerate(masks):
            mm = m[a:a + len(df)]
            k = int(mm.sum())
            if k:
                mats[j][pos[j]:pos[j] + k] = df.loc[mm, cols].to_numpy(dtype=np.float32)
                pos[j] += k
        a += len(df)
    assert all(p == len(x) for p, x in zip(pos, mats))
    return mats, cols


def read_ids(path):
    t = pq.read_table(path, columns=["ri", "si", "y"])
    return t.column("ri").to_numpy(), t.column("si").to_numpy(), t.column("y").to_numpy()


def es_holdout(si: np.ndarray, frac: float) -> np.ndarray:
    """Hash-selected fraction of S1 entities held out of training for early stopping (independent of fold_of)."""
    return ((si.astype(np.uint64) * np.uint64(40503)) % np.uint64(1000)).astype(np.int32) < frac * 1000


def train_one(paths, keep, si, y, neg_rate, rounds, device, seed, params, es_frac=0.0, es_rounds=50):
    """Train on the kept rows. With es_frac > 0, rows of a held-out fraction of the training S1s form the
    early-stopping set (never the evaluated fold); the booster is truncated at the best iteration."""
    va = keep & es_holdout(si, es_frac) if es_frac > 0 else np.zeros_like(keep)
    tr = keep & ~va
    (Xtr, Xva), cols = select_matrices(paths, [tr, va])
    wt = lambda yy: np.where(yy == 1, 1.0, 1.0 / neg_rate).astype(np.float32)
    dm = xgb.QuantileDMatrix(Xtr, label=y[tr], weight=wt(y[tr]), max_bin=params["max_bin"], feature_names=cols)
    del Xtr
    evals, kw = [], {}
    if len(Xva):
        dv = xgb.QuantileDMatrix(Xva, label=y[va], weight=wt(y[va]), ref=dm, feature_names=cols)
        evals, kw = [(dv, "es")], {"early_stopping_rounds": es_rounds, "verbose_eval": False}
    del Xva
    bst = xgb.train({**params, "device": device, "seed": seed}, dm, num_boost_round=rounds, evals=evals, **kw)
    if evals:
        bst = bst[: bst.best_iteration + 1]
    return bst


def oof(paths, out: Path, model_dir: Path, n_folds: int, neg_rate: float, rounds: int, devices, params=None,
        es_frac: float = 0.0, upstream: str | None = None):
    """upstream: fingerprint of the training features; fold models left from a run with a different
    configuration or different features are refused instead of being reused."""
    params = {**PARAMS, **(params or {})}
    t = time.time()
    ri, si, y = read_ids(paths[0])
    fold = fold_of(si, n_folds)
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"n_folds": n_folds, "neg_rate": neg_rate, "rounds": rounds, "es_frac": es_frac, "params": params,
           "features": upstream}
    cfg_path = model_dir / "model_config.json"
    if cfg_path.exists() and json.load(open(cfg_path)) != json.loads(json.dumps(cfg)):
        raise StaleModels(f"{model_dir} holds models of a different configuration; use a new --work")
    if not cfg_path.exists() and any(model_dir.glob("xgb_fold*")):
        raise StaleModels(f"{model_dir} holds fold models without a configuration; use a new --work")
    json.dump(cfg, open(cfg_path, "w"), indent=1)
    errors = []

    def run(k):
        try:
            rng = np.random.default_rng(k)
            keep = (fold != k) & ((y == 1) | (rng.random(len(y)) < neg_rate))
            bst = train_one(paths, keep, si, y, neg_rate, rounds, devices[k % len(devices)], k, params, es_frac)
            bst.save_model(str(model_dir / f"xgb_fold{k}.ubj"))   # binary JSON: ~40% smaller, lossless
            print(f"fold {k} trained ({bst.num_boosted_rounds()} rounds) {time.time() - t:.0f}s", flush=True)
        except BaseException as e:
            errors.append(e)

    todo = [k for k in range(n_folds) if not any((model_dir / f"xgb_fold{k}{e}").exists() for e in (".ubj", ".json"))]
    for a in range(0, len(todo), len(devices)):
        th = [threading.Thread(target=run, args=(k,)) for k in todo[a:a + len(devices)]]
        [x.start() for x in th]
        [x.join() for x in th]
        if errors:
            raise errors[0]
    # predict with boosters reloaded from disk on one device (thread-trained boosters crashed on GPU predict)
    boosters = load_boosters(model_dir, devices[0])
    p = np.zeros(len(y), dtype=np.float32)
    a = 0
    for df in aligned_batches(paths):
        f = fold[a:a + len(df)]
        for k, bst in enumerate(boosters):
            m = f == k
            if m.any():
                p[a:a + len(df)][m] = bst.inplace_predict(df.loc[m, bst.feature_names])
        a += len(df)
    print(f"OOF predicted {time.time() - t:.0f}s", flush=True)
    imp = pd.Series(boosters[0].get_score(importance_type="gain"))
    print((imp / imp.sum()).sort_values(ascending=False).head(30).round(4).to_string())
    pd.DataFrame({"ri": ri, "si": si, "y": y, "p": p}).to_parquet(out, index=False)


def load_boosters(model_dir: Path, device):
    files = sorted([f for f in Path(model_dir).glob("xgb_fold*") if f.suffix in (".ubj", ".json")],
                   key=lambda f: int(f.stem[len("xgb_fold"):]))
    ks = [int(f.stem[len("xgb_fold"):]) for f in files]
    cfg_path = Path(model_dir) / "model_config.json"
    expected = json.load(open(cfg_path))["n_folds"] if cfg_path.exists() else len(files)
    if ks != list(range(expected)):
        raise StaleModels(f"{model_dir}: need exactly one xgb_fold<k>.ubj|.json for k = 0..{expected - 1}, "
                          f"found {[f.name for f in files]}")
    out = []
    for f in files:
        bst = xgb.Booster(model_file=str(f))
        bst.set_param({"device": device})  # one GPU per process: multi-GPU predict from CPU data crashes
        out.append(bst)
    return out


def predict(paths, out: Path, model_dir: Path, devices, batch_rows: int = 5_000_000):
    """Average of the fold models; feature files are streamed in batches to bound memory."""
    boosters = load_boosters(model_dir, devices[0])
    need = set(boosters[0].feature_names) | {"ri", "si"}
    tmp = Path(out).with_suffix(".tmp.parquet")
    writer = None
    for df in aligned_batches(paths, batch_rows, need):         # each batch is written out: memory ~ one batch
        p = np.mean([b.inplace_predict(df[b.feature_names]) for b in boosters], axis=0).astype(np.float32)
        t = pa.Table.from_pandas(pd.DataFrame({"ri": df.ri.values, "si": df.si.values, "p": p}), preserve_index=False)
        writer = writer or pq.ParquetWriter(tmp, t.schema)
        writer.write_table(t)
    writer.close()
    tmp.rename(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["oof", "predict"])
    ap.add_argument("--feats", required=True, help="comma-separated row-aligned feature parquet files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--neg-rate", type=float, default=0.25)
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--gpus", default="cuda:0,cuda:1")
    a = ap.parse_args()
    devs = [d for d in a.gpus.split(",") if d] or ["cpu"]
    paths = [Path(p) for p in a.feats.split(",")]
    if a.mode == "oof":
        oof(paths, Path(a.out), Path(a.model_dir), a.folds, a.neg_rate, a.rounds, devs)
    else:
        predict(paths, Path(a.out), Path(a.model_dir), devs)


if __name__ == "__main__":
    main()
