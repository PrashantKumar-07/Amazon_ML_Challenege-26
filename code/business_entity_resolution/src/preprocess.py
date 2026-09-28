"""Read raw TSVs, normalise, cache to parquet: <cache>/<split>_s{1,2,3}.parquet."""
import argparse
import json
import os
import shutil
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from . import fingerprint as fpr
from . import token_map
from .io_utils import gt_pairs, read_ground_truth, read_source
from .normalize import digits, has_non_latin, norm_addr, norm_name
from .token_map import apply_token_map, learn_token_map


def _norm_chunk(args):
    names, addrs = args
    nn = [norm_name(x) for x in names]
    na = [norm_addr(x) for x in addrs]
    return nn, na, [digits(x) for x in na], [has_non_latin(x) for x in names]


def normalise_df(df: pd.DataFrame, workers: int) -> pd.DataFrame:
    n = len(df)
    bounds = np.linspace(0, n, max(1, workers * 4) + 1, dtype=int)
    chunks = [(df.business_name.values[a:b].tolist(), df.business_address.values[a:b].tolist())
              for a, b in zip(bounds[:-1], bounds[1:])]
    with Pool(workers) as pool:
        res = pool.map(_norm_chunk, chunks)
    out = pd.DataFrame({
        "id": df.entity_id.values,
        "name_raw": df.business_name.values,
        "addr_raw": df.business_address.values,
        "country": df.country.values,
    })
    out["name"] = [x for r in res for x in r[0]]
    out["addr"] = [x for r in res for x in r[1]]
    out["addr_digits"] = [x for r in res for x in r[2]]
    out["name_nonlatin"] = [x for r in res for x in r[3]]
    return out


def build_token_map(data_dir: Path, cache: Path) -> dict:
    s1 = pd.read_parquet(cache / "train_s1.parquet", columns=["id", "name_raw"]).set_index("id").name_raw
    recs = pd.concat([pd.read_parquet(cache / f"train_s{s}.parquet", columns=["id", "name_raw", "name_nonlatin"])
                      for s in (2, 3)])
    recs = recs[recs.name_nonlatin].set_index("id").name_raw
    pairs = gt_pairs(read_ground_truth(data_dir / "train" / "train_ground_truth.tsv"))
    pairs = pairs[pairs.rid.isin(recs.index)]
    return learn_token_map(s1.reindex(pairs.s1).values, recs.reindex(pairs.rid).values, entity_ids=pairs.s1.values)


def apply_map_to_cache(path: Path, tmap: dict):
    df = pd.read_parquet(path)
    if "name_mapped" in df.columns:
        raise fpr.StaleArtifact(f"{path} is already mapped but its metadata says otherwise; use a new --work")
    m = df.name_nonlatin.values
    df["name_mapped"] = False
    df.loc[m, "name"] = [norm_name(apply_token_map(x, tmap)) for x in df.name_raw.values[m]]
    df.loc[m, "name_mapped"] = True
    df.to_parquet(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="folder containing train/ and test/")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count()))
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--token-map", default=None,
                    help="pre-learned token_map.json (required when train is not among --splits)")
    args = ap.parse_args()
    splits = [x for x in args.splits.split(",") if x]
    data, cache = Path(args.data_dir), Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    code = fpr.code_digest("normalize", "preprocess", "token_map", "io_utils")
    src = lambda split, s: data / split / f"{split}_source{s}.tsv"
    fp_norm = lambda split, s: fpr.fingerprint(stage="normalise", data=fpr.file_digest(src(split, s)), code=code)

    # 1) normalised caches (fingerprint of data + code; the token map is applied in step 3)
    for split in splits:
        for s in (1, 2, 3):
            dst = cache / f"{split}_s{s}.parquet"
            if dst.exists():
                got = fpr.read(dst)
                if got is not None and json.loads(fpr.meta_path(dst).read_text()).get("norm") == fp_norm(split, s):
                    continue
                raise fpr.StaleArtifact(f"{dst} was produced by different data or code; delete the cache or use a "
                                        f"new --work")
            t = time.time()
            df = read_source(src(split, s))
            normalise_df(df, args.workers).to_parquet(dst, index=False)
            fpr.write(dst, fp_norm(split, s), norm=fp_norm(split, s), mapped=None)
            print(f"{dst.name}: {len(df):,} rows in {time.time() - t:.0f}s", flush=True)

    # 2) token map: learned from train pairs, or given; a given map must not silently lose to a cached one
    map_path = cache / "token_map.json"
    if args.token_map:
        given = Path(args.token_map)
        if map_path.exists() and fpr.full_digest(map_path) != fpr.full_digest(given):
            raise fpr.StaleArtifact(f"{map_path} differs from --token-map {given}; use a new --work")
        if not map_path.exists():
            shutil.copyfile(given, map_path)                    # byte copy: digest comparable to --token-map
            fpr.write(map_path, "given:" + fpr.full_digest(given))
    learned_fp = None
    if "train" in splits and not args.token_map:
        learned_fp = fpr.fingerprint(stage="token_map", code=code, gt=fpr.file_digest(data / "train" / "train_ground_truth.tsv"),
                                     train=[fp_norm("train", s) for s in (1, 2, 3)])
    if map_path.exists():
        if learned_fp is not None:
            fpr.check_reusable(map_path, learned_fp)
        tmap = token_map.load(map_path)
    else:
        if "train" not in splits:
            raise ValueError("no token map: pass --token-map or include the train split")
        tmap = build_token_map(data, cache)
        token_map.save(tmap, map_path)
        fpr.write(map_path, learned_fp)
    tm_digest = fpr.full_digest(map_path)
    print(f"token map: {len(tmap):,} entries", flush=True)

    # 3) apply the map; the final fingerprint of each cache file covers data, code and map
    for split in splits:
        for s in (1, 2, 3):
            dst = cache / f"{split}_s{s}.parquet"
            meta = json.loads(fpr.meta_path(dst).read_text())
            final = fpr.fingerprint(norm=meta["norm"], token_map=tm_digest)
            if meta.get("mapped") == tm_digest:
                continue
            if meta.get("mapped") is not None:
                raise fpr.StaleArtifact(f"{dst} was mapped with a different token map; use a new --work")
            apply_map_to_cache(dst, tmap)
            fpr.write(dst, final, norm=meta["norm"], mapped=tm_digest)
    print("token map applied", flush=True)


if __name__ == "__main__":
    main()
