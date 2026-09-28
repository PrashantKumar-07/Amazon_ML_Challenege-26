"""Cross-encoder features: logit + record-side context (gap to the record's best other candidate, rank).
Context is computed only over the rows each view scores (train 'in': v4 p>=T; train 'loco': LOCO p>=T; test: all)."""
import sys, numpy as np, pandas as pd
D = "/data/nishant/Nishant/Prashant/AmazonML26/work_v5/data"
T = 0.003


def context(ri, z, rows):
    g = np.full(len(z), np.nan, np.float32); r = np.full(len(z), np.nan, np.float32)
    rs, zs = ri[rows], z[rows]
    o = np.lexsort((-zs, rs)); rs, zs, idx = rs[o], zs[o], rows[o]
    start = np.r_[True, rs[1:] != rs[:-1]]
    grp = np.cumsum(start) - 1; first = np.flatnonzero(start)
    rank = np.arange(len(rs)) - first[grp]
    nxt = first + 1
    has2 = nxt < len(rs)
    has2[has2] = rs[nxt[has2]] == rs[first[has2]]
    top2 = np.where(has2, zs[np.minimum(nxt, len(rs) - 1)], -20.0)
    other = np.where(rank == 0, top2[grp], zs[first][grp])
    g[idx] = zs - other; r[idx] = rank
    return g, r


def save(name, split, x):
    np.save(f"{D}/extra_{name}_{split}.npy", x.astype(np.float32))


def train():
    m = np.load(f"{D}/train_meta.npz", allow_pickle=True)
    ri, f2, cty, p4, pl_ = m["ri"], m["fold2"], m["cty"], m["p4"], m["ploco"]
    a0, a1 = np.load(f"{D}/ce_f0_train.npy"), np.load(f"{D}/ce_f1_train.npy")
    oof = np.where(f2 == 0, a1, a0)                      # model f1 scored fold-0 rows and vice versa
    assert not np.isnan(oof).any()
    save("ce", "train", oof)
    g, r = context(ri, oof, np.flatnonzero(p4 >= T)); save("rg_ce", "train", g); save("rr_ce", "train", r)
    import os
    if not os.path.exists(f"{D}/ce_India_on_US_train.npy"):
        print("train CE feats done (no LOCO CE yet)"); return
    lu, li = np.load(f"{D}/ce_US_on_India_train.npy"), np.load(f"{D}/ce_India_on_US_train.npy")
    loco = np.where(cty == "India", lu, li)
    assert not np.isnan(loco).any()
    save("ce_loco", "train", loco)
    g, r = context(ri, loco, np.flatnonzero(pl_ >= T)); save("rg_ce_loco", "train", g); save("rr_ce_loco", "train", r)
    print("train CE feats done")


def test():
    m = np.load(f"{D}/test_meta.npz", allow_pickle=True)
    z = (np.load(f"{D}/ce_f0_test.npy") + np.load(f"{D}/ce_f1_test.npy")) / 2
    assert not np.isnan(z).any()
    save("ce", "test", z)
    g, r = context(m["ri"], z, np.arange(len(z))); save("rg_ce", "test", g); save("rr_ce", "test", r)
    print("test CE feats done")


if __name__ == "__main__":
    {"train": train, "test": test}[sys.argv[1]]()
