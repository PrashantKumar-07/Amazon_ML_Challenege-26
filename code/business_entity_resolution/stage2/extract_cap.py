"""Reduced-table matrices with capped-IDF token features: copy {split}_X.npy and replace every column produced by
xfeatures (x_*, rg_x_*, rr_x_*) with the values from <BER_W5>/{split}/xfeats_cap.parquet (row-aligned, keys checked)."""
import json, os, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
ROOT = Path(os.environ.get("BER_ROOT", Path(__file__).resolve().parent.parent.parent.parent))
W5 = Path(os.environ.get("BER_W5", ROOT / "work" / "stage2"))
D = str(W5 / "data")
split = sys.argv[1]
cols = json.load(open(f"{D}/feat_cols.json"))
m = np.load(f"{D}/{split}_meta.npz", allow_pickle=True)
rows, ri, si = m["rows"], m["ri"], m["si"]
X = np.array(np.load(f"{D}/{split}_X.npy", mmap_mode="r"))
pf = pq.ParquetFile(W5 / split / "xfeats_cap.parquet")
xcols = [c for c in pf.schema_arrow.names if c in cols]
print(len(xcols), "columns replaced")
pos, start = 0, 0
for b in pf.iter_batches(batch_size=5_000_000, columns=["ri", "si"] + xcols):
    n = b.num_rows
    k = np.searchsorted(rows, start + n) - pos           # rows falling in [start, start+n)
    if k:
        loc = rows[pos:pos + k] - start
        d = b.to_pandas()
        assert np.array_equal(d.ri.values[loc], ri[pos:pos + k]) and np.array_equal(d.si.values[loc], si[pos:pos + k])
        for c in xcols:
            X[pos:pos + k, cols.index(c)] = d[c].values[loc]
        pos += k
    start += n
assert pos == len(rows), (pos, len(rows))
np.save(f"{D}/{split}_Xcap.npy", X)
print(split, "done", X.shape)
