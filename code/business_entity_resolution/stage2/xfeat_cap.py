"""Token features with corpus-size-invariant IDF: idf = min(log(n/df), log(1e5)); unseen token -> log(1e5).
Wraps the production src.xfeatures builder (only country_stats changes)."""
import math, sys
from pathlib import Path
sys.path.insert(0, "/data/nishant/Nishant/Prashant/AmazonML26/code/business_entity_resolution")
from src import xfeatures as xf
CAP = math.log(1e5)
_orig = xf.country_stats
def capped(names, addrs):
    s = _orig(names, addrs)
    for k in ("idf_n", "idf_a", "idf_d"):
        s[k] = {t: min(v, CAP) for t, v in s[k].items()}
    s["dflt"] = CAP
    return s
xf.country_stats = capped
split = sys.argv[1]
W4 = Path("/data/nishant/Nishant/Prashant/AmazonML26/work_v4")
xf.build(W4 / "cache", split, W4 / split / "cands.parquet", Path(f"/data/nishant/Nishant/Prashant/AmazonML26/work_v5/{split}/xfeats_cap.parquet"), 128)
