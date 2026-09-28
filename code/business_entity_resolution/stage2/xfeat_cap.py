"""Token features with corpus-size-invariant IDF: idf = min(log(n/df), log(1e5)); unseen token -> log(1e5).
Wraps the production src.xfeatures builder (only country_stats changes)."""
import math, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import xfeatures as xf
ROOT = Path(os.environ.get("BER_ROOT", Path(__file__).resolve().parent.parent.parent.parent))
W4 = Path(os.environ.get("BER_W4", ROOT / "work" / "stage1"))
W5 = Path(os.environ.get("BER_W5", ROOT / "work" / "stage2"))
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
xf.build(W4 / "cache", split, W4 / split / "cands.parquet", W5 / split / "xfeats_cap.parquet", 128)
