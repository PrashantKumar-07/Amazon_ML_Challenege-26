"""Fingerprints for cached pipeline outputs.

Every cached artifact gets a sidecar `<file>.meta.json` holding a fingerprint of what produced it: the input
data, the code of the modules involved, the configuration and the fingerprints of upstream artifacts. A cached
file is reused only when its recorded fingerprint equals the one the current run would produce; otherwise the
run stops instead of silently mixing results from different code, data or settings."""
import hashlib
import json
from pathlib import Path

SRC = Path(__file__).resolve().parent


def _h(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def file_digest(path, chunk: int = 4 << 20) -> str:
    """Size + hash of the first and last chunk (fast for multi-GB files, catches re-extracted/edited data)."""
    path = Path(path)
    size = path.stat().st_size
    with open(path, "rb") as f:
        head = f.read(chunk)
        f.seek(max(0, size - chunk))
        tail = f.read(chunk)
    return f"{size}:{_h(head + tail)}"


def full_digest(path) -> str:
    return _h(Path(path).read_bytes())


def code_digest(*modules: str) -> str:
    return _h(b"".join((SRC / f"{m}.py").read_bytes() for m in sorted(modules)))


def fingerprint(**parts) -> str:
    return _h(json.dumps(parts, sort_keys=True, default=str).encode())


def meta_path(path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".meta.json")


def read(path) -> str | None:
    m = meta_path(path)
    return json.loads(m.read_text())["fingerprint"] if m.exists() else None


def write(path, fp: str, **info):
    meta_path(path).write_text(json.dumps({"fingerprint": fp, **info}, indent=1, default=str))


class StaleArtifact(RuntimeError):
    pass


def check_reusable(path, fp: str) -> bool:
    """True if `path` exists with a matching fingerprint; False if it does not exist; raises if stale."""
    path = Path(path)
    if not path.exists():
        return False
    got = read(path)
    if got != fp:
        raise StaleArtifact(
            f"{path} exists but was produced by different data/code/configuration "
            f"(recorded {got}, expected {fp}). Delete it (and everything downstream) or use a new --work.")
    return True
