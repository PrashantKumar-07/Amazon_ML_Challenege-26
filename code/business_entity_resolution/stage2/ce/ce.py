"""Cross-encoder pair scorer (multilingual-e5-small, MIT) on normalised 'name | address' text.
train  : python ce.py train   --split train --sel "fold2==0" --out DIR
predict: python ce.py predict --split train|test --sel "fold2==1" --model DIR --out scores.npy
--sel is evaluated on the reduced-table meta arrays (fold2, fold5, cty, y, p4, ploco)."""
import argparse, math, os, time, json
import numpy as np, polars as pl, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

os.environ.setdefault("HF_HOME", "/data/nishant/Nishant/Prashant/AmazonML26/models_hf")
W, D = "/data/nishant/Nishant/Prashant/AmazonML26/work_v4", "/data/nishant/Nishant/Prashant/AmazonML26/work_v5/data"
BASE = os.environ.get("CE_BASE", "intfloat/multilingual-e5-small")
MAXLEN = 128
T0 = time.time()


def log(*a):
    print(*a, f"[{time.time() - T0:.0f}s]", flush=True)


def texts(split, ri, si):
    def side(tabs):
        df = pl.concat([pl.read_parquet(f"{W}/cache/{split}_{t}.parquet", columns=["name", "addr"]) for t in tabs])
        return (df["name"].fill_null("") + " | " + df["addr"].fill_null("")).to_numpy()
    s1 = side(["s1"]); rec = side(["s2", "s3"])
    return s1[si], rec[ri]


def select(split, expr):
    m = np.load(f"{D}/{split}_meta.npz", allow_pickle=True)
    env = {k: m[k] for k in m.files}
    sel = np.ones(len(env["ri"]), bool) if expr in ("", "all") else eval(expr, {"np": np}, env)
    return np.flatnonzero(sel), env


class Batches(torch.utils.data.Dataset):
    def __init__(self, a, b, y, idx, bs, tok):
        self.a, self.b, self.y, self.bs, self.tok = a, b, y, bs, tok
        self.chunks = [idx[i:i + bs] for i in range(0, len(idx), bs)]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, k):
        c = self.chunks[k]
        e = self.tok(list(self.a[c]), list(self.b[c]), truncation="longest_first", max_length=MAXLEN,
                     padding=True, return_tensors="pt")
        e["labels"] = torch.from_numpy(self.y[c].astype(np.float32)) if self.y is not None else torch.from_numpy(c)
        return dict(e)


def train(a):
    idx, env = select(a.split, a.sel)
    ri, si, y = env["ri"][idx], env["si"][idx], env["y"][idx]
    A, B = texts(a.split, ri, si)
    log(f"train rows {len(idx):,} pos {y.mean():.3f}")
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1).cuda()
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(idx))
    hold = order[:20000]; order = order[20000:]
    ds = Batches(A, B, y, order, a.bs, tok)
    dl = torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False, num_workers=6, prefetch_factor=8,
                                     persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = len(ds) * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.03 * steps), steps)
    hv = Batches(A, B, y, hold, 1024, tok)
    step = 0
    for ep in range(a.epochs):
        model.train(); run = 0.0
        for bt in dl:
            lab = bt.pop("labels").cuda(non_blocking=True)
            bt = {k: v.cuda(non_blocking=True) for k, v in bt.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model(**bt).logits.squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z.float(), lab)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            run = 0.98 * run + 0.02 * loss.item() if step else loss.item(); step += 1
            if step % 1000 == 0 or step == steps:
                model.eval(); zs = []
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for k in range(len(hv)):
                        e = hv[k]; e.pop("labels")
                        zs.append(model(**{kk: v.cuda() for kk, v in e.items()}).logits.squeeze(-1).float().cpu())
                zs = torch.cat(zs).numpy(); yh = y[hold]
                ll = float(np.mean(np.logaddexp(0, zs) - yh * zs)); acc = float(((zs > 0) == (yh == 1)).mean())
                log(f"ep {ep} step {step}/{steps} train_loss {run:.4f} hold_logloss {ll:.4f} hold_acc {acc:.4f}")
                model.train()
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    json.dump(vars(a), open(f"{a.out}/ce_cfg.json", "w"))
    log("saved", a.out)


@torch.no_grad()
def predict(a):
    idx, env = select(a.split, a.sel)
    ri, si = env["ri"][idx], env["si"][idx]
    A, B = texts(a.split, ri, si)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model).cuda().eval()
    L = np.char.str_len(A.astype(str)) + np.char.str_len(B.astype(str))
    order = np.argsort(L, kind="stable")
    ds = Batches(A, B, None, order, a.bs, tok)
    dl = torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False, num_workers=6, prefetch_factor=8)
    out = np.empty(len(idx), np.float32); n = 0
    for bt in dl:
        c = bt.pop("labels").numpy()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out[c] = model(**{k: v.cuda(non_blocking=True) for k, v in bt.items()}).logits.squeeze(-1).float().cpu().numpy()
        n += len(c)
        if n % (a.bs * 2000) < a.bs:
            log(f"predicted {n:,}/{len(idx):,}")
    full = np.full(len(env["ri"]), np.nan, np.float32); full[idx] = out
    np.save(a.out, full)
    log("saved", a.out, "rows", len(idx))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "predict"]); ap.add_argument("--split", default="train")
    ap.add_argument("--sel", default="all"); ap.add_argument("--out", required=True); ap.add_argument("--model", default="")
    ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.mode == "predict" and a.bs == 256:
        a.bs = 1024
    train(a) if a.mode == "train" else predict(a)
