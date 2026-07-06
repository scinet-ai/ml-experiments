"""
grok.py — Reproduce grokking on three toy tasks and log mechanism-agnostic
progress measures every N steps.

Tasks:
  - add : (a + b) mod p            [1-layer transformer]
  - mul : (a * b) mod p            [1-layer transformer]
  - parity : parity of k fixed bits among n   [1-hidden-layer MLP]

Mechanism-agnostic measures (computed ONLY from weights/activations, no
task-specific circuit knowledge):
  - weight_l2          : total L2 norm of all trainable params
  - w_eff_rank         : mean participation-ratio effective rank of 2D weight
                         matrices  (PR = (sum s^2)^2 / sum s^4  over singular vals)
  - act_eff_rank       : effective rank of the hidden-activation covariance
                         on a fixed probe batch
  - act_sparsity       : fraction of near-zero post-nonlinearity activations
  - act_kurtosis       : excess kurtosis of post-nonlinearity activations
  - gzip_bytes         : MDL proxy = gzip size of int8-quantized weights

Everything runs standalone with no dataset download (data is synthetic).
"""
import argparse, csv, gzip, io, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = os.environ.get("GROK_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")


# ----------------------------- data -----------------------------------------
def make_modular(p, op, train_frac, seed):
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    if op == "add":
        c = (a + b) % p
    elif op == "mul":
        c = (a * b) % p
    else:
        raise ValueError(op)
    eq = torch.full_like(a, p)  # '=' token id == p
    x = torch.stack([a, b, eq], dim=1)  # (N, 3)  tokens in [0, p]
    y = c
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=g)
    n_tr = int(train_frac * x.shape[0])
    tr, te = perm[:n_tr], perm[n_tr:]
    return x[tr], y[tr], x[te], y[te], p + 1  # vocab size


def make_parity(n, k, n_train, n_test, seed):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:k]  # k fixed relevant bits
    total = n_train + n_test
    bits = (torch.rand(total, n, generator=g) < 0.5).float()  # {0,1}
    y = (bits[:, idx].sum(dim=1) % 2).long()
    x = bits * 2 - 1  # map to {-1,+1}
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


# ----------------------------- models ---------------------------------------
class OneLayerTransformer(nn.Module):
    """Minimal 1-layer, attention + MLP transformer for modular arithmetic."""
    def __init__(self, vocab, d_model=128, n_heads=4, n_ctx=3, d_out=None):
        super().__init__()
        d_out = d_out or (vocab - 1)  # predict class in [0, p)
        self.d_model, self.n_heads = d_model, n_heads
        self.dh = d_model // n_heads
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(torch.randn(n_ctx, d_model) * 0.02)
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.mlp_in = nn.Linear(d_model, 4 * d_model, bias=False)
        self.mlp_out = nn.Linear(4 * d_model, d_model, bias=False)
        self.unembed = nn.Linear(d_model, d_out, bias=False)
        self._hidden = None  # last MLP post-ReLU activations (probe cache)

    def forward(self, x):
        B, T = x.shape
        h = self.embed(x) + self.pos[:T]
        q = self.Wq(h).view(B, T, self.n_heads, self.dh).transpose(1, 2)
        k = self.Wk(h).view(B, T, self.n_heads, self.dh).transpose(1, 2)
        v = self.Wv(h).view(B, T, self.n_heads, self.dh).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        att = att.softmax(dim=-1)
        z = (att @ v).transpose(1, 2).reshape(B, T, self.d_model)
        h = h + self.Wo(z)
        hid = F.relu(self.mlp_in(h))
        self._hidden = hid[:, -1, :].detach()  # activations at '=' position
        h = h + self.mlp_out(hid)
        logits = self.unembed(h[:, -1, :])  # read out at last ('=') position
        return logits

    def hidden(self):
        return self._hidden


class MLP(nn.Module):
    """1-hidden-layer MLP for sparse parity.

    init_scale multiplies the initial weights (Omnigrok-style large init that
    puts the net in a memorizing regime, so weight decay induces DELAYED
    grokking instead of immediate generalization).
    """
    def __init__(self, n_in, width=256, n_out=2, init_scale=1.0):
        super().__init__()
        self.fc1 = nn.Linear(n_in, width, bias=True)
        self.fc2 = nn.Linear(width, n_out, bias=True)
        if init_scale != 1.0:
            with torch.no_grad():
                self.fc1.weight.mul_(init_scale)
                self.fc2.weight.mul_(init_scale)
        self._hidden = None

    def forward(self, x):
        hid = F.relu(self.fc1(x))
        self._hidden = hid.detach()
        return self.fc2(hid)

    def hidden(self):
        return self._hidden


# ----------------------------- measures -------------------------------------
def eff_rank_from_svals(s):
    """Participation ratio of squared singular values (effective rank)."""
    s2 = s ** 2
    denom = (s2 ** 2).sum()
    if denom <= 0:
        return 0.0
    return float((s2.sum() ** 2) / denom)


@torch.no_grad()
def weight_measures(model):
    total_sq = 0.0
    eff_ranks = []
    for _, prm in model.named_parameters():
        total_sq += float((prm.detach() ** 2).sum())
        w = prm.detach()
        if w.ndim == 2 and min(w.shape) > 1:
            s = torch.linalg.svdvals(w.float().cpu())
            eff_ranks.append(eff_rank_from_svals(s))
    return math.sqrt(total_sq), (float(np.mean(eff_ranks)) if eff_ranks else 0.0)


@torch.no_grad()
def gzip_bytes(model):
    """MDL proxy: gzip size of per-tensor int8-quantized weights."""
    buf = bytearray()
    for _, prm in model.named_parameters():
        w = prm.detach().float().cpu().numpy().ravel()
        scale = (np.abs(w).max() + 1e-12) / 127.0
        q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
        buf += q.tobytes()
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", compresslevel=9) as f:
        f.write(bytes(buf))
    return len(out.getvalue())


@torch.no_grad()
def activation_measures(model, probe_x):
    model(probe_x)
    h = model.hidden().float().cpu().numpy()  # (n_probe, d_hidden)
    # effective rank of activation covariance
    hc = h - h.mean(axis=0, keepdims=True)
    cov = hc.T @ hc / max(1, h.shape[0])
    ev = np.linalg.eigvalsh(cov)
    ev = np.clip(ev, 0, None)
    denom = (ev ** 2).sum()
    act_eff = float((ev.sum() ** 2) / denom) if denom > 0 else 0.0
    # sparsity: fraction of near-zero (post-ReLU) activations
    thr = 1e-3 * (np.abs(h).max() + 1e-12)
    sparsity = float((np.abs(h) < thr).mean())
    # excess kurtosis of activations
    hf = h.ravel()
    mu, sd = hf.mean(), hf.std() + 1e-12
    kurt = float((((hf - mu) / sd) ** 4).mean() - 3.0)
    return act_eff, sparsity, kurt


# ----------------------------- train ----------------------------------------
def accuracy(model, x, y, bs=4096):
    correct = 0
    for i in range(0, x.shape[0], bs):
        logits = model(x[i:i + bs])
        correct += int((logits.argmax(-1) == y[i:i + bs]).sum())
    return correct / x.shape[0]


def train(task="add", p=97, n=40, k=3, seed=0, steps=30000, log_every=100,
          d_model=128, width=256, lr=1e-3, wd=1.0, train_frac=0.4,
          n_train=1200, n_test=1200, init_scale=1.0, out_csv=None, quiet=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if task in ("add", "mul"):
        xtr, ytr, xte, yte, vocab = make_modular(p, task, train_frac, seed)
        model = OneLayerTransformer(vocab, d_model=d_model)
    elif task == "parity":
        xtr, ytr, xte, yte = make_parity(n, k, n_train, n_test, seed)
        model = MLP(n, width=width, init_scale=init_scale)
    else:
        raise ValueError(task)

    dev = torch.device(DEVICE)
    model = model.to(dev)
    xtr, ytr, xte, yte = xtr.to(dev), ytr.to(dev), xte.to(dev), yte.to(dev)
    # fixed probe batch for activation measures (subset of test inputs)
    probe = xte[:min(512, xte.shape[0])]

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98),
                            weight_decay=wd)
    rows = []
    t0 = time.time()
    for step in range(steps + 1):
        model.train()
        logits = model(xtr)
        loss = F.cross_entropy(logits, ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % log_every == 0:
            model.eval()
            with torch.no_grad():
                tr_acc = accuracy(model, xtr, ytr)
                te_acc = accuracy(model, xte, yte)
                te_loss = float(F.cross_entropy(model(xte), yte))
            wl2, weff = weight_measures(model)
            gz = gzip_bytes(model)
            aeff, asp, akurt = activation_measures(model, probe)
            rows.append(dict(step=step, train_acc=tr_acc, test_acc=te_acc,
                             train_loss=float(loss), test_loss=te_loss,
                             weight_l2=wl2, w_eff_rank=weff, act_eff_rank=aeff,
                             act_sparsity=asp, act_kurtosis=akurt,
                             gzip_bytes=gz))
            if not quiet:
                print(f"[{task} s{seed}] step {step:6d} tr {tr_acc:.3f} te {te_acc:.3f} "
                      f"wl2 {wl2:7.2f} weff {weff:6.2f} gz {gz:6d} "
                      f"aeff {aeff:6.2f} sp {asp:.3f}", flush=True)
    if out_csv:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    if not quiet:
        print(f"[{task} s{seed}] done in {time.time()-t0:.1f}s -> {out_csv}", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="add", choices=["add", "mul", "parity"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train_frac", type=float, default=0.4)
    ap.add_argument("--n_train", type=int, default=1200)
    ap.add_argument("--n_test", type=int, default=1200)
    ap.add_argument("--init_scale", type=float, default=1.0)
    ap.add_argument("--out_csv", default=None)
    a = ap.parse_args()
    train(task=a.task, p=a.p, n=a.n, k=a.k, seed=a.seed, steps=a.steps,
          log_every=a.log_every, wd=a.wd, lr=a.lr, train_frac=a.train_frac,
          n_train=a.n_train, n_test=a.n_test, init_scale=a.init_scale,
          out_csv=a.out_csv)
