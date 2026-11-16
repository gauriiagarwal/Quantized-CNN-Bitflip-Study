"""Bit-flip fault injection on the stored weights of a LeNet-style CNN (simulation only).
Needs: torch, numpy.  The landing/other pages never import this file."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LAYERS = ["conv1", "conv2", "fc1", "fc2", "fc3"]
NBITS = {"FP32": 32, "INT8": 8, "INT4": 4}
QMAX = {"INT8": 127, "INT4": 7}


class LeNet(nn.Module):
    """LeNet-5 with padding=2 on conv1: exactly 61,706 parameters."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


def make_cnn():
    return LeNet()


def load_model(path):
    """Load a saved state_dict. Keys are matched by name first, then by order and shape."""
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, nn.Module):
        obj = obj.state_dict()
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    m = make_cnn()
    try:
        m.load_state_dict(obj)
    except Exception:
        tensors = [v for v in obj.values() if torch.is_tensor(v)]
        mine = list(m.state_dict().items())
        if len(tensors) != len(mine) or any(a.shape != b.shape for a, (_, b) in zip(tensors, mine)):
            raise ValueError("Saved model does not match the expected LeNet layout "
                             "(conv1 1x6x5x5, conv2 6x16x5x5, fc1 400x120, fc2 120x84, fc3 84x10). "
                             "Run print(model) in Colab and send it.")
        m.load_state_dict({k: t for (k, _), t in zip(mine, tensors)})
    return m.eval()


def weights_of(model):
    return {n: getattr(model, n).weight.detach().cpu().numpy().astype(np.float32) for n in LAYERS}


def quantize(w, fmt):
    """Symmetric per-tensor quantization. Returns (integer array, scale)."""
    qmax = QMAX[fmt]
    scale = float(np.abs(w).max()) / qmax if np.abs(w).max() > 0 else 1.0
    q = np.clip(np.round(w / scale), -qmax, qmax).astype(np.int32)
    return q, scale


def _to_bits(w, fmt):
    """Weights -> unsigned integer pattern as stored in memory (plus a decoder)."""
    if fmt == "FP32":
        u = np.ascontiguousarray(w, dtype=np.float32).view(np.uint32).astype(np.uint64)
        return u, (lambda b: b.astype(np.uint32).view(np.float32))
    q, scale = quantize(w, fmt)
    nb = NBITS[fmt]
    u = (q & ((1 << nb) - 1)).astype(np.uint64)

    def dec(b):
        b = b.astype(np.int64)
        b = np.where(b >= (1 << (nb - 1)), b - (1 << nb), b)
        return (b * scale).astype(np.float32)
    return u, dec


def _popcount(x, nb):
    return sum(((x >> i) & 1) for i in range(nb)).astype(np.int64)


def _with_weights(base, weights):
    m = make_cnn()
    m.load_state_dict(base.state_dict())
    with torch.no_grad():
        for n in LAYERS:
            getattr(m, n).weight.copy_(torch.from_numpy(weights[n].astype(np.float32)))
    return m.eval()


def corrupt(base, fmt, mode, seed=0, layer="fc1", bit=0, frac=0.02, ber=1e-3,
            want=("base", "none", "parity", "nulling")):
    """Return dict(models={base, none, parity, nulling}, info={...}).
    mode 'targeted': flip bit `bit` in a random `frac` of the weights of `layer`.
    mode 'ber': every stored bit (weight bits + 1 parity bit) flips with probability `ber` in all layers."""
    rng = np.random.default_rng(seed)
    nb = NBITS[fmt]
    clean = weights_of(base)
    maxabs = {n: float(np.abs(clean[n]).max()) for n in LAYERS}
    w_base, w_none, w_par, w_nul = {}, {}, {}, {}
    n_flip = n_detect = n_weights_hit = 0
    example = None
    for name in LAYERS:
        w = clean[name]
        flat = w.reshape(-1)
        u, dec = _to_bits(flat, fmt)
        w_base[name] = dec(u).reshape(w.shape)
        mask = np.zeros(len(u), dtype=np.uint64)
        pflip = np.zeros(len(u), dtype=np.int64)
        if mode == "targeted":
            if name == layer:
                k = max(1, int(round(frac * len(u))))
                idx = rng.choice(len(u), size=k, replace=False)
                mask[idx] = np.uint64(1) << np.uint64(bit)
        else:
            flips = rng.random((len(u), nb + 1)) < ber
            for i in range(nb):
                mask |= flips[:, i].astype(np.uint64) << np.uint64(i)
            pflip = flips[:, nb].astype(np.int64)
        hit = mask != 0
        n_flip += int(_popcount(mask, nb).sum() + pflip.sum())
        n_weights_hit += int(hit.sum())
        bad = dec(u ^ mask)
        odd = ((_popcount(mask, nb) + pflip) % 2) == 1
        n_detect += int(odd.sum())
        par = np.where(odd, np.float32(0), bad)
        with np.errstate(invalid="ignore"):
            nul = np.where(np.isnan(bad) | (np.abs(bad) > maxabs[name]), np.float32(0), bad)
        w_none[name] = bad.reshape(w.shape)
        w_par[name] = par.reshape(w.shape)
        w_nul[name] = nul.reshape(w.shape)
        if example is None and hit.any() and (mode == "ber" or name == layer):
            j = int(np.flatnonzero(hit)[0])
            example = dict(layer=name, before=float(dec(u)[j]), after=float(bad[j]),
                           bits_before=int(u[j]), bits_after=int(u[j] ^ mask[j]), nbits=nb)
    src = {"base": w_base, "none": w_none, "parity": w_par, "nulling": w_nul}
    models = {k: _with_weights(base, src[k]) for k in want}
    info = dict(flipped_bits=n_flip, weights_hit=n_weights_hit, detected=n_detect,
                parity_overhead_pct=100.0 / nb, example=example, fmt=fmt)
    return dict(models=models, info=info)


@torch.no_grad()
def predict(model, X):
    """X: float tensor (N,1,28,28). Returns (labels, confidence)."""
    out = model(X)
    out = torch.nan_to_num(out, nan=0.0, posinf=1e30, neginf=-1e30)
    p = torch.softmax(out.double(), dim=1)
    conf, lab = p.max(1)
    return lab.numpy(), conf.numpy()


@torch.no_grad()
def probs(model, X):
    """Softmax probabilities, shape (N, 10). NaN/inf in the logits are cleaned first."""
    out = torch.nan_to_num(model(X), nan=0.0, posinf=1e30, neginf=-1e30)
    return torch.softmax(out.double(), dim=1).numpy()


def accuracy(model, X, y):
    lab, _ = predict(model, X)
    return float((lab == np.asarray(y)).mean() * 100)
