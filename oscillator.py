"""
The pieces every script in this study shares: the problem, the data, the
network, the loss, the training schedule, the classical fits, and the warm
start. Each case script imports from here, so there is one copy of each.

    u'' + mu u' + k u = 0,   u(0) = 1,  u'(0) = 0,   t in [0, 1]
    d = 2, omega_0 = 20   ->   mu = 2d = 4,  k = omega_0^2 = 400

Set PINN_QUICK=1 in the environment for a smoke test: every schedule is cut
to a twentieth, and results go to results_quick/ so the published ones are
left alone.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

D, W0, T_MAX = 2.0, 20.0, 1.0
MU, K = 2 * D, W0 ** 2
OMEGA = np.sqrt(K - D ** 2)   # a numpy float on purpose: it sets the dtype of the basis
K_SCALE = 100.0               # every residual is divided by this before squaring

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QUICK = os.environ.get("PINN_QUICK") == "1"
RESULTS = Path(__file__).resolve().parent / ("results_quick" if QUICK else "results")


# ---------------------------------------------------------------------------
#  The reference solution and the measurements
# ---------------------------------------------------------------------------
def rk4(t, mu=MU, k=K, u0=1.0, v0=0.0, t_max=T_MAX):
    """Fourth-order Runge-Kutta at dt = 1e-4, interpolated onto t.
    It generates every measurement and grades every result."""
    dt = 1e-4
    n = int(t_max / dt) + 1
    ts = np.linspace(0, t_max, n)
    xs = np.empty(n)
    x, v = float(u0), float(v0)
    for i in range(n):
        xs[i] = x
        a1 = -mu * v - k * x
        x2, v2 = x + 0.5 * dt * v, v + 0.5 * dt * a1
        a2 = -mu * v2 - k * x2
        x3, v3 = x + 0.5 * dt * v2, v + 0.5 * dt * a2
        a3 = -mu * v3 - k * x3
        x4, v4 = x + dt * v3, v + dt * a3
        a4 = -mu * v4 - k * x4
        x += dt / 6 * (v + 2 * v2 + 2 * v3 + v4)
        v += dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
    return np.interp(t, ts, xs)


def gapped_data(seed=1, n=15, noise=0.05, gap=(0.45, 0.65)):
    """Case 2: n noisy measurements, none allowed inside `gap`."""
    rng = np.random.default_rng(seed)
    cand = rng.uniform(0, T_MAX, 2000)
    cand = cand[(cand < gap[0]) | (cand > gap[1])]
    t = np.sort(cand[:n])
    return t, rk4(t) + noise * rng.standard_normal(n)


def scattered_data(seed=1, n=20, noise=0.05):
    """Case 3: n noisy measurements anywhere in the window."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, T_MAX, n))
    return t, rk4(t) + noise * rng.standard_normal(n)


def record_gap(t):
    """The widest hole in the record. It is wider than the window the sampler
    was excluded from, because the nearest points landed outside that window."""
    i = int(np.argmax(np.diff(t)))
    return float(t[i]), float(t[i + 1])


def column(x):
    """A numpy array as a float32 column on the device."""
    return torch.tensor(x, dtype=torch.float32, device=DEVICE).view(-1, 1)


def grid(n, requires_grad=False, t_max=T_MAX):
    """n evenly spaced times on the device: collocation points or a test grid."""
    t = torch.linspace(0, t_max, n, device=DEVICE).view(-1, 1)
    return t.requires_grad_(True) if requires_grad else t


# ---------------------------------------------------------------------------
#  The network and the loss
# ---------------------------------------------------------------------------
class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


ACTIVATIONS = {"tanh": nn.Tanh, "sin": Sin, "relu": nn.ReLU}


def mlp(hidden=32, layers=3, activation="tanh"):
    """A fully connected t -> u network."""
    act = ACTIVATIONS[activation]
    mods = [nn.Linear(1, hidden), act()]
    for _ in range(layers - 1):
        mods += [nn.Linear(hidden, hidden), act()]
    mods.append(nn.Linear(hidden, 1))
    return nn.Sequential(*mods).to(DEVICE)


class HardIC(nn.Module):
    """u = 1 + t^2 N(t), which satisfies u(0) = 1 and u'(0) = 0 for any weights."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, t):
        return 1.0 + t ** 2 * self.net(t)


def residual(net, t, mu=MU, k=K):
    """u'' + mu u' + k u at t, by automatic differentiation, in raw units.
    mu and k may be numbers or trainable tensors."""
    u = net(t)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t, torch.ones_like(u_t), create_graph=True)[0]
    return u_tt + mu * u_t + k * u


def physics_loss(net, t, mu=MU, k=K, k_scale=K_SCALE):
    return torch.mean((residual(net, t, mu, k) / k_scale) ** 2)


def data_loss(net, t, u):
    return torch.mean((net(t) - u) ** 2)


# ---------------------------------------------------------------------------
#  Training: Adam on a cosine schedule, then L-BFGS
# ---------------------------------------------------------------------------
def adam(params, loss_fn, steps, lr=5e-3, eta_min=1e-4, record=None):
    """steps + 1 Adam updates. eta_min=None turns the cosine schedule off.
    record(loss), if given, is called after every update."""
    steps = max(1, steps // 20) if QUICK else steps
    opt = torch.optim.Adam(params, lr=lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=eta_min)
             if eta_min is not None else None)
    for _ in range(steps + 1):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        if record is not None:
            record(loss)


def lbfgs(params, loss_fn, steps, record=None):
    """L-BFGS with a strong Wolfe line search, for up to `steps` iterations."""
    steps = max(1, steps // 20) if (QUICK and steps > 0) else steps
    if steps <= 0:
        return
    opt = torch.optim.LBFGS(params, max_iter=steps, history_size=50, tolerance_grad=1e-12,
                            tolerance_change=1e-14, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        if record is not None:
            record(loss)
        return loss

    opt.step(closure)


def fit(params, loss_fn, adam_steps, lbfgs_steps, lr=5e-3, eta_min=1e-4, record=None):
    adam(params, loss_fn, adam_steps, lr, eta_min, record)
    lbfgs(params, loss_fn, lbfgs_steps, record)


def predict(net, t):
    with torch.no_grad():
        return net(t).cpu().numpy().flatten()


def warm_start(make_net, t_phys, t_data=None, u_data=None, steps=5):
    """Take the GPU off its cold path before the first measured training.

    On a CUDA device the first training in a process runs before cuBLAS holds a
    handle, and takes a different arithmetic route to the same sums. Where it
    lands depends on what the process happened to do first: at equal weights in
    Case 2, cold runs have come out at 2.492e-4 and at 2.660e-4. Every warm run
    comes out at 2.707e-4, bit for bit. This runs a few discarded optimizer
    steps on a throwaway network of the caller's own architecture, then
    restores the random-number state, so the measured runs cannot tell it
    happened. On a CPU there is no cold path, and it does nothing.
    """
    if DEVICE.type != "cuda":
        return
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all()
    net = make_net()
    opt = torch.optim.Adam(list(net.parameters()), lr=5e-3)
    for _ in range(steps):
        opt.zero_grad()
        loss = physics_loss(net, t_phys)
        if t_data is not None:
            loss = loss + data_loss(net, t_data, u_data)
        loss.backward()
        opt.step()
    del net, opt, loss
    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state_all(cuda_state)


# ---------------------------------------------------------------------------
#  Classical least squares on the closed form  u = e^{-dt} (A cos wt + B sin wt)
# ---------------------------------------------------------------------------
def decay_basis(t, d, w):
    e = np.exp(-d * t)
    return np.stack([e * np.cos(w * t), e * np.sin(w * t)], axis=1)


def fit_amplitudes(t, u, d, w):
    """A and B with d and omega fixed: ordinary linear least squares."""
    coef, *_ = np.linalg.lstsq(decay_basis(t, d, w), u, rcond=None)
    return coef


def fit_damping(t, u, k=K, n=4000):
    """d, A and B with k known: profile d over a grid, A and B linear at each."""
    best = (np.inf, None, None, None)
    for d in np.linspace(1e-3, np.sqrt(k) * 0.999, n):
        w = np.sqrt(k - d * d)
        P = decay_basis(t, d, w)
        coef, *_ = np.linalg.lstsq(P, u, rcond=None)
        r = float(np.sum((P @ coef - u) ** 2))
        if r < best[0]:
            best = (r, d, w, coef)
    return best[1], best[2], best[3]


def _grid_search(t, u, ds, ws):
    best = (np.inf, None, None, None)
    for d in ds:
        e = np.exp(-d * t)
        for w in ws:
            P = np.stack([e * np.cos(w * t), e * np.sin(w * t)], axis=1)
            coef, *_ = np.linalg.lstsq(P, u, rcond=None)
            r = float(np.sum((P @ coef - u) ** 2))
            if r < best[0]:
                best = (r, d, w, coef)
    return best


def fit_free(t, u, w_range=(8.0, 32.0), n_w=260, n_refine=120, dw=0.4):
    """d, omega, A and B all free: a coarse grid over (d, omega), then a finer
    one around its best point. A and B are linear at every grid point."""
    _, d, w, _ = _grid_search(t, u, np.linspace(0.05, 8.0, 160), np.linspace(*w_range, n_w))
    _, d, w, coef = _grid_search(t, u, np.linspace(max(0.01, d - 0.25), d + 0.25, n_refine),
                                 np.linspace(w - dw, w + dw, n_refine))
    return d, w, coef


# ---------------------------------------------------------------------------
#  Output
# ---------------------------------------------------------------------------
def environment():
    return dict(torch=torch.__version__, numpy=np.__version__, device=str(DEVICE))


def save_json(name, payload):
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / name, "w") as f:
        json.dump(dict(payload, env=environment()), f, indent=2)
    print(f"Saved {RESULTS.name}/{name}")


def save_figure(fig, name, dpi=120):
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / name, dpi=dpi, bbox_inches="tight")
    print(f"Saved {RESULTS.name}/{name}")
