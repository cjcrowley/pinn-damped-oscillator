"""
========================================================================
  IS THE RUN-TO-RUN SCATTER A BUG, OR THE HARDWARE?
========================================================================
case2_weight_study.py trains the identical configuration twice -- once as
a scan point, once as the pilot for the balanced heuristic. On the GPU
those two came back 6% apart. On the CPU they were bit-identical.

Three things could explain that, and only one of them is benign:

  (a) the measurements are being redrawn between runs      -> a bug
  (b) the network is being initialised differently         -> a bug
  (c) the arithmetic itself is not reproducible on the GPU -> the hardware

This settles it. Each repeat re-derives the data and the initial weights
from the same seed and FINGERPRINTS them, so (a) and (b) become visible
rather than assumed. Then the same configuration is trained several times
and the spread of the answers is reported.

The third mode is a control: torch.use_deterministic_algorithms(True),
with CUBLAS_WORKSPACE_CONFIG set before torch is imported, forces cuBLAS
onto reproducible reduction orders.

WHAT IT ACTUALLY FOUND. The fingerprints are identical everywhere, so (a)
and (b) are both out. But the deterministic mode does NOT remove the
scatter, which rules out ordinary kernel non-determinism too. The pattern
is narrower than that: the FIRST training in a process disagrees with
every one after it, and repeats 2..n are bit-identical. Run with --warmup
-- a few optimiser steps thrown away before the repeats start -- and the
scatter disappears completely. The first pass happens before cuBLAS holds
a handle, and takes a different route to the same sum.

So it is (c), but specifically a cold-start effect, not a per-call one.
The first version of this check trained 0.0625 before 20, so every repeat at
20 was already warm, and it concluded that 20 was immune. It is not. Put 20
first (--configs 20) and its cold run disagrees with its warm repeats too:
solution MSE 2.660e-4 cold against 2.707e-4 warm. The under-constrained
setting shows the effect more (about 6%), but equal weights is not exempt,
which is why every study script now calls warm_start() before its first
measured training.

Two configurations are tested on purpose:
  lam_phys = 0.0625  as first written -- physics barely constrains
  lam_phys = 20      equal weights    -- physics pins the curve down

Run:  python determinism_check.py --mode cpu
      python determinism_check.py --mode cuda
      python determinism_check.py --mode cuda-det
      python determinism_check.py --mode cuda --configs 20            (20 cold, then warm)
      python determinism_check.py --mode cuda --configs 20 --warmup   (20 warm throughout)
========================================================================
"""
import argparse
import hashlib
import os

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["cpu", "cuda", "cuda-det"], required=True)
parser.add_argument("--reps", type=int, default=3)
parser.add_argument("--configs", type=str, default="0.0625,20",
                    help="comma-separated lambda_phys values, trained in this order; the "
                         "first repeat of the first one is the cold run unless --warmup")
parser.add_argument("--warmup", action="store_true",
                    help="throw away one short training first, so the cuBLAS handle and "
                         "CUDA context are already initialised before the repeats start")
args = parser.parse_args()

# MUST happen before torch is imported, or cuBLAS will already hold a handle.
if args.mode == "cuda-det":
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
import torch.nn as nn

if args.mode == "cuda-det":
    torch.use_deterministic_algorithms(True)

D_TRUE, W0_TRUE, TMAX = 2.0, 20.0, 1.0
MU_TRUE, K_TRUE = 2 * D_TRUE, W0_TRUE ** 2
N_DATA, NOISE, SEED = 15, 0.05, 1
GAP = (0.45, 0.65)
HIDDEN, LAYERS = 32, 3
LR, N_COLLOC, K_SCALE = 5e-3, 200, 100.0
ADAM_EPOCHS, LBFGS_STEPS = 10000, 1500
LAM_DATA = 20.0
CONFIGS = [float(x) for x in args.configs.split(",")]

dev = torch.device("cpu" if args.mode == "cpu" else "cuda")
print(f"mode={args.mode}  device={dev}  torch={torch.__version__}  reps={args.reps}")


def fingerprint(x):
    """Stable hash of a tensor's exact bytes."""
    a = x.detach().cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def reference_solution(t_grid, mu, k):
    dt = 1e-4
    n = int(TMAX / dt) + 1
    ts = np.linspace(0, TMAX, n)
    xs = np.empty(n)
    x, v = 1.0, 0.0
    for i in range(n):
        xs[i] = x

        def deriv(x, v):
            return v, -mu * v - k * x

        k1x, k1v = deriv(x, v)
        k2x, k2v = deriv(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v)
        k3x, k3v = deriv(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v)
        k4x, k4v = deriv(x + dt * k3x, v + dt * k3v)
        x += dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        v += dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return np.interp(t_grid, ts, xs)


def build_data():
    """Redrawn from the seed on EVERY repeat, so a drift would show up."""
    rng = np.random.default_rng(SEED)
    lo, hi = GAP
    cand = rng.uniform(0, TMAX, 2000)
    cand = cand[(cand < lo) | (cand > hi)]
    t_np = np.sort(cand[:N_DATA])
    u_np = reference_solution(t_np, MU_TRUE, K_TRUE) + NOISE * rng.standard_normal(N_DATA)
    return (torch.tensor(t_np, dtype=torch.float32, device=dev).view(-1, 1),
            torch.tensor(u_np, dtype=torch.float32, device=dev).view(-1, 1))


t_phys = torch.linspace(0, TMAX, N_COLLOC, device=dev).view(-1, 1).requires_grad_(True)
t_test = torch.linspace(0, TMAX, 400, device=dev).view(-1, 1)
u_truth = reference_solution(t_test.cpu().numpy().flatten(), MU_TRUE, K_TRUE)


def make_net():
    layers = [nn.Linear(1, HIDDEN), nn.Tanh()]
    for _ in range(LAYERS - 1):
        layers += [nn.Linear(HIDDEN, HIDDEN), nn.Tanh()]
    layers += [nn.Linear(HIDDEN, 1)]
    return nn.Sequential(*layers).to(dev)


def train(lam_phys):
    t_d, u_d = build_data()
    torch.manual_seed(SEED)
    net = make_net()
    init_fp = hashlib.sha256(
        b"".join(fingerprint(p).encode() for p in net.parameters())).hexdigest()[:12]

    def loss_fn():
        l_data = torch.mean((net(t_d) - u_d) ** 2)
        u = net(t_phys)
        u_t = torch.autograd.grad(u, t_phys, torch.ones_like(u), create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t_phys, torch.ones_like(u_t), create_graph=True)[0]
        r = (u_tt + MU_TRUE * u_t + K_TRUE * u) / K_SCALE
        return LAM_DATA * l_data + lam_phys * torch.mean(r ** 2)

    params = list(net.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ADAM_EPOCHS, eta_min=1e-4)
    for _ in range(ADAM_EPOCHS + 1):
        opt.zero_grad()
        loss_fn().backward()
        opt.step()
        sch.step()
    lb = torch.optim.LBFGS(params, max_iter=LBFGS_STEPS, history_size=50,
                           tolerance_grad=1e-12, tolerance_change=1e-14,
                           line_search_fn="strong_wolfe")

    def closure():
        lb.zero_grad()
        l = loss_fn()
        l.backward()
        return l

    lb.step(closure)
    with torch.no_grad():
        u_pred = net(t_test).cpu().numpy().flatten()
    return dict(data_fp=fingerprint(t_d) + "/" + fingerprint(u_d),
                init_fp=init_fp,
                mse=float(np.mean((u_pred - u_truth) ** 2)))


if args.warmup:
    # A few optimiser steps is enough to force the context and the cuBLAS
    # handle into existence. The result is discarded.
    _t, _u = build_data()
    torch.manual_seed(SEED)
    _net = make_net()
    _opt = torch.optim.Adam(list(_net.parameters()), lr=LR)
    for _ in range(5):
        _opt.zero_grad()
        _l = torch.mean((_net(_t) - _u) ** 2)
        _uu = _net(t_phys)
        _ut = torch.autograd.grad(_uu, t_phys, torch.ones_like(_uu), create_graph=True)[0]
        _utt = torch.autograd.grad(_ut, t_phys, torch.ones_like(_ut), create_graph=True)[0]
        (_l + torch.mean(((_utt + MU_TRUE * _ut + K_TRUE * _uu) / K_SCALE) ** 2)).backward()
        _opt.step()
    print("(warmup training discarded)\n")

print()
for lam_phys in CONFIGS:
    rows = [train(lam_phys) for _ in range(args.reps)]
    mses = [r["mse"] for r in rows]
    data_fps = {r["data_fp"] for r in rows}
    init_fps = {r["init_fp"] for r in rows}
    spread = (max(mses) - min(mses)) / min(mses) * 100
    label = "as-written" if lam_phys == 0.0625 else "best"
    print(f"lam_phys = {lam_phys:<8g} ({label})")
    print(f"   data fingerprint : {'IDENTICAL' if len(data_fps) == 1 else 'DIFFERS!!'}"
          f"  {sorted(data_fps)}")
    print(f"   init fingerprint : {'IDENTICAL' if len(init_fps) == 1 else 'DIFFERS!!'}"
          f"  {sorted(init_fps)}")
    print(f"   MSE per repeat   : {['%.6e' % m for m in mses]}")
    print(f"   reproducible     : {'YES (bit-identical)' if len(set(mses)) == 1 else 'NO'}"
          f"   spread {spread:.2f}%")
    print()
