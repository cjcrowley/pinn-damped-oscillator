"""
========================================================================
  CASE 2 WEIGHT STUDY -- does any of it survive a change of seed?
========================================================================
case2_weight_study.py ran one seed, and found the best setting sitting
just below a cliff: past it the network stops solving the
problem and collapses onto u = 0, which satisfies the homogeneous
equation exactly and describes nothing. That is exactly the situation
where a single seed is worth nothing.

This re-runs the interesting weights across several seeds. Changing the
seed redraws BOTH the 15 noisy points and the network initialisation, so
each column is a genuine independent repeat of the whole experiment. The
classical fit is recomputed per seed on that seed's data, and the number
that matters is the RATIO PINN/LSQ, which is scale free.

Only the ratio lambda_phys/lambda_data has any meaning, which the
one-seed run confirms by landing equal-ratio pairs on nearly the same answer.
lambda_data is therefore pinned at 20 throughout, and
lambda_phys carries the whole story. Residual normalised by K_SCALE=100,
as everywhere else here.

Every process warm-starts before its first training (warm_start.py); without
that, the first run was on the GPU's cold path. The grid adds 30, 40, 50 and
80 around the best setting.

Run:  python case2_weight_seeds.py                          (all seeds, one process)
  or  python case2_weight_seeds.py --seeds 3 --out s3.json  (one process per seed)
      python case2_weight_seeds.py --merge s1.json s2.json s3.json s4.json s5.json
Out:  case2_weight_seeds.json
========================================================================
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn

D_TRUE, W0_TRUE, TMAX = 2.0, 20.0, 1.0
MU_TRUE, K_TRUE = 2 * D_TRUE, W0_TRUE ** 2
N_DATA, NOISE = 15, 0.05
GAP = (0.45, 0.65)
HIDDEN, LAYERS = 32, 3
LR, N_COLLOC = 5e-3, 200

# Fixed constant the ODE residual is divided by -- the same in every script here.
K_SCALE = 100.0
ADAM_EPOCHS, LBFGS_STEPS = 10000, 1500

LAM_DATA = 20.0
LAM_PHYS = [0.0625, 1.25, 6.25, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0]
SEEDS = [1, 2, 3, 4, 5]

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


t_test_np = np.linspace(0, TMAX, 400)
t_test = torch.tensor(t_test_np, dtype=torch.float32, device=dev).view(-1, 1)
u_truth = reference_solution(t_test_np, MU_TRUE, K_TRUE)
t_phys = torch.linspace(0, TMAX, N_COLLOC, device=dev).view(-1, 1).requires_grad_(True)
lo, hi = GAP


def form_basis(t, d, w):
    e = np.exp(-d * t)
    return np.stack([e * np.cos(w * t), e * np.sin(w * t)], axis=1)


def make_data(seed):
    """Same draw as pinn_data_inverse.py, but with the seed swept."""
    rng = np.random.default_rng(seed)
    cand = rng.uniform(0, TMAX, 2000)
    cand = cand[(cand < lo) | (cand > hi)]
    t_np = np.sort(cand[:N_DATA])
    u_np = reference_solution(t_np, MU_TRUE, K_TRUE) + NOISE * rng.standard_normal(N_DATA)
    return t_np, u_np


def make_net():
    layers = [nn.Linear(1, HIDDEN), nn.Tanh()]
    for _ in range(LAYERS - 1):
        layers += [nn.Linear(HIDDEN, HIDDEN), nn.Tanh()]
    layers += [nn.Linear(HIDDEN, 1)]
    return nn.Sequential(*layers).to(dev)


def train(t_np, u_np, seed, lam_phys):
    t_d = torch.tensor(t_np, dtype=torch.float32, device=dev).view(-1, 1)
    u_d = torch.tensor(u_np, dtype=torch.float32, device=dev).view(-1, 1)
    torch.manual_seed(seed)
    net = make_net()

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
    return dict(mse=float(np.mean((u_pred - u_truth) ** 2)),
                peak=float(np.max(np.abs(u_pred))))


parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS),
                    help="comma-separated seeds to run in this process")
parser.add_argument("--out", type=str, default="case2_weight_seeds.json")
parser.add_argument("--merge", nargs="+", default=None,
                    help="per-seed JSON files from separate processes, combined into one table")
args = parser.parse_args()

w_true = np.sqrt(K_TRUE - D_TRUE ** 2)
rows, lsq = {}, {}


def save(path, extra=None):
    payload = dict(lsq={str(s): v for s, v in lsq.items()},
                   runs={f"{s}_{lp:g}": v for (s, lp), v in rows.items()},
                   lam_data=LAM_DATA, seeds=sorted(lsq),
                   lam_phys=sorted({lp for (_, lp) in rows}),
                   env=dict(torch=torch.__version__, numpy=np.__version__))
    payload.update(extra or {})
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


if args.merge:
    for path in args.merge:
        part = json.load(open(path))
        lsq.update({int(s): v for s, v in part["lsq"].items()})
        for key, v in part["runs"].items():
            s, lp = key.split("_")
            rows[(int(s), float(lp))] = v
else:
    seeds_here = [int(s) for s in args.seeds.split(",")]
    # Warm start (warm_start.py) before the first measured training in this
    # process. Without it the first run was on the GPU's cold path -- in the
    # original single-process run, seed 1 at lambda_phys = 0.0625.
    from warm_start import warm_start
    _t, _u = make_data(seeds_here[0])
    warm_start(make_net, t_phys, mu=MU_TRUE, k=K_TRUE, k_scale=K_SCALE,
               t_data=torch.tensor(_t, dtype=torch.float32, device=dev).view(-1, 1),
               u_data=torch.tensor(_u, dtype=torch.float32, device=dev).view(-1, 1))
    for seed in seeds_here:
        t_np, u_np = make_data(seed)
        coef, *_ = np.linalg.lstsq(form_basis(t_np, D_TRUE, w_true), u_np, rcond=None)
        lsq[seed] = float(np.mean((form_basis(t_test_np, D_TRUE, w_true) @ coef - u_truth) ** 2))
        print(f"\nseed {seed}:  known-form LSQ MSE {lsq[seed]:.3e}", flush=True)
        for lp in LAM_PHYS:
            r = train(t_np, u_np, seed, lp)
            rows[(seed, lp)] = r
            print(f"   lam_phys={lp:>7g}  MSE={r['mse']:.3e}  "
                  f"ratio={r['mse'] / lsq[seed]:>7.2f}x  peak|u|={r['peak']:.3f}", flush=True)
            save(args.out)   # after every run, so a long job can be read part-way

seeds = sorted(lsq)
lams = sorted({lp for (_, lp) in rows})
print("\n" + "=" * 90)
print("PINN MSE / LSQ MSE, by seed        (lam_data = 20 throughout)")
print("-" * 90)
print(f"{'lam_phys':>9} " + "".join(f"{('seed ' + str(s)):>10}" for s in seeds) + f"{'median':>10}")
for lp in lams:
    ratios = [rows[(s, lp)]["mse"] / lsq[s] for s in seeds if (s, lp) in rows]
    print(f"{lp:>9g} " + "".join(f"{x:>10.2f}" for x in ratios)
          + f"{float(np.median(ratios)):>10.2f}")
print("-" * 90)
best_per_seed = {s: min((lp for lp in lams if (s, lp) in rows),
                        key=lambda lp: rows[(s, lp)]["mse"]) for s in seeds}
print("best lam_phys per seed: " + ", ".join(f"seed {s}: {best_per_seed[s]:g}" for s in seeds))
collapsed = [(s, lp) for s in seeds for lp in lams
             if (s, lp) in rows and rows[(s, lp)]["peak"] < 0.2]
print(f"collapsed runs (peak|u| < 0.2): {collapsed if collapsed else 'none'}")
print("=" * 90)
save(args.out, dict(best_per_seed={str(s): v for s, v in best_per_seed.items()}))
print(f"Saved {args.out}")
