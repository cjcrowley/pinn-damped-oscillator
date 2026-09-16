"""
========================================================================
  CASE 2 K_SCALE STUDY -- is the best weight a property of the problem,
  or of the constant the residual is divided by?
========================================================================
The physics term is  lambda_phys * mean((r / K_SCALE)^2).  Only the ratio
lambda_phys / K_SCALE^2 ever reaches the optimiser, so changing K_SCALE
should do nothing but relabel the weight axis: the best lambda_phys should
move by exactly (K_SCALE / 100)^2, and "equal weights" (lambda_phys =
lambda_data = 20) should stop being the same setting.

This runs the identical Case 2 problem -- same 15 points, same network,
same schedule as case2_weight_study.py -- with K_SCALE = 100, 200 and 400,
and 300 as a control, over one lattice of EFFECTIVE weights

    lambda_eff = lambda_phys * (100 / K_SCALE)^2 = 1.25, 2.5, ..., 320.

Why those constants. Dividing by 200 or 400 is dividing by 100 and then by
a power of two, which binary floating point does exactly, so those runs
should be bit-identical to K_SCALE = 100 at the same effective weight, not
merely close. 300 brings in a factor of three, which binary cannot
represent, so it can only agree up to rounding -- and near the collapse,
rounding is enough to change the outcome.

Every process warm-starts first (warm_start.py), so every run is on the
GPU's reproducible route.

Run one process per constant (they can share one GPU), then combine:
    python case2_kscale_study.py --k 100
    python case2_kscale_study.py --k 200
    python case2_kscale_study.py --k 400
    python case2_kscale_study.py --k 300
    python case2_kscale_study.py --plot
Out:  case2_kscale_K<K>.json per constant (in --out-dir),
      case2_kscale_study.json, case2_kscale_study_result.png
========================================================================
"""
import argparse
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- setup identical to case2_weight_study.py ------------------------
D_TRUE, W0_TRUE, TMAX = 2.0, 20.0, 1.0
MU_TRUE, K_TRUE = 2 * D_TRUE, W0_TRUE ** 2
N_DATA, NOISE, SEED = 15, 0.05, 1
GAP = (0.45, 0.65)
HIDDEN, LAYERS = 32, 3
LR, N_COLLOC = 5e-3, 200
ADAM_EPOCHS, LBFGS_STEPS = 10000, 1500
LAM_DATA = 20.0
LAM_EFF = [1.25, 2.5, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 160.0, 320.0]

parser = argparse.ArgumentParser()
parser.add_argument("--k", type=float, default=None,
                    help="constant the ODE residual is divided by")
parser.add_argument("--plot", action="store_true",
                    help="combine the per-constant results and draw the figure")
parser.add_argument("--out-dir", default=".",
                    help="directory for the per-constant JSON files")
args = parser.parse_args()


# ======================================================================
#  --plot: combine and draw, no training
# ======================================================================
if args.plot:
    colours = {100: "tab:blue", 200: "tab:orange", 400: "tab:red", 300: "tab:purple"}
    markers = {100: "o", 200: "s", 400: "D", 300: "^"}
    data = {}
    for K in (100, 200, 400, 300):
        path = os.path.join(args.out_dir, f"case2_kscale_K{K}.json")
        if os.path.exists(path):
            data[K] = json.load(open(path))
    if 100 not in data:
        raise SystemExit("need case2_kscale_K100.json in --out-dir")
    lsq = data[100]["mse_lsq"]
    print(f"classical fit MSE {lsq:.4e}")

    summary = {}
    for K, d in data.items():
        runs = sorted(d["runs"], key=lambda r: r["lam_phys"])
        best = min(runs, key=lambda r: r["mse"])
        eq = next((r for r in runs if r["lam_phys"] == LAM_DATA), None)
        summary[K] = dict(best_lam_phys=best["lam_phys"], best_ratio=best["mse"] / lsq,
                          equal_weights_ratio=eq["mse"] / lsq if eq else None,
                          predicted_best=best["lam_phys"] if K == 100 else None)
    for K in data:
        summary[K]["predicted_best"] = summary[100]["best_lam_phys"] * (K / 100.0) ** 2
        s = summary[K]
        print(f"K={K:<4g} best lambda_phys = {s['best_lam_phys']:<7g} "
              f"(predicted {s['predicted_best']:g}, {s['best_ratio']:.2f}x classical)   "
              f"equal weights: {s['equal_weights_ratio']:.2f}x classical")

    base = {round(r["lam_eff"], 6): r for r in data[100]["runs"]}
    identity = {}
    for K, d in data.items():
        if K == 100:
            continue
        pairs = [(r, base[round(r["lam_eff"], 6)]) for r in d["runs"]
                 if round(r["lam_eff"], 6) in base]
        same = sum(r["u_sha"] == b["u_sha"] for r, b in pairs)
        worst = max((abs(r["mse"] - b["mse"]) / b["mse"] for r, b in pairs), default=0.0)
        identity[K] = dict(bit_identical=same, compared=len(pairs), worst_relative_mse_difference=worst)
        print(f"K={K:g}: {same}/{len(pairs)} runs bit-identical to K=100 at the same effective "
              f"weight (largest MSE difference {worst:.2g})")

    CAP, TOP = 11.0, 16.0
    fig, a = plt.subplots(figsize=(10.5, 5.8))
    for K in (100, 200, 400, 300):
        if K not in data:
            continue
        runs = sorted(data[K]["runs"], key=lambda r: r["lam_phys"])
        lam = [r["lam_phys"] for r in runs]
        y = [min(r["mse"] / lsq, CAP) for r in runs]
        a.loglog(lam, y, "--" if K == 300 else "-", marker=markers[K], color=colours[K],
                 ms=6, lw=1.6, label=f"K = {K:g}" + ("   (control)" if K == 300 else ""))
        eq = next((r for r in runs if r["lam_phys"] == LAM_DATA), None)
        if eq:
            a.plot([LAM_DATA], [min(eq["mse"] / lsq, CAP)], marker="*", ms=17,
                   color=colours[K], mec="k", zorder=6)
        b = min(runs, key=lambda r: r["mse"])
        a.annotate(f"best\n{b['lam_phys']:g}", (b["lam_phys"], b["mse"] / lsq),
                   textcoords="offset points", xytext=(0, -24), ha="center",
                   fontsize=8.5, color=colours[K])
    a.axhline(1.0, color="tab:green", ls=":", lw=1.5, label="classical fit")
    a.axvline(LAM_DATA, color="0.45", lw=0.8)
    a.text(LAM_DATA * 1.06, 12.5 * 0.62, "equal weights\n(stars)", fontsize=8.5, color="0.3")
    a.axhspan(CAP * 0.93, TOP, color="0.92", zorder=0)
    a.text(0.012, 0.975, "collapsed onto u = 0  (about 590x the classical error)",
           transform=a.transAxes, ha="left", va="top", fontsize=8.5, color="0.35")
    a.set_ylim(0.78, TOP)
    a.set_xlabel(r"$\lambda_{phys}$   ($\lambda_{data}$ = 20 throughout)")
    a.set_ylabel("solution MSE / classical MSE")
    a.set_title("Case 2 with the residual divided by K: the best weight moves as $K^2$,\n"
                "and equal weights slides away from it", fontsize=11.5)
    a.legend(fontsize=9, loc="center left")
    a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig("case2_kscale_study_result.png", dpi=140, bbox_inches="tight")

    with open("case2_kscale_study.json", "w") as f:
        json.dump(dict(mse_lsq=lsq, lam_data=LAM_DATA, lam_eff=LAM_EFF,
                       summary={str(K): v for K, v in summary.items()},
                       identity_vs_K100={str(K): v for K, v in identity.items()},
                       runs={str(K): d["runs"] for K, d in data.items()},
                       env=data[100]["env"]), f, indent=2)
    print("Saved case2_kscale_study_result.png and case2_kscale_study.json")
    raise SystemExit(0)


# ======================================================================
#  training mode: one constant per process
# ======================================================================
if args.k is None:
    raise SystemExit("give --k 100|200|300|400, or --plot")
K_SCALE = float(args.k)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={dev}  K_SCALE={K_SCALE:g}  adam={ADAM_EPOCHS}  lbfgs={LBFGS_STEPS}")


def reference_solution(t_grid, mu, k):
    """RK4 at dt=1e-4 -- generates the data AND grades the results."""
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


# ---- the same 15 noisy points, drawn the same way ----
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)
lo, hi = GAP
cand = rng.uniform(0, TMAX, 2000)
cand = cand[(cand < lo) | (cand > hi)]
t_data_np = np.sort(cand[:N_DATA])
u_data_np = reference_solution(t_data_np, MU_TRUE, K_TRUE) \
    + NOISE * rng.standard_normal(N_DATA)

t_data = torch.tensor(t_data_np, dtype=torch.float32, device=dev).view(-1, 1)
u_data = torch.tensor(u_data_np, dtype=torch.float32, device=dev).view(-1, 1)
t_phys = torch.linspace(0, TMAX, N_COLLOC, device=dev).view(-1, 1).requires_grad_(True)
t_test = torch.linspace(0, TMAX, 400, device=dev).view(-1, 1)
t_test_np = t_test.cpu().numpy().flatten()
u_truth = reference_solution(t_test_np, MU_TRUE, K_TRUE)

_gi = int(np.argmax(np.diff(t_data_np)))
gap_lo, gap_hi = float(t_data_np[_gi]), float(t_data_np[_gi + 1])
in_gap = (t_test_np >= gap_lo) & (t_test_np <= gap_hi)


# ---- classical baseline: mu known, so only A and B are free ----
def form_basis(t, d, w):
    e = np.exp(-d * t)
    return np.stack([e * np.cos(w * t), e * np.sin(w * t)], axis=1)


w_true = np.sqrt(K_TRUE - D_TRUE ** 2)
coef_lsq, *_ = np.linalg.lstsq(form_basis(t_data_np, D_TRUE, w_true), u_data_np, rcond=None)
u_lsq = form_basis(t_test_np, D_TRUE, w_true) @ coef_lsq
mse_lsq = float(np.mean((u_lsq - u_truth) ** 2))
mse_lsq_gap = float(np.mean((u_lsq - u_truth)[in_gap] ** 2))
print(f"[known-form LSQ, 2 params] MSE {mse_lsq:.3e}  gap-MSE {mse_lsq_gap:.3e}")


def make_net():
    layers = [nn.Linear(1, HIDDEN), nn.Tanh()]
    for _ in range(LAYERS - 1):
        layers += [nn.Linear(HIDDEN, HIDDEN), nn.Tanh()]
    layers += [nn.Linear(HIDDEN, 1)]
    return nn.Sequential(*layers).to(dev)


def train(lam_data, lam_phys, use_physics=True, tag=""):
    """Loss = lam_data*L_data + lam_phys*L_phys, mu and k fixed at truth.
    Identical to case2_weight_study.py, including the work done after
    training, so the history between runs is identical too."""
    torch.manual_seed(SEED)
    net = make_net()

    def terms():
        """Return the UNWEIGHTED data and physics terms."""
        l_data = torch.mean((net(t_data) - u_data) ** 2)
        if not use_physics:
            return l_data, torch.zeros((), device=dev)
        u = net(t_phys)
        u_t = torch.autograd.grad(u, t_phys, torch.ones_like(u), create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t_phys, torch.ones_like(u_t), create_graph=True)[0]
        r = (u_tt + MU_TRUE * u_t + K_TRUE * u) / K_SCALE
        return l_data, torch.mean(r ** 2)

    def loss_fn():
        l_data, l_phys = terms()
        return lam_data * l_data + lam_phys * l_phys

    params = list(net.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ADAM_EPOCHS, eta_min=1e-4)
    for _ in range(ADAM_EPOCHS + 1):
        opt.zero_grad()
        loss_fn().backward()
        opt.step()
        sch.step()

    if LBFGS_STEPS > 0:
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
    l_data, l_phys = terms()
    # leftover residual in RAW units, against equation terms of order k*u ~ 400
    u = net(t_phys)
    u_t = torch.autograd.grad(u, t_phys, torch.ones_like(u), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t_phys, torch.ones_like(u_t), create_graph=True)[0]
    resid = (u_tt + MU_TRUE * u_t + K_TRUE * u).detach().cpu().numpy().flatten()

    out = dict(tag=tag, lam_data=float(lam_data), lam_phys=float(lam_phys),
               mse=float(np.mean((u_pred - u_truth) ** 2)),
               mse_gap=float(np.mean((u_pred - u_truth)[in_gap] ** 2)),
               l_data=float(l_data), l_phys=float(l_phys),
               resid_rms=float(np.sqrt(np.mean(resid ** 2))),
               peak=float(np.max(np.abs(u_pred))), u=u_pred)
    print(f"  [{tag:>16}] lam_data={lam_data:<6g} lam_phys={lam_phys:<9g} "
          f"MSE={out['mse']:.3e}  gap={out['mse_gap']:.3e}  "
          f"resid_rms={out['resid_rms']:.2e}", flush=True)
    return out


# Warm start before the first measured run. The residual scale inside the
# warm-up is a literal 100, so the warm-up is the same computation in every
# process whatever K_SCALE is.
from warm_start import warm_start
warm_start(make_net, t_phys, mu=MU_TRUE, k=K_TRUE, k_scale=100.0, t_data=t_data, u_data=u_data)

lams = [le * (K_SCALE / 100.0) ** 2 for le in LAM_EFF]
if LAM_DATA not in lams:
    lams.append(LAM_DATA)   # equal weights; last, so the lattice keeps the same run order

os.makedirs(args.out_dir, exist_ok=True)
out_path = os.path.join(args.out_dir, f"case2_kscale_K{K_SCALE:g}.json")
runs = []
for lp in lams:
    r = train(LAM_DATA, lp, tag=f"K{K_SCALE:g} lam {lp:g}")
    u = np.asarray(r.pop("u"), dtype=np.float32)
    r["u_sha"] = hashlib.sha256(u.tobytes()).hexdigest()[:16]
    r["lam_eff"] = lp * (100.0 / K_SCALE) ** 2
    runs.append(r)
    with open(out_path, "w") as f:
        json.dump(dict(K=K_SCALE, lam_data=LAM_DATA, mse_lsq=mse_lsq, mse_lsq_gap=mse_lsq_gap,
                       adam=ADAM_EPOCHS, lbfgs=LBFGS_STEPS, runs=runs,
                       env=dict(torch=torch.__version__, numpy=np.__version__,
                                device=str(dev))), f, indent=1)
print(f"Saved {out_path}")
