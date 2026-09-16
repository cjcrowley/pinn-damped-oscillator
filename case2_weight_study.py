"""
========================================================================
  CASE 2 WEIGHT STUDY -- how much of the PINN's loss was the weighting?
========================================================================
pinn_data_inverse.py DEMO 1 compares a PINN against a known-form least
squares fit on 15 noisy points with a gap, with mu and k KNOWN. It runs
at data_weight = 20 and, for years, NO physics weight at all -- the term
was written without one and inherited a 1.0. That pairing was never
swept. This script sweeps it.

The point is not only "what is the best lambda_phys". Anybody can find
that with an answer key in hand. The point is what a person WITHOUT the
answer key would have picked, and how far off that lands. So the runs
split into two groups:

  A PRIORI  -- choosable with no ground truth whatsoever
     as-written     lam_data=20, lam_phys=0.0625  (the accident, see below)
     naive equal    lam_data=1,  lam_phys=1       (the obvious first thing)
     equal lambdas  lam_data=20, lam_phys=20      (equal weights, data kept)
     balanced       lam_data=20, lam_phys set so the two weighted loss
                    terms are equal at the end of a pilot run -- a real
                    practitioner heuristic, no truth required

  ORACLE    -- needs the answer key, so nobody gets it in a real problem
     optimal        lam_data=20, lam_phys = argmin(MSE vs truth) over a scan

NOTE ON NORMALISATION. Every script in this study now divides the residual
by the same fixed constant K_SCALE = 100, so a lambda_phys means the same
thing everywhere. It did not always: pinn_data_inverse.py used to divide by
k = 400 instead. That constant is squared and absorbed into lambda_phys, so
its old implicit weight of 1.0 is a weight of 1/16 = 0.0625 in these units.
That is the "as-written" row below.

Run:  python case2_weight_study.py [--quick]
Out:  case2_weight_study_result.png, case2_weight_study.json
========================================================================
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- setup copied verbatim from pinn_data_inverse.py so the problem,
# ---- the data and the network are the same object of study ----------
D_TRUE, W0_TRUE, TMAX = 2.0, 20.0, 1.0
MU_TRUE, K_TRUE = 2 * D_TRUE, W0_TRUE ** 2
N_DATA, NOISE, SEED = 15, 0.05, 1
GAP = (0.45, 0.65)
HIDDEN, LAYERS = 32, 3
LR, N_COLLOC = 5e-3, 200

# Fixed constant the ODE residual is divided by -- the same in every script here.
K_SCALE = 100.0

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true",
                    help="short schedule, for timing and smoke tests only")
args = parser.parse_args()
ADAM_EPOCHS = 1000 if args.quick else 10000
LBFGS_STEPS = 150 if args.quick else 1500

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={dev}  adam={ADAM_EPOCHS}  lbfgs={LBFGS_STEPS}")


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

# The shaded band must span the hole in the RECORD, which is wider than the
# window the sampler was excluded from: the nearest points landed outside it.
_gi = int(np.argmax(np.diff(t_data_np)))
gap_lo, gap_hi = float(t_data_np[_gi]), float(t_data_np[_gi + 1])

# Score the gap over the REAL hole in the record, the same interval the plots
# shade -- not the narrower window the sampler was excluded from.
in_gap = (t_test_np >= gap_lo) & (t_test_np <= gap_hi)
print(f"data: {N_DATA} points, widest gap "
      f"{t_data_np[np.argmax(np.diff(t_data_np))]:.3f}..."
      f"{t_data_np[np.argmax(np.diff(t_data_np)) + 1]:.3f}")


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
    """Loss = lam_data*L_data + lam_phys*L_phys, mu and k fixed at truth."""
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
          f"resid_rms={out['resid_rms']:.2e}")
    return out


# ======================================================================
#  1. The oracle scan -- needs the truth, so it is not a real strategy
# ======================================================================
# Warm start (warm_start.py). The first scan point used to be the first training
# in the process and ran on the GPU's cold path -- the published 0.0625 row was
# that cold run. Every other row always ran warm.
from warm_start import warm_start
warm_start(make_net, t_phys, mu=MU_TRUE, k=K_TRUE, k_scale=K_SCALE, t_data=t_data, u_data=u_data)

print("\nORACLE SCAN over lambda_phys (lam_data fixed at 20)")
# 30, 40, 50 and 80 added: the original half-decade grid jumped from 20 straight
# to 60, and so could not see where between them the best setting sits.
SCAN = [0.0625, 0.2, 0.6, 2.0, 6.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0,
        200.0, 600.0, 2000.0, 6000.0]
scan_runs = [train(20.0, lp, tag=f"scan {lp:g}") for lp in SCAN]
best = min(scan_runs, key=lambda r: r["mse"])
print(f"  -> best lambda_phys = {best['lam_phys']:g}  (MSE {best['mse']:.3e})")

# ======================================================================
#  2. The a priori choices -- what you could actually have picked
# ======================================================================
print("\nA PRIORI CHOICES (no ground truth used)")
shipped = next(r for r in scan_runs if r["lam_phys"] == 0.0625)
shipped = dict(shipped, tag="as-written")
print(f"  [{'as-written':>16}] reusing scan point lam_phys=0.0625")

naive = train(1.0, 1.0, tag="naive equal")
equal_lam = train(20.0, 20.0, tag="equal lambdas")

# "balanced": set lam_phys so the two WEIGHTED terms match at the end of a
# pilot run. Uses only quantities visible during training.
pilot = train(20.0, 0.0625, tag="pilot")
lam_balanced = 20.0 * pilot["l_data"] / max(pilot["l_phys"], 1e-30)
print(f"  balanced heuristic: lam_phys = 20 * {pilot['l_data']:.3e} / "
      f"{pilot['l_phys']:.3e} = {lam_balanced:.4g}")
balanced = train(20.0, lam_balanced, tag="balanced")

# free-form network, for the same reference line the original figure used
freeform = train(20.0, 0.0, use_physics=False, tag="free-form NN")

# ======================================================================
#  3. Report
# ======================================================================
apriori = [naive, shipped, equal_lam, balanced]
oracle = dict(best, tag="ORACLE optimal")

print("\n" + "=" * 78)
print(f"{'run':>18} {'lam_data':>9} {'lam_phys':>10} {'MSE':>11} {'gap MSE':>11} {'x LSQ':>7}")
print("-" * 78)
for r in apriori + [oracle]:
    print(f"{r['tag']:>18} {r['lam_data']:>9g} {r['lam_phys']:>10.4g} "
          f"{r['mse']:>11.3e} {r['mse_gap']:>11.3e} {r['mse'] / mse_lsq:>7.1f}")
print(f"{'free-form NN':>18} {20:>9g} {0:>10g} {freeform['mse']:>11.3e} "
      f"{freeform['mse_gap']:>11.3e} {freeform['mse'] / mse_lsq:>7.1f}")
print(f"{'known-form LSQ':>18} {'-':>9} {'-':>10} {mse_lsq:>11.3e} "
      f"{mse_lsq_gap:>11.3e} {1.0:>7.1f}")
print("=" * 78)

payload = dict(
    mse_lsq=mse_lsq, mse_lsq_gap=mse_lsq_gap,
    t=t_test_np.tolist(), u_truth=u_truth.tolist(), u_lsq=u_lsq.tolist(),
    t_data=t_data_np.tolist(), u_data=u_data_np.tolist(),
    scan=[dict(r, u=r["u"].tolist()) for r in scan_runs],
    apriori=[dict(r, u=r["u"].tolist()) for r in apriori],
    oracle=dict(oracle, u=oracle["u"].tolist()),
    freeform=dict(freeform, u=freeform["u"].tolist()),
    env=dict(torch=torch.__version__, numpy=np.__version__, device=str(dev),
             adam=ADAM_EPOCHS, lbfgs=LBFGS_STEPS),
)
with open("case2_weight_study.json", "w") as f:
    json.dump(payload, f, indent=2)

# ---------------------------- figure ----------------------------------
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 4, hspace=0.38, wspace=0.28)

# top row: the four a priori reconstructions
for j, r in enumerate(apriori):
    a = fig.add_subplot(gs[0, j])
    a.axvspan(gap_lo, gap_hi, color="tab:orange", alpha=0.10)
    a.plot(t_test_np, u_truth, color="tab:gray", lw=3, alpha=0.6, label="Truth")
    a.plot(t_test_np, r["u"], "--", color="tab:blue", lw=2, label="PINN")
    a.plot(t_test_np, u_lsq, "-", color="tab:green", lw=1.3, label="Known-form LSQ")
    a.scatter(t_data_np, u_data_np, s=16, color="black", zorder=5)
    a.set_title(f"{r['tag']}\n"
                r"$\lambda_{data}$=" + f"{r['lam_data']:g}  "
                r"$\lambda_{phys}$=" + f"{r['lam_phys']:.4g}\n"
                f"MSE {r['mse']:.1e}  ({r['mse'] / mse_lsq:.1f}x LSQ)", fontsize=9)
    a.set_xlabel("t")
    a.set_ylim(-1.15, 1.35)
    a.grid(alpha=0.3)
    if j == 0:
        a.set_ylabel("u(t)")
        a.legend(fontsize=7, loc="upper right")

# bottom left: the scan, with the a priori choices marked on it
a = fig.add_subplot(gs[1, :2])
lp = [r["lam_phys"] for r in scan_runs]
a.loglog(lp, [r["mse"] for r in scan_runs], "o-", color="tab:blue",
         label=r"PINN, $\lambda_{data}$=20")
a.axhline(mse_lsq, ls="--", color="tab:green", lw=2,
          label=f"known-form LSQ (2 params) = {mse_lsq:.1e}")
a.plot([oracle["lam_phys"]], [oracle["mse"]], "*", ms=20, color="tab:red", zorder=6,
       label=f"oracle best, $\\lambda_{{phys}}$={oracle['lam_phys']:.0f} "
             f"(needs the truth)")
for r, colour in ((shipped, "tab:purple"), (equal_lam, "tab:orange"),
                  (balanced, "tab:brown")):
    a.plot([r["lam_phys"]], [r["mse"]], "s", ms=9, color=colour, zorder=6,
           label=f"{r['tag']}, $\\lambda_{{phys}}$={r['lam_phys']:.4g}")
a.set_xlabel(r"$\lambda_{phys}$   (residual normalised by K_SCALE=100)")
a.set_ylabel("solution MSE vs truth")
a.set_title("What the weight is worth, and what you could have known", fontsize=11)
a.legend(fontsize=8, loc="upper left")
a.grid(alpha=0.3, which="both")

# bottom right: the two things that trade off
a = fig.add_subplot(gs[1, 2])
a.loglog(lp, [r["l_data"] for r in scan_runs], "o-", color="tab:red", label="data misfit")
a.loglog(lp, [r["l_phys"] for r in scan_runs], "s-", color="tab:purple", label="physics residual")
a.set_xlabel(r"$\lambda_{phys}$")
a.set_ylabel("unweighted loss term")
a.set_title("The trade", fontsize=11)
a.legend(fontsize=8)
a.grid(alpha=0.3, which="both")

a = fig.add_subplot(gs[1, 3])
a.semilogx(lp, [r["peak"] for r in scan_runs], "o-", color="tab:cyan")
a.axhline(1.0, ls="--", color="tab:gray", label="true peak |u| = 1")
a.set_xlabel(r"$\lambda_{phys}$")
a.set_ylabel("peak |u| of the reconstruction")
a.set_title("Collapse toward $u\\equiv0$", fontsize=11)
a.legend(fontsize=8)
a.grid(alpha=0.3, which="both")

fig.suptitle("Case 2 (mu and k known, 15 noisy points with a gap): "
             "the physics weight was never swept", fontsize=13)
fig.savefig("case2_weight_study_result.png", dpi=120, bbox_inches="tight")
print("\nSaved case2_weight_study_result.png and case2_weight_study.json")
