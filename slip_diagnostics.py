"""
========================================================================
  THE DIAGNOSTICS THAT WERE NEVER A SCRIPT
========================================================================
Two sections of the write-up -- "where the phase slip comes from" and
"what survives at the best setting" -- rested on analysis done by hand
and never folded back into code. This is that code.

It rebuilds the same twenty noisy points slippage_weight_experiment.py
uses, trains the same two configurations, and then asks the questions
the prose asks:

  UNDER-WEIGHTED  (lam_data=30, lam_phys=1)
    - where does the network cross zero, and how late is it?
    - is that lag consistent with a frequency error? (a frequency error
      accumulates: same sign, growing linearly, never recovering)
    - fit the closed form to the NETWORK'S OWN CURVE, all four
      parameters free. If the best member of the solution family still
      cannot describe it, the curve is not a solution of anything.

  BEST            (lam_data=30, lam_phys=30)
    - error relative to the LOCAL signal amplitude, by quarter, since
      MSE over a decaying signal reports on the loud beginning
    - take the recovered coefficients and the network's own initial
      state, integrate that IVP properly, and see how far the network
      drifts from the exact solution of its own equation
    - what initial conditions did the reconstruction actually imply?

SIGN CONVENTION, stated because the original had it backwards: a lag is
POSITIVE when the network crosses zero LATER than the truth. A network
running slow (omega below the true omega) therefore shows a positive,
growing lag.

Run:  python slip_diagnostics.py
Out:  slip_diagnostics_result.png, bestcase_diagnostics_result.png,
      slip_diagnostics.json
========================================================================
"""
import json

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- identical setup to slippage_weight_experiment.py ----
D_TRUE, W0_TRUE, TMAX = 2.0, 20.0, 1.0
MU_TRUE, K_TRUE = 2 * D_TRUE, W0_TRUE ** 2
N_DATA, NOISE, SEED = 20, 0.05, 1
DATA_WEIGHT, K_SCALE = 30.0, 100.0
ADAM_EPOCHS, LR, LBFGS_STEPS, N_COLLOC = 6000, 5e-3, 1000, 200
W_TRUE = np.sqrt(K_TRUE - D_TRUE ** 2)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(SEED)
print(f"device={dev}  torch={torch.__version__}")


def reference_solution(t_grid, mu, k):
    dt = 1e-4
    n = int(TMAX / dt) + 1
    ts = np.linspace(0, TMAX, n)
    xs = np.empty(n)
    x, v = 1.0, 0.0
    for i in range(n):
        xs[i] = x
        a1v = -mu * v - k * x
        x2, v2 = x + 0.5 * dt * v, v + 0.5 * dt * a1v
        a2v = -mu * v2 - k * x2
        x3, v3 = x + 0.5 * dt * v2, v + 0.5 * dt * a2v
        a3v = -mu * v3 - k * x3
        x4, v4 = x + dt * v3, v + dt * a3v
        a4v = -mu * v4 - k * x4
        x += dt / 6 * (v + 2 * v2 + 2 * v3 + v4)
        v += dt / 6 * (a1v + 2 * a2v + 2 * a3v + a4v)
    return np.interp(t_grid, ts, xs)


def integrate_ivp(t_grid, mu, k, u0, v0):
    """RK4 for u'' + mu u' + k u = 0 from an ARBITRARY initial state."""
    dt = 1e-4
    n = int(TMAX / dt) + 1
    ts = np.linspace(0, TMAX, n)
    xs = np.empty(n)
    x, v = float(u0), float(v0)
    for i in range(n):
        xs[i] = x
        a1v = -mu * v - k * x
        x2, v2 = x + 0.5 * dt * v, v + 0.5 * dt * a1v
        a2v = -mu * v2 - k * x2
        x3, v3 = x + 0.5 * dt * v2, v + 0.5 * dt * a2v
        a3v = -mu * v3 - k * x3
        x4, v4 = x + dt * v3, v + dt * a3v
        a4v = -mu * v4 - k * x4
        x += dt / 6 * (v + 2 * v2 + 2 * v3 + v4)
        v += dt / 6 * (a1v + 2 * a2v + 2 * a3v + a4v)
    return np.interp(t_grid, ts, xs)


# ---- the same twenty noisy points ----
t_data_np = np.sort(rng.uniform(0, TMAX, N_DATA))
u_data_np = reference_solution(t_data_np, MU_TRUE, K_TRUE) + NOISE * rng.standard_normal(N_DATA)
t_data = torch.tensor(t_data_np, dtype=torch.float32, device=dev).view(-1, 1)
u_data = torch.tensor(u_data_np, dtype=torch.float32, device=dev).view(-1, 1)
t_phys = torch.linspace(0, TMAX, N_COLLOC, device=dev).view(-1, 1).requires_grad_(True)

t_fine = np.linspace(0, TMAX, 4001)
u_truth = reference_solution(t_fine, MU_TRUE, K_TRUE)
t_test = torch.tensor(t_fine, dtype=torch.float32, device=dev).view(-1, 1)


# ---- classical fit: d, omega, A, B all free ----
def basis(t, d, w):
    e = np.exp(-d * t)
    return np.stack([e * np.cos(w * t), e * np.sin(w * t)], axis=1)


def _grid(t, u, ds, ws):
    best = (np.inf, None, None, None)
    for d in ds:
        e = np.exp(-d * t)
        for w in ws:
            P = np.stack([e * np.cos(w * t), e * np.sin(w * t)], axis=1)
            c, *_ = np.linalg.lstsq(P, u, rcond=None)
            r = float(np.sum((P @ c - u) ** 2))
            if r < best[0]:
                best = (r, d, w, c)
    return best


def classical_fit(t, u):
    _, d, w, _ = _grid(t, u, np.linspace(0.05, 8, 160), np.linspace(8, 32, 260))
    _, d, w, c = _grid(t, u, np.linspace(max(0.01, d - 0.25), d + 0.25, 120),
                             np.linspace(w - 0.4, w + 0.4, 120))
    return d, w, c


def make_net():
    layers = [nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(),
              nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1)]
    return nn.Sequential(*layers).to(dev)


def train(physics_weight):
    """Same trainer as slippage_weight_experiment.py, mu and k trainable."""
    torch.manual_seed(SEED)
    net = make_net()
    mu_p = nn.Parameter(torch.tensor(1.0, device=dev))
    kp = nn.Parameter(torch.tensor(100.0 / K_SCALE, device=dev))
    params = list(net.parameters()) + [mu_p, kp]

    def loss_fn():
        data = DATA_WEIGHT * torch.mean((net(t_data) - u_data) ** 2)
        u = net(t_phys)
        u_t = torch.autograd.grad(u, t_phys, torch.ones_like(u), create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t_phys, torch.ones_like(u_t), create_graph=True)[0]
        r = (u_tt + mu_p * u_t + K_SCALE * kp * u) / K_SCALE
        return data + physics_weight * torch.mean(r ** 2)

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

    mu_hat, k_hat = mu_p.item(), (K_SCALE * kp).item()
    with torch.no_grad():
        u_pinn = net(t_test).cpu().numpy().flatten()

    # the network's own initial state, by autograd
    t0 = torch.zeros(1, 1, device=dev, requires_grad=True)
    u0 = net(t0)
    v0 = torch.autograd.grad(u0, t0, torch.ones_like(u0), create_graph=False)[0]
    u0, v0 = float(u0.item()), float(v0.item())

    # leftover residual in RAW units at the collocation points
    u = net(t_phys)
    u_t = torch.autograd.grad(u, t_phys, torch.ones_like(u), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t_phys, torch.ones_like(u_t), create_graph=True)[0]
    resid = (u_tt + mu_hat * u_t + k_hat * u).detach().cpu().numpy().flatten()
    term = np.abs(k_hat * u.detach().cpu().numpy().flatten())

    return dict(mu=mu_hat, k=k_hat, u=u_pinn, u0=u0, v0=v0,
                resid_rms=float(np.sqrt(np.mean(resid ** 2))),
                term_rms=float(np.sqrt(np.mean(term ** 2))),
                mse=float(np.mean((u_pinn - u_truth) ** 2)))


def zero_crossings(t, y):
    """Times where y changes sign, linearly interpolated."""
    s = np.sign(y)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    out = []
    for i in idx:
        f = y[i] / (y[i] - y[i + 1])
        out.append(t[i] + f * (t[i + 1] - t[i]))
    return np.array(out)


def omega_of(mu, k):
    d = mu / 2
    return float(np.sqrt(max(k - d * d, 1e-12)))


# ======================================================================
print("\nclassical fit (d, omega, A, B all free)")
d_c, w_c, coef_c = classical_fit(t_data_np, u_data_np)
u_class = basis(t_fine, d_c, w_c) @ coef_c
mse_class = float(np.mean((u_class - u_truth) ** 2))
mu_c, k_c = 2 * d_c, w_c ** 2 + d_c ** 2
print(f"  mu={mu_c:.4f} ({100*(mu_c-MU_TRUE)/MU_TRUE:+.2f}%)  "
      f"k={k_c:.2f} ({100*(k_c-K_TRUE)/K_TRUE:+.2f}%)  omega={w_c:.4f}  MSE={mse_class:.3e}")

print("\ntraining under-weighted (lam_phys=1) and best (lam_phys=30)")
# Warm start (warm_start.py): lam_phys=1 used to be the first training in the
# process and ran on the GPU's cold path; lam_phys=30 always ran warm.
from warm_start import warm_start
warm_start(make_net, t_phys, t_data=t_data, u_data=u_data)
slip = train(1.0)
best = train(30.0)
for tag, r in (("lam_phys=1", slip), ("lam_phys=30", best)):
    print(f"  [{tag:>12}] mu={r['mu']:.4f} k={r['k']:.2f} omega={omega_of(r['mu'],r['k']):.4f} "
          f"MSE={r['mse']:.3e}  u(0)={r['u0']:.4f} u'(0)={r['v0']:.4f}")

# ======================================================================
#  1. THE PHASE LAG -- is it a frequency error?
# ======================================================================
print("\n--- phase lag at each zero crossing, lam_phys=1 ---")
w_slip = omega_of(slip["mu"], slip["k"])
zc_true = zero_crossings(t_fine, u_truth)
zc_net = zero_crossings(t_fine, slip["u"])
n = min(len(zc_true), len(zc_net))
zc_true, zc_net = zc_true[:n], zc_net[:n]

# measured: positive = network crosses LATER than truth
lag_meas = (zc_net - zc_true) * W_TRUE * 180 / np.pi
# a pure frequency error would put the n-th crossing at t*(w_true/w_net)
lag_freq = zc_true * (W_TRUE / w_slip - 1.0) * W_TRUE * 180 / np.pi

print(f"  omega: network {w_slip:.4f} vs true {W_TRUE:.4f}  (delta {w_slip-W_TRUE:+.4f} rad/s)")
print(f"  {'crossing t':>12} {'measured':>10} {'if freq err':>12}")
for tt, lm, lf in zip(zc_true, lag_meas, lag_freq):
    print(f"  {tt:>12.3f} {lm:>9.1f}d {lf:>11.1f}d")
sign_changes = int(np.sum(np.sign(lag_meas[:-1]) * np.sign(lag_meas[1:]) < 0))
print(f"  measured lag changes sign {sign_changes} time(s); a frequency error cannot change sign")

# ======================================================================
#  2. IS THE CURVE A SOLUTION AT ALL?
# ======================================================================
print("\n--- closest damped sinusoid to the NETWORK'S OWN curve ---")
d_f, w_f, coef_f = classical_fit(t_fine, slip["u"])
u_closest = basis(t_fine, d_f, w_f) @ coef_f
rms_closest = float(np.sqrt(np.mean((u_closest - slip["u"]) ** 2)))
rms_truth_to_net = float(np.sqrt(np.mean((u_truth - slip["u"]) ** 2)))
print(f"  best-fit family member: d={d_f:.3f}  omega={w_f:.3f}")
print(f"  it still misses the network by RMS {rms_closest:.3e}")
print(f"  the TRUE solution misses the network by RMS {rms_truth_to_net:.3e}")
print(f"  measurement noise sigma = {NOISE}")

# ======================================================================
#  3. BEST CASE: error relative to the LOCAL amplitude
# ======================================================================
print("\n--- best case: RMS error as a fraction of the local envelope ---")
env = np.exp(-D_TRUE * t_fine)
quarters = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
rel = {"classical": [], "pinn": []}
for a, b in quarters:
    m = (t_fine >= a) & (t_fine < b) if b < 1.0 else (t_fine >= a)
    den = np.sqrt(np.mean(env[m] ** 2))
    rel["classical"].append(100 * np.sqrt(np.mean((u_class - u_truth)[m] ** 2)) / den)
    rel["pinn"].append(100 * np.sqrt(np.mean((best["u"] - u_truth)[m] ** 2)) / den)
print(f"  {'window':>14} {'classical':>10} {'PINN':>8}")
for (a, b), c, p in zip(quarters, rel["classical"], rel["pinn"]):
    print(f"  [{a:.2f}, {b:.2f}] {c:>13.1f}% {p:>7.1f}%")
print(f"  classical tightens by {rel['classical'][0]/rel['classical'][-1]:.1f}x across the record; "
      f"PINN by {rel['pinn'][0]/rel['pinn'][-1]:.1f}x")

# ======================================================================
#  4. BEST CASE: does the network solve its own equation?
# ======================================================================
print("\n--- best case: drift from the exact solution of its OWN equation ---")
u_own = integrate_ivp(t_fine, best["mu"], best["k"], best["u0"], best["v0"])
drift = np.abs(best["u"] - u_own)
drift_q = []
for a, b in quarters:
    m = (t_fine >= a) & (t_fine < b) if b < 1.0 else (t_fine >= a)
    drift_q.append(100 * np.sqrt(np.mean(drift[m] ** 2)) / np.sqrt(np.mean(env[m] ** 2)))
print(f"  integrating mu={best['mu']:.4f}, k={best['k']:.2f} from the network's own "
      f"u(0)={best['u0']:.4f}, u'(0)={best['v0']:.4f}")
for (a, b), dq in zip(quarters, drift_q):
    print(f"  [{a:.2f}, {b:.2f}]  drift {dq:.2f}% of envelope")
print(f"  leftover residual RMS {best['resid_rms']:.2f} against terms of order "
      f"{best['term_rms']:.0f}  ({100*best['resid_rms']/best['term_rms']:.2f}%)")
print(f"  reconstruction implies u(0)={best['u0']:.3f}, u'(0)={best['v0']:.3f} "
      f"(truth: 1.000, 0.000)")

# ======================================================================
#  figures
# ======================================================================
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
a = ax[0]
a.plot(t_fine, u_truth, color="tab:gray", lw=3, alpha=0.6, label="Truth")
a.plot(t_fine, slip["u"], "--", color="tab:green", lw=2, label="PINN, $\\lambda_{phys}$=1")
a.plot(t_fine, u_closest, ":", color="tab:red", lw=2,
       label=f"closest damped sinusoid\n(d={d_f:.2f}, $\\omega$={w_f:.2f})")
a.scatter(t_data_np, u_data_np, s=18, color="black", zorder=5)
a.set_xlabel("t"); a.set_ylabel("u(t)"); a.grid(alpha=0.3); a.legend(fontsize=8)
a.set_title("The network is the outlier", fontsize=11)

a = ax[1]
a.axhline(0, color="tab:gray", lw=1)
a.plot(zc_true, lag_meas, "o-", color="tab:green", lw=2, label="measured lag")
a.plot(zc_true, lag_freq, "s--", color="tab:purple", lw=2,
       label=f"if it were a frequency error\n($\\omega$={w_slip:.2f} vs {W_TRUE:.2f})")
a.set_xlabel("t of zero crossing"); a.set_ylabel("lag (degrees, + = late)")
a.grid(alpha=0.3); a.legend(fontsize=8)
a.set_title("A frequency error cannot change sign", fontsize=11)

a = ax[2]
a.plot(t_fine, slip["u"] - u_closest, color="tab:red", lw=1.6,
       label=f"what no damped sinusoid can remove\n(RMS {rms_closest:.1e})")
a.axhline(NOISE, ls="--", color="tab:gray", lw=1.5, label=f"noise $\\sigma$ = {NOISE}")
a.axhline(-NOISE, ls="--", color="tab:gray", lw=1.5)
a.set_xlabel("t"); a.set_ylabel("residual"); a.grid(alpha=0.3); a.legend(fontsize=8)
a.set_title("Structured, and under the noise floor", fontsize=11)
fig.suptitle("Under-weighted physics ($\\lambda_{data}$=30, $\\lambda_{phys}$=1): "
             "the curve is not a solution of anything", fontsize=13)
fig.tight_layout()
fig.savefig("slip_diagnostics_result.png", dpi=120, bbox_inches="tight")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
a = ax[0]
a.plot(t_fine, u_truth, color="tab:gray", lw=3, alpha=0.6, label="Truth")
a.plot(t_fine, best["u"], "--", color="tab:green", lw=2, label="PINN")
a.plot(t_fine, u_class, "-", color="tab:purple", lw=1.4, label="Classical LSQ")
a.scatter(t_data_np, u_data_np, s=18, color="black", zorder=5)
a.set_xlim(0.72, 1.0); a.set_ylim(-0.30, 0.30)
a.set_xlabel("t"); a.set_ylabel("u(t)"); a.grid(alpha=0.3); a.legend(fontsize=8)
a.set_title("The last cycle, magnified", fontsize=11)

a = ax[1]
w = 0.35
xs = np.arange(len(quarters))
a.bar(xs - w/2, rel["classical"], w, color="tab:purple", label="Classical LSQ")
a.bar(xs + w/2, rel["pinn"], w, color="tab:green", label="PINN")
a.set_xticks(xs); a.set_xticklabels([f"{a0:.2f}-{b0:.2f}" for a0, b0 in quarters])
a.set_xlabel("window in t"); a.set_ylabel("RMS error / local amplitude (%)")
a.grid(alpha=0.3, axis="y"); a.legend(fontsize=8)
a.set_title("The classical fit tightens; the PINN does not", fontsize=11)

a = ax[2]
a.semilogy(t_fine, drift / env, color="tab:red", lw=1.8)
a.set_xlabel("t"); a.set_ylabel("|PINN - own solution| / envelope")
a.grid(alpha=0.3, which="both")
a.set_title("Drift from the exact solution of its own equation", fontsize=11)
fig.suptitle("Best setting ($\\lambda_{data}=\\lambda_{phys}$=30): what the headline numbers hide",
             fontsize=13)
fig.tight_layout()
fig.savefig("bestcase_diagnostics_result.png", dpi=120, bbox_inches="tight")

json.dump(dict(
    classical=dict(mu=mu_c, k=k_c, omega=w_c, mse=mse_class),
    slip=dict(mu=slip["mu"], k=slip["k"], omega=w_slip, mse=slip["mse"],
              u0=slip["u0"], v0=slip["v0"],
              crossings=zc_true.tolist(), lag_measured=lag_meas.tolist(),
              lag_if_frequency_error=lag_freq.tolist(), sign_changes=sign_changes,
              closest_d=d_f, closest_omega=w_f, rms_closest=rms_closest,
              rms_truth_to_net=rms_truth_to_net),
    best=dict(mu=best["mu"], k=best["k"], omega=omega_of(best["mu"], best["k"]),
              mse=best["mse"], u0=best["u0"], v0=best["v0"],
              resid_rms=best["resid_rms"], term_rms=best["term_rms"],
              rel_err_classical=rel["classical"], rel_err_pinn=rel["pinn"],
              drift_quarters=drift_q),
    env=dict(torch=torch.__version__, device=str(dev)),
), open("slip_diagnostics.json", "w"), indent=2)
print("\nSaved slip_diagnostics_result.png, bestcase_diagnostics_result.png, slip_diagnostics.json")
