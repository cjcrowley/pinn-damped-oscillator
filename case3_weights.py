"""
Case 3 with the physics weight swept, and what its runs hide (Figures 6, 7, 8).

The same twenty points as case3_inverse.py, lambda_data = 30, and
lambda_phys = 1, 30, 300 and 3000 on a shorter schedule (6000 Adam steps,
1000 L-BFGS). Figure 7 is the sweep. Two of its runs are then examined:

  lambda_phys = 1  (Figure 6)
    Where does the network cross zero, and how late? Is the lag consistent with
    a frequency error, which accumulates, keeps its sign and never recovers?
    Then fit the closed form to the network's own curve with all four
    parameters free: if the best member of the solution family still cannot
    describe it, the curve is not a solution of anything.

  lambda_phys = 30  (Figure 8)
    The error relative to the local signal amplitude, by quarter, since MSE on
    a decaying signal reports on the loud beginning. The drift from the exact
    solution of the network's own recovered equation, started from its own
    initial state. And the initial conditions the reconstruction implies.

A lag is positive when the network crosses zero later than the truth, so a
network running slow shows a positive, growing lag.

Writes results/case3_weights.json, results/case3_weights.png,
results/case3_slip.png and results/case3_best.png.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import oscillator as osc

LAM_DATA = 30.0
WEIGHTS = [1.0, 30.0, 300.0, 3000.0]
NOISE = 0.05

t_data_np, u_data_np = osc.scattered_data()
t_data, u_data = osc.column(t_data_np), osc.column(u_data_np)
t_phys = osc.grid(200, requires_grad=True)
t_test = osc.grid(500)                        # the sweep is graded on this grid
t_test_np = t_test.cpu().numpy().flatten()
u_truth = osc.rk4(t_test_np)
t_fine = np.linspace(0, osc.T_MAX, 4001)      # finer, for the zero crossings and quarters
u_truth_fine = osc.rk4(t_fine)
t_fine_dev = osc.column(t_fine)


def train(lam_phys):
    torch.manual_seed(1)
    net = osc.mlp()
    mu_p = nn.Parameter(torch.tensor(1.0, device=osc.DEVICE))
    kp = nn.Parameter(torch.tensor(1.0, device=osc.DEVICE))   # k = K_SCALE * kp
    osc.fit(list(net.parameters()) + [mu_p, kp],
            lambda: LAM_DATA * osc.data_loss(net, t_data, u_data)
            + lam_phys * osc.physics_loss(net, t_phys, mu=mu_p, k=osc.K_SCALE * kp),
            6000, 1000)
    mu, k = mu_p.item(), (osc.K_SCALE * kp).item()
    u, u_fine = osc.predict(net, t_test), osc.predict(net, t_fine_dev)
    t0 = torch.zeros(1, 1, device=osc.DEVICE, requires_grad=True)
    u0 = net(t0)
    v0 = torch.autograd.grad(u0, t0, torch.ones_like(u0))[0]
    resid = osc.residual(net, t_phys, mu_p, osc.K_SCALE * kp).detach().cpu().numpy().flatten()
    term = np.abs(k * net(t_phys).detach().cpu().numpy().flatten())
    return dict(lam_phys=lam_phys, mu=mu, k=k, u=u, u_fine=u_fine,
                u0=float(u0.item()), v0=float(v0.item()),
                mse=float(np.mean((u - u_truth) ** 2)),
                mse_fine=float(np.mean((u_fine - u_truth_fine) ** 2)),
                resid_rms=float(np.sqrt(np.mean(resid ** 2))),
                term_rms=float(np.sqrt(np.mean(term ** 2))))


def zero_crossings(t, y):
    """Times where y changes sign, linearly interpolated."""
    idx = np.where(np.sign(y)[:-1] * np.sign(y)[1:] < 0)[0]
    return np.array([t[i] + y[i] / (y[i] - y[i + 1]) * (t[i + 1] - t[i]) for i in idx])


def omega_of(mu, k):
    d = mu / 2
    return float(np.sqrt(max(k - d * d, 1e-12)))


osc.warm_start(osc.mlp, t_phys, t_data, u_data)
runs = []
for w in WEIGHTS:
    r = train(w)
    runs.append(r)
    print(f"lambda_phys={w:<6g} mu={r['mu']:.3f}  k={r['k']:6.1f}  MSE={r['mse']:.2e}  "
          f"residual RMS={r['resid_rms']:.2e}", flush=True)
slip, best = runs[0], runs[1]

# ---- classical fit on the data, d, omega, A and B all free ----
d_c, w_c, coef_c = osc.fit_free(t_data_np, u_data_np)
u_class = osc.decay_basis(t_fine, d_c, w_c) @ coef_c
mse_class = float(np.mean((u_class - u_truth_fine) ** 2))
mu_c, k_c = 2 * d_c, w_c ** 2 + d_c ** 2

# ---- lambda_phys = 1: the phase lag, and whether any solution matches the curve ----
w_slip = omega_of(slip["mu"], slip["k"])
zc_true, zc_net = zero_crossings(t_fine, u_truth_fine), zero_crossings(t_fine, slip["u_fine"])
n = min(len(zc_true), len(zc_net))
zc_true, zc_net = zc_true[:n], zc_net[:n]
lag_meas = (zc_net - zc_true) * osc.OMEGA * 180 / np.pi
lag_freq = zc_true * (osc.OMEGA / w_slip - 1.0) * osc.OMEGA * 180 / np.pi
sign_changes = int(np.sum(np.sign(lag_meas[:-1]) * np.sign(lag_meas[1:]) < 0))
d_f, w_f, coef_f = osc.fit_free(t_fine, slip["u_fine"])
u_closest = osc.decay_basis(t_fine, d_f, w_f) @ coef_f
rms_closest = float(np.sqrt(np.mean((u_closest - slip["u_fine"]) ** 2)))
rms_truth_to_net = float(np.sqrt(np.mean((u_truth_fine - slip["u_fine"]) ** 2)))
print(f"\nlambda_phys=1: omega {w_slip:.4f} against {osc.OMEGA:.4f}; lag changes sign {sign_changes} time(s)")
print("  lag at each crossing (measured / if a frequency error): "
      + ", ".join(f"{m:+.1f}/{f:+.1f}" for m, f in zip(lag_meas, lag_freq)))
print(f"  closest damped sinusoid d={d_f:.3f} omega={w_f:.3f} still misses by RMS {rms_closest:.2e}; "
      f"the truth misses by {rms_truth_to_net:.2e}")

# ---- lambda_phys = 30: error against the local amplitude, and drift from its own equation ----
env = np.exp(-osc.D * t_fine)
quarters = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
masks = [(t_fine >= a) & (t_fine < b) if b < 1.0 else (t_fine >= a) for a, b in quarters]
rel = {"classical": [], "pinn": []}
for m in masks:
    den = np.sqrt(np.mean(env[m] ** 2))
    rel["classical"].append(100 * np.sqrt(np.mean((u_class - u_truth_fine)[m] ** 2)) / den)
    rel["pinn"].append(100 * np.sqrt(np.mean((best["u_fine"] - u_truth_fine)[m] ** 2)) / den)
u_own = osc.rk4(t_fine, mu=best["mu"], k=best["k"], u0=best["u0"], v0=best["v0"])
drift = np.abs(best["u_fine"] - u_own)
drift_q = [100 * np.sqrt(np.mean(drift[m] ** 2)) / np.sqrt(np.mean(env[m] ** 2)) for m in masks]
print(f"\nlambda_phys=30: classical mu={mu_c:.4f} k={k_c:.2f} | PINN mu={best['mu']:.4f} k={best['k']:.2f}")
print("  RMS error / local amplitude, classical: " + ", ".join(f"{x:.1f}%" for x in rel["classical"])
      + " | PINN: " + ", ".join(f"{x:.1f}%" for x in rel["pinn"]))
print("  drift from its own equation: " + ", ".join(f"{x:.2f}%" for x in drift_q))
print(f"  implied u(0)={best['u0']:.3f}, u'(0)={best['v0']:.3f}; residual RMS {best['resid_rms']:.2f} "
      f"against terms of {best['term_rms']:.0f}")

osc.save_json("case3_weights.json", dict(
    sweep=dict(data_weight=LAM_DATA,
               runs=[dict(physics_weight=r["lam_phys"], mu=r["mu"], k=r["k"], mse=r["mse"],
                          resid_rms=r["resid_rms"], peak=float(np.max(np.abs(r["u"])))) for r in runs]),
    diagnostics=dict(
        classical=dict(mu=mu_c, k=k_c, omega=w_c, mse=mse_class),
        slip=dict(mu=slip["mu"], k=slip["k"], omega=w_slip, mse=slip["mse_fine"], u0=slip["u0"], v0=slip["v0"],
                  crossings=zc_true.tolist(), lag_measured=lag_meas.tolist(),
                  lag_if_frequency_error=lag_freq.tolist(), sign_changes=sign_changes,
                  closest_d=d_f, closest_omega=w_f, rms_closest=rms_closest, rms_truth_to_net=rms_truth_to_net),
        best=dict(mu=best["mu"], k=best["k"], omega=omega_of(best["mu"], best["k"]), mse=best["mse_fine"],
                  u0=best["u0"], v0=best["v0"], resid_rms=best["resid_rms"], term_rms=best["term_rms"],
                  rel_err_classical=rel["classical"], rel_err_pinn=rel["pinn"], drift_quarters=drift_q))))

# ---------------------------------------------------------------- Figure 7: the sweep
fig, ax = plt.subplots(2, len(WEIGHTS), figsize=(4.3 * len(WEIGHTS), 8))
for j, r in enumerate(runs):
    a = ax[0, j]
    a.plot(t_test_np, u_truth, color="tab:gray", lw=3, alpha=0.6, label="Truth")
    a.plot(t_test_np, r["u"], "--", color="tab:green", lw=2, label="PINN")
    a.scatter(t_data_np, u_data_np, s=18, color="black", zorder=5)
    a.set_title(f"physics_weight = {r['lam_phys']:.0f}\nk={r['k']:.1f} (true 400)  MSE={r['mse']:.1e}", fontsize=10)
    a.set_xlabel("t"); a.grid(alpha=0.3)
    if j == 0:
        a.set_ylabel("u(t)"); a.legend(fontsize=8)
axk, axmse, axres, axmu = ax[1, 0], ax[1, 1], ax[1, 2], ax[1, 3]
axk.semilogx(WEIGHTS, [r["k"] for r in runs], "o-", color="tab:green"); axk.axhline(osc.K, ls="--", color="tab:gray")
axk.set_title("recovered k -> 400"); axk.set_xlabel("physics_weight"); axk.set_ylabel("k"); axk.grid(alpha=0.3)
axmu.semilogx(WEIGHTS, [r["mu"] for r in runs], "o-", color="tab:blue"); axmu.axhline(osc.MU, ls="--", color="tab:gray")
axmu.set_title("recovered mu -> 4"); axmu.set_xlabel("physics_weight"); axmu.set_ylabel("mu"); axmu.grid(alpha=0.3)
axmse.loglog(WEIGHTS, [r["mse"] for r in runs], "o-", color="tab:red")
axmse.set_title("solution MSE vs truth"); axmse.set_xlabel("physics_weight"); axmse.set_ylabel("MSE"); axmse.grid(alpha=0.3)
axres.loglog(WEIGHTS, [r["resid_rms"] for r in runs], "o-", color="tab:purple")
axres.set_title("leftover ODE residual (RMS)"); axres.set_xlabel("physics_weight"); axres.set_ylabel("|residual|")
axres.grid(alpha=0.3)
fig.suptitle("Weighting the physics harder: does the slippage die? (data_weight fixed at 30)", fontsize=13)
fig.tight_layout()
osc.save_figure(fig, "case3_weights.png")

# ---------------------------------------------------------------- Figure 6: lambda_phys = 1
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
a = ax[0]
a.plot(t_fine, u_truth_fine, color="tab:gray", lw=3, alpha=0.6, label="Truth")
a.plot(t_fine, slip["u_fine"], "--", color="tab:green", lw=2, label="PINN, $\\lambda_{phys}$=1")
a.plot(t_fine, u_closest, ":", color="tab:red", lw=2,
       label=f"closest damped sinusoid\n(d={d_f:.2f}, $\\omega$={w_f:.2f})")
a.scatter(t_data_np, u_data_np, s=18, color="black", zorder=5)
a.set_xlabel("t"); a.set_ylabel("u(t)"); a.grid(alpha=0.3); a.legend(fontsize=8)
a.set_title("The network is the outlier", fontsize=11)
a = ax[1]
a.axhline(0, color="tab:gray", lw=1)
a.plot(zc_true, lag_meas, "o-", color="tab:green", lw=2, label="measured lag")
a.plot(zc_true, lag_freq, "s--", color="tab:purple", lw=2,
       label=f"if it were a frequency error\n($\\omega$={w_slip:.2f} vs {osc.OMEGA:.2f})")
a.set_xlabel("t of zero crossing"); a.set_ylabel("lag (degrees, + = late)")
a.grid(alpha=0.3); a.legend(fontsize=8)
a.set_title("A frequency error cannot change sign", fontsize=11)
a = ax[2]
a.plot(t_fine, slip["u_fine"] - u_closest, color="tab:red", lw=1.6,
       label=f"what no damped sinusoid can remove\n(RMS {rms_closest:.1e})")
a.axhline(NOISE, ls="--", color="tab:gray", lw=1.5, label=f"noise $\\sigma$ = {NOISE}")
a.axhline(-NOISE, ls="--", color="tab:gray", lw=1.5)
a.set_xlabel("t"); a.set_ylabel("residual"); a.grid(alpha=0.3); a.legend(fontsize=8)
a.set_title("Structured, and under the noise floor", fontsize=11)
fig.suptitle("Under-weighted physics ($\\lambda_{data}$=30, $\\lambda_{phys}$=1): "
             "the curve is not a solution of anything", fontsize=13)
fig.tight_layout()
osc.save_figure(fig, "case3_slip.png")

# ---------------------------------------------------------------- Figure 8: lambda_phys = 30
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
a = ax[0]
a.plot(t_fine, u_truth_fine, color="tab:gray", lw=3, alpha=0.6, label="Truth")
a.plot(t_fine, best["u_fine"], "--", color="tab:green", lw=2, label="PINN")
a.plot(t_fine, u_class, "-", color="tab:purple", lw=1.4, label="Classical LSQ")
a.scatter(t_data_np, u_data_np, s=18, color="black", zorder=5)
a.set_xlim(0.72, 1.0); a.set_ylim(-0.30, 0.30)
a.set_xlabel("t"); a.set_ylabel("u(t)"); a.grid(alpha=0.3); a.legend(fontsize=8)
a.set_title("The last cycle, magnified", fontsize=11)
a = ax[1]
bw, xs = 0.35, np.arange(len(quarters))
a.bar(xs - bw / 2, rel["classical"], bw, color="tab:purple", label="Classical LSQ")
a.bar(xs + bw / 2, rel["pinn"], bw, color="tab:green", label="PINN")
a.set_xticks(xs); a.set_xticklabels([f"{a0:.2f}-{b0:.2f}" for a0, b0 in quarters])
a.set_xlabel("window in t"); a.set_ylabel("RMS error / local amplitude (%)")
a.grid(alpha=0.3, axis="y"); a.legend(fontsize=8)
a.set_title("The classical fit tightens; the PINN does not", fontsize=11)
a = ax[2]
a.semilogy(t_fine, drift / env, color="tab:red", lw=1.8)
a.set_xlabel("t"); a.set_ylabel("|PINN - own solution| / envelope")
a.grid(alpha=0.3, which="both")
a.set_title("Drift from the exact solution of its own equation", fontsize=11)
fig.suptitle("Best setting ($\\lambda_{data}=\\lambda_{phys}$=30): what the headline numbers hide", fontsize=13)
fig.tight_layout()
osc.save_figure(fig, "case3_best.png")
