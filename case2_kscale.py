"""
Case 2 with the residual divided by a different constant (Figure 4).

The physics term is lambda_phys * mean((r / K)^2), so only lambda_phys / K^2
ever reaches the optimizer. Changing K should do nothing but relabel the weight
axis: the best lambda_phys should move by exactly (K / 100)^2, and equal weights
(lambda_phys = lambda_data = 20) should stop being the same setting.

This runs the weight study's problem with K = 100, 200 and 400, and 300 as a
control, over one lattice of effective weights lambda_phys * (100 / K)^2.
Dividing by 200 or 400 is dividing by 100 and then by a power of two, which
binary floating point does exactly, so those runs should be bit-identical to
K = 100 at the same effective weight. A factor of three is not exact in binary,
so K = 300 can agree only up to rounding, and next to the collapse the rounding
decides the outcome.

Writes results/case2_kscale.json and results/case2_kscale.png.
"""
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import oscillator as osc

LAM_DATA = 20.0
LAM_EFF = [1.25, 2.5, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 160.0, 320.0]
CONSTANTS = [100, 200, 400, 300]

t_data_np, u_data_np = osc.gapped_data()
t_data, u_data = osc.column(t_data_np), osc.column(u_data_np)
t_phys = osc.grid(200, requires_grad=True)
t_test = osc.grid(400)
t_test_np = t_test.cpu().numpy().flatten()
u_truth = osc.rk4(t_test_np)
gap_lo, gap_hi = osc.record_gap(t_data_np)
in_gap = (t_test_np >= gap_lo) & (t_test_np <= gap_hi)
u_lsq = osc.decay_basis(t_test_np, osc.D, osc.OMEGA) @ osc.fit_amplitudes(t_data_np, u_data_np, osc.D, osc.OMEGA)
mse_lsq = float(np.mean((u_lsq - u_truth) ** 2))


def train(k_scale, lam_phys):
    torch.manual_seed(1)
    net = osc.mlp()
    osc.fit(list(net.parameters()),
            lambda: LAM_DATA * osc.data_loss(net, t_data, u_data)
            + lam_phys * osc.physics_loss(net, t_phys, k_scale=k_scale),
            10000, 1500)
    u = osc.predict(net, t_test)
    resid = osc.residual(net, t_phys).detach().cpu().numpy().flatten()
    return dict(lam_data=LAM_DATA, lam_phys=float(lam_phys), lam_eff=lam_phys * (100.0 / k_scale) ** 2,
                mse=float(np.mean((u - u_truth) ** 2)),
                mse_gap=float(np.mean((u - u_truth)[in_gap] ** 2)),
                l_data=float(osc.data_loss(net, t_data, u_data)),
                l_phys=float(osc.physics_loss(net, t_phys, k_scale=k_scale)),
                resid_rms=float(np.sqrt(np.mean(resid ** 2))),
                peak=float(np.max(np.abs(u))),
                u_sha=hashlib.sha256(np.asarray(u, dtype=np.float32).tobytes()).hexdigest()[:16])


osc.warm_start(osc.mlp, t_phys, t_data, u_data)

runs = {}
for K in CONSTANTS:
    lams = [le * (K / 100.0) ** 2 for le in LAM_EFF]
    if LAM_DATA not in lams:
        lams.append(LAM_DATA)   # equal weights, last, so the lattice keeps its order
    runs[K] = []
    for lp in lams:
        r = train(K, lp)
        runs[K].append(r)
        print(f"  K={K:<4} lambda_phys={lp:<8g} MSE={r['mse']:.3e}  ({r['mse'] / mse_lsq:.2f}x classical)",
              flush=True)

summary, identity = {}, {}
base = {round(r["lam_eff"], 6): r for r in runs[100]}
for K in CONSTANTS:
    ordered = sorted(runs[K], key=lambda r: r["lam_phys"])
    best = min(ordered, key=lambda r: r["mse"])
    equal = next(r for r in ordered if r["lam_phys"] == LAM_DATA)
    summary[K] = dict(best_lam_phys=best["lam_phys"], best_ratio=best["mse"] / mse_lsq,
                      equal_weights_ratio=equal["mse"] / mse_lsq)
for K in CONSTANTS:
    s = summary[K]
    s["predicted_best"] = summary[100]["best_lam_phys"] * (K / 100.0) ** 2
    print(f"K={K:<4} best lambda_phys {s['best_lam_phys']:<7g} (predicted {s['predicted_best']:g}, "
          f"{s['best_ratio']:.2f}x classical)   equal weights {s['equal_weights_ratio']:.2f}x classical")
    if K != 100:
        pairs = [(r, base[round(r["lam_eff"], 6)]) for r in runs[K] if round(r["lam_eff"], 6) in base]
        same = sum(r["u_sha"] == b["u_sha"] for r, b in pairs)
        worst = max((abs(r["mse"] - b["mse"]) / b["mse"] for r, b in pairs), default=0.0)
        identity[K] = dict(bit_identical=same, compared=len(pairs), worst_relative_mse_difference=worst)
        print(f"       {same}/{len(pairs)} runs bit-identical to K=100 at the same effective weight")

osc.save_json("case2_kscale.json", dict(
    mse_lsq=mse_lsq, lam_data=LAM_DATA, lam_eff=LAM_EFF,
    summary={str(K): v for K, v in summary.items()},
    identity_vs_K100={str(K): v for K, v in identity.items()},
    runs={str(K): v for K, v in runs.items()}))

# ---------------------------------------------------------------- figure
colours = {100: "tab:blue", 200: "tab:orange", 400: "tab:red", 300: "tab:purple"}
markers = {100: "o", 200: "s", 400: "D", 300: "^"}
CAP, TOP = 11.0, 16.0
fig, a = plt.subplots(figsize=(10.5, 5.8))
for K in CONSTANTS:
    ordered = sorted(runs[K], key=lambda r: r["lam_phys"])
    a.loglog([r["lam_phys"] for r in ordered], [min(r["mse"] / mse_lsq, CAP) for r in ordered],
             "--" if K == 300 else "-", marker=markers[K], color=colours[K], ms=6, lw=1.6,
             label=f"K = {K:g}" + ("   (control)" if K == 300 else ""))
    equal = next(r for r in ordered if r["lam_phys"] == LAM_DATA)
    a.plot([LAM_DATA], [min(equal["mse"] / mse_lsq, CAP)], marker="*", ms=17, color=colours[K], mec="k", zorder=6)
    b = min(ordered, key=lambda r: r["mse"])
    a.annotate(f"best\n{b['lam_phys']:g}", (b["lam_phys"], b["mse"] / mse_lsq), textcoords="offset points",
               xytext=(0, -24), ha="center", fontsize=8.5, color=colours[K])
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
osc.save_figure(fig, "case2_kscale.png", dpi=140)
