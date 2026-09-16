"""
Case 2 with the physics weight swept (Figure 3).

Fifteen noisy points with a gap, mu and k known, lambda_data = 20 throughout,
lambda_phys walked across five decades. The runs split into the settings a
person could choose with no answer key, and the one graded against the truth:

  a priori   as-written     lambda_phys = 0.0625: what the first version of the
                            study ran, without ever typing a physics weight
             naive equal    lambda_data = lambda_phys = 1
             equal lambdas  lambda_data = lambda_phys = 20
             balanced       lambda_phys set so the two weighted terms are equal
                            at the end of a trial run (the as-written run)
  oracle     the best point of the scan, which needs the truth to find

Writes results/case2_weights.json and results/case2_weights.png.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import oscillator as osc

LAM_DATA = 20.0
SCAN = [0.0625, 0.2, 0.6, 2.0, 6.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0,
        200.0, 600.0, 2000.0, 6000.0]

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
mse_lsq_gap = float(np.mean((u_lsq - u_truth)[in_gap] ** 2))
print(f"known-form least squares: MSE {mse_lsq:.3e}, in the gap {mse_lsq_gap:.3e}")


def train(lam_data, lam_phys, tag, use_physics=True):
    torch.manual_seed(1)
    net = osc.mlp()

    def loss_fn():
        loss = lam_data * osc.data_loss(net, t_data, u_data)
        if use_physics:
            loss = loss + lam_phys * osc.physics_loss(net, t_phys)
        return loss

    osc.fit(list(net.parameters()), loss_fn, 10000, 1500)
    u = osc.predict(net, t_test)
    resid = osc.residual(net, t_phys).detach().cpu().numpy().flatten()
    out = dict(tag=tag, lam_data=float(lam_data), lam_phys=float(lam_phys),
               mse=float(np.mean((u - u_truth) ** 2)),
               mse_gap=float(np.mean((u - u_truth)[in_gap] ** 2)),
               l_data=float(osc.data_loss(net, t_data, u_data)),
               l_phys=float(osc.physics_loss(net, t_phys)) if use_physics else 0.0,
               resid_rms=float(np.sqrt(np.mean(resid ** 2))),
               peak=float(np.max(np.abs(u))), u=u)
    print(f"  [{tag:>16}] lambda_phys={lam_phys:<9.4g} MSE={out['mse']:.3e}  "
          f"gap={out['mse_gap']:.3e}  ({out['mse'] / mse_lsq:.2f}x least squares)", flush=True)
    return out


osc.warm_start(osc.mlp, t_phys, t_data, u_data)

scan = [train(LAM_DATA, lp, f"scan {lp:g}") for lp in SCAN]
best = min(scan, key=lambda r: r["mse"])
as_written = dict(next(r for r in scan if r["lam_phys"] == 0.0625), tag="as-written")
equal = dict(next(r for r in scan if r["lam_phys"] == 20.0), tag="equal lambdas")
naive = train(1.0, 1.0, "naive equal")
balanced = train(LAM_DATA, LAM_DATA * as_written["l_data"] / max(as_written["l_phys"], 1e-30), "balanced")
freeform = train(LAM_DATA, 0.0, "free-form NN", use_physics=False)
apriori = [naive, as_written, equal, balanced]
oracle = dict(best, tag="ORACLE optimal")
print(f"best lambda_phys in the scan: {best['lam_phys']:g}")

osc.save_json("case2_weights.json", dict(
    mse_lsq=mse_lsq, mse_lsq_gap=mse_lsq_gap,
    t=t_test_np.tolist(), u_truth=u_truth.tolist(), u_lsq=u_lsq.tolist(),
    t_data=t_data_np.tolist(), u_data=u_data_np.tolist(),
    scan=[dict(r, u=r["u"].tolist()) for r in scan],
    apriori=[dict(r, u=r["u"].tolist()) for r in apriori],
    oracle=dict(oracle, u=oracle["u"].tolist()),
    freeform=dict(freeform, u=freeform["u"].tolist())))

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 4, hspace=0.38, wspace=0.28)

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

a = fig.add_subplot(gs[1, :2])
lp = [r["lam_phys"] for r in scan]
a.loglog(lp, [r["mse"] for r in scan], "o-", color="tab:blue", label=r"PINN, $\lambda_{data}$=20")
a.axhline(mse_lsq, ls="--", color="tab:green", lw=2, label=f"known-form LSQ (2 params) = {mse_lsq:.1e}")
a.plot([oracle["lam_phys"]], [oracle["mse"]], "*", ms=20, color="tab:red", zorder=6,
       label=f"oracle best, $\\lambda_{{phys}}$={oracle['lam_phys']:.0f} (needs the truth)")
for r, colour in ((as_written, "tab:purple"), (equal, "tab:orange"), (balanced, "tab:brown")):
    a.plot([r["lam_phys"]], [r["mse"]], "s", ms=9, color=colour, zorder=6,
           label=f"{r['tag']}, $\\lambda_{{phys}}$={r['lam_phys']:.4g}")
a.set_xlabel(r"$\lambda_{phys}$   (residual normalised by K_SCALE=100)")
a.set_ylabel("solution MSE vs truth")
a.set_title("What the weight is worth, and what you could have known", fontsize=11)
a.legend(fontsize=8, loc="upper left")
a.grid(alpha=0.3, which="both")

a = fig.add_subplot(gs[1, 2])
a.loglog(lp, [r["l_data"] for r in scan], "o-", color="tab:red", label="data misfit")
a.loglog(lp, [r["l_phys"] for r in scan], "s-", color="tab:purple", label="physics residual")
a.set_xlabel(r"$\lambda_{phys}$")
a.set_ylabel("unweighted loss term")
a.set_title("The trade", fontsize=11)
a.legend(fontsize=8)
a.grid(alpha=0.3, which="both")

a = fig.add_subplot(gs[1, 3])
a.semilogx(lp, [r["peak"] for r in scan], "o-", color="tab:cyan")
a.axhline(1.0, ls="--", color="tab:gray", label="true peak |u| = 1")
a.set_xlabel(r"$\lambda_{phys}$")
a.set_ylabel("peak |u| of the reconstruction")
a.set_title("Collapse toward $u\\equiv0$", fontsize=11)
a.legend(fontsize=8)
a.grid(alpha=0.3, which="both")

fig.suptitle("Case 2 (mu and k known, 15 noisy points with a gap): "
             "the physics weight was never swept", fontsize=13)
osc.save_figure(fig, "case2_weights.png")
