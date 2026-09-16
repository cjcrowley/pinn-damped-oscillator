"""
Case 2: sparse, noisy data plus known physics (Figure 2).

Fifteen noisy measurements with a gap in the middle, mu and k known. Three
competitors on identical data: a free-form network trained on the points alone,
the PINN with both terms live (lambda_data = lambda_phys = 20), and classical
least squares on the known closed form, which with mu known has only the two
amplitudes A and B left to find.

Then a first inverse problem on the same points: k held at truth, mu unknown.
The PINN learns mu as a trainable parameter; two classical fits recover it,
one with k known and one with k free.

Writes results/case2_assimilation.json and results/case2_assimilation.png.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import oscillator as osc

LAM_DATA = LAM_PHYS = 20.0
MU_GUESS = 1.0

t_data_np, u_data_np = osc.gapped_data()
t_data, u_data = osc.column(t_data_np), osc.column(u_data_np)
t_phys = osc.grid(200, requires_grad=True)
t_test = osc.grid(400)
t_test_np = t_test.cpu().numpy().flatten()
u_truth = osc.rk4(t_test_np)
gap_lo, gap_hi = osc.record_gap(t_data_np)


def train(use_physics=True, trainable_mu=False):
    """Returns the reconstruction, its MSE, and the history and final value of mu."""
    torch.manual_seed(1)
    net = osc.mlp()
    params, mu, mu_hist = list(net.parameters()), osc.MU, []
    if trainable_mu:
        mu = nn.Parameter(torch.tensor(MU_GUESS, device=osc.DEVICE))
        params.append(mu)

    def loss_fn():
        loss = LAM_DATA * osc.data_loss(net, t_data, u_data)
        if use_physics:
            loss = loss + LAM_PHYS * osc.physics_loss(net, t_phys, mu=mu)
        return loss

    osc.fit(params, loss_fn, 10000, 1500,
            record=(lambda loss: mu_hist.append(mu.item())) if trainable_mu else None)
    u = osc.predict(net, t_test)
    return u, float(np.mean((u - u_truth) ** 2)), mu_hist, (mu.item() if trainable_mu else None)


osc.warm_start(osc.mlp, t_phys, t_data, u_data)

# ---- the fair fight, mu and k known ----
u_pinn, mse_pinn, _, _ = train()
u_free_nn, mse_free_nn, _, _ = train(use_physics=False)
u_lsq = osc.decay_basis(t_test_np, osc.D, osc.OMEGA) @ osc.fit_amplitudes(t_data_np, u_data_np, osc.D, osc.OMEGA)
mse_lsq = float(np.mean((u_lsq - u_truth) ** 2))
print(f"PINN {mse_pinn:.3e} | free-form network {mse_free_nn:.3e} | known-form least squares {mse_lsq:.3e}")

# ---- the same points, mu unknown ----
u_inv, mse_inv, mu_hist, mu_pinn = train(trainable_mu=True)
d_known, w_known, coef_known = osc.fit_damping(t_data_np, u_data_np)
mu_known = 2 * d_known
u_known = osc.decay_basis(t_test_np, d_known, w_known) @ coef_known
mse_known = float(np.mean((u_known - u_truth) ** 2))
d_free, w_free, coef_free = osc.fit_free(t_data_np, u_data_np, w_range=(10.0, 30.0), n_w=220, n_refine=90, dw=0.3)
mu_free = 2 * d_free
u_free = osc.decay_basis(t_test_np, d_free, w_free) @ coef_free
mse_free = float(np.mean((u_free - u_truth) ** 2))
print(f"mu: PINN {mu_pinn:.4f} | least squares, k known {mu_known:.4f} | "
      f"least squares, k free {mu_free:.4f}   (true {osc.MU})")

osc.save_json("case2_assimilation.json", dict(
    demo1=dict(t=t_test_np.tolist(), u_truth=u_truth.tolist(), u_pinn=u_pinn.tolist(),
               u_freeform=u_free_nn.tolist(), u_classical=u_lsq.tolist(),
               t_data=t_data_np.tolist(), u_data=u_data_np.tolist(), gap=[gap_lo, gap_hi], noise=0.05,
               mse=dict(pinn=mse_pinn, freeform=mse_free_nn, classical=mse_lsq),
               weights=dict(data=LAM_DATA, phys=LAM_PHYS)),
    demo2=dict(mu_pinn=mu_pinn, mu_lsq_k_known=float(mu_known), mu_lsq_k_free=float(mu_free),
               mse_pinn=mse_inv, mse_lsq_k_known=mse_known, mse_lsq_k_free=mse_free, mu_hist=mu_hist)))

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(2, 2, figsize=(14, 9))

a = ax[0, 0]
a.axvspan(gap_lo, gap_hi, color="tab:orange", alpha=0.08)
a.text((gap_lo + gap_hi) / 2, -0.9, "no data\n(gap)", ha="center", va="bottom", fontsize=8, color="tab:orange")
a.plot(t_test_np, u_truth, color="tab:gray", lw=3, alpha=0.6, label="Truth (unknown)")
a.plot(t_test_np, u_free_nn, ":", color="tab:red", lw=2, label=f"Free-form NN, data only (MSE {mse_free_nn:.1e})")
a.plot(t_test_np, u_pinn, "--", color="tab:blue", lw=2, label=f"PINN: data + physics (MSE {mse_pinn:.1e})")
a.plot(t_test_np, u_lsq, "-", color="tab:green", lw=1.6, label=f"Known-form LSQ (MSE {mse_lsq:.1e})")
a.scatter(t_data_np, u_data_np, s=35, color="black", zorder=5, label="Noisy sensors")
a.set_ylim(-2.2, 3.4)
a.set_title("DEMO 1  Fair fight: NN vs PINN vs known-form least squares")
a.set_xlabel("t"); a.set_ylabel("u(t)"); a.legend(fontsize=8, loc="upper right"); a.grid(alpha=0.3)

a = ax[0, 1]
a.semilogy(t_test_np, np.abs(u_free_nn - u_truth) + 1e-12, ":", color="tab:red", lw=2, label="Free-form NN")
a.semilogy(t_test_np, np.abs(u_pinn - u_truth) + 1e-12, "--", color="tab:blue", lw=2, label="PINN (data+physics)")
a.semilogy(t_test_np, np.abs(u_lsq - u_truth) + 1e-12, "-", color="tab:green", lw=1.6, label="Known-form LSQ")
a.axvspan(gap_lo, gap_hi, color="tab:orange", alpha=0.08)
a.set_title("Error vs truth (log scale) -- lower is better")
a.set_xlabel("t"); a.set_ylabel("|prediction - truth|"); a.legend(fontsize=8); a.grid(alpha=0.3)

a = ax[1, 0]
a.plot(t_test_np, u_truth, color="tab:gray", lw=3, alpha=0.6, label="Truth")
a.plot(t_test_np, u_inv, "--", color="tab:green", lw=2, label=f"PINN, k known (MSE {mse_inv:.1e})")
a.plot(t_test_np, u_known, "-", color="tab:purple", lw=1.4, label=f"LSQ, k known (MSE {mse_known:.1e})")
a.plot(t_test_np, u_free, "-", color="tab:orange", lw=1.4, label=f"LSQ, k free (MSE {mse_free:.1e})")
a.scatter(t_data_np, u_data_np, s=35, color="black", zorder=5, label="Noisy sensors")
a.set_title("DEMO 2  Solution recovered while mu was unknown")
a.set_xlabel("t"); a.set_ylabel("u(t)"); a.legend(fontsize=8); a.grid(alpha=0.3)

a = ax[1, 1]
a.plot(mu_hist, color="tab:green", lw=2, label=f"PINN (k known) -> {mu_pinn:.3f}")
a.axhline(mu_known, color="tab:purple", ls="-.", lw=1.8, label=f"LSQ, k known = {mu_known:.3f}")
a.axhline(mu_free, color="tab:orange", ls="-.", lw=1.8, label=f"LSQ, k free = {mu_free:.3f}")
a.axhline(osc.MU, color="tab:gray", ls="--", lw=2, label=f"true mu = {osc.MU}")
a.set_title(f"Recovering mu (true {osc.MU}):  all three noise-limited")
a.set_xlabel("optimiser iteration"); a.set_ylabel("mu estimate"); a.legend(fontsize=8); a.grid(alpha=0.3)

fig.suptitle("PINNs with data:  fit + physics (top)   and   parameter discovery (bottom)", fontsize=13)
fig.tight_layout()
osc.save_figure(fig, "case2_assimilation.png")
