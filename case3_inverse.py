"""
Case 3: both coefficients unknown (Figure 5 and the Case 3 table).

Twenty noisy measurements and only the form of the equation, a linear
second-order system with mu and k unknown. The coefficients are recovered two
ways, each knowing nothing else up front:

  classical   fit u = e^{-dt} (A cos wt + B sin wt) with d, omega, A and B all
              free, so mu = 2d and k = omega^2 + d^2
  PINN        a network u(t), with mu and k trainable parameters in the
              residual, starting from deliberately wrong guesses (1 and 100)

Writes results/case3_inverse.json and results/case3_inverse.png.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import oscillator as osc

LAM_DATA = LAM_PHYS = 30.0

t_data_np, u_data_np = osc.scattered_data()
t_data, u_data = osc.column(t_data_np), osc.column(u_data_np)
t_phys = osc.grid(200, requires_grad=True)
t_test = osc.grid(500)
t_test_np = t_test.cpu().numpy().flatten()
u_truth = osc.rk4(t_test_np)


def pct(value, true):
    return f"{100 * (value - true) / true:+.2f}%"


# ---- classical least squares ----
d_c, w_c, coef_c = osc.fit_free(t_data_np, u_data_np, n_refine=100)
mu_c, k_c = 2 * d_c, w_c ** 2 + d_c ** 2
u_c = osc.decay_basis(t_test_np, d_c, w_c) @ coef_c
mse_c = float(np.mean((u_c - u_truth) ** 2))
print(f"classical: mu={mu_c:.4f} ({pct(mu_c, osc.MU)})  k={k_c:.2f} ({pct(k_c, osc.K)})  MSE {mse_c:.3e}")

# ---- PINN, mu and k trainable; k is learned as K_SCALE * kp to keep it O(1) ----
osc.warm_start(osc.mlp, t_phys, t_data, u_data)
torch.manual_seed(1)
net = osc.mlp()
mu_p = nn.Parameter(torch.tensor(1.0, device=osc.DEVICE))
kp = nn.Parameter(torch.tensor(1.0, device=osc.DEVICE))
mu_hist, k_hist = [], []


def record(loss):
    mu_hist.append(mu_p.item())
    k_hist.append((osc.K_SCALE * kp).item())


osc.fit(list(net.parameters()) + [mu_p, kp],
        lambda: LAM_DATA * osc.data_loss(net, t_data, u_data)
        + LAM_PHYS * osc.physics_loss(net, t_phys, mu=mu_p, k=osc.K_SCALE * kp),
        12000, 2000, record=record)
u_pinn = osc.predict(net, t_test)
mu_pinn, k_pinn = mu_p.item(), (osc.K_SCALE * kp).item()
mse_pinn = float(np.mean((u_pinn - u_truth) ** 2))
print(f"PINN:      mu={mu_pinn:.4f} ({pct(mu_pinn, osc.MU)})  k={k_pinn:.2f} ({pct(k_pinn, osc.K)})  "
      f"MSE {mse_pinn:.3e}")

osc.save_json("case3_inverse.json", dict(
    classical=dict(mu=float(mu_c), k=float(k_c), mse=mse_c),
    pinn=dict(mu=mu_pinn, k=k_pinn, mse=mse_pinn, mu_max=max(mu_hist), k_min=min(k_hist)),
    mu_hist=mu_hist, k_hist=k_hist))

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(2, 2, figsize=(14, 9))

a = ax[0, 0]
a.plot(t_test_np, u_truth, color="tab:gray", lw=3, alpha=0.6, label="Truth")
a.plot(t_test_np, u_c, "-", color="tab:purple", lw=1.6, label=f"Classical LSQ (MSE {mse_c:.1e})")
a.plot(t_test_np, u_pinn, "--", color="tab:green", lw=2, label=f"PINN (MSE {mse_pinn:.1e})")
a.scatter(t_data_np, u_data_np, s=35, color="black", zorder=5, label="Noisy data")
a.set_title("Reconstructed solution"); a.set_xlabel("t"); a.set_ylabel("u(t)")
a.legend(fontsize=8); a.grid(alpha=0.3)

a = ax[0, 1]
a.semilogy(t_test_np, np.abs(u_c - u_truth) + 1e-12, "-", color="tab:purple", lw=1.6, label="Classical LSQ")
a.semilogy(t_test_np, np.abs(u_pinn - u_truth) + 1e-12, "--", color="tab:green", lw=2, label="PINN")
a.set_title("Error vs truth (log)"); a.set_xlabel("t"); a.set_ylabel("|pred - truth|")
a.legend(fontsize=8); a.grid(alpha=0.3)

a = ax[1, 0]
a.plot(mu_hist, color="tab:green", lw=2, label=f"PINN -> {mu_pinn:.3f}")
a.axhline(mu_c, color="tab:purple", ls="-.", lw=1.8, label=f"classical = {mu_c:.3f}")
a.axhline(osc.MU, color="tab:gray", ls="--", lw=2, label=f"true = {osc.MU}")
a.set_title("Recovering  mu  (damping)"); a.set_xlabel("optimiser iteration"); a.set_ylabel("mu")
a.legend(fontsize=8); a.grid(alpha=0.3)

a = ax[1, 1]
a.plot(k_hist, color="tab:green", lw=2, label=f"PINN -> {k_pinn:.1f}")
a.axhline(k_c, color="tab:purple", ls="-.", lw=1.8, label=f"classical = {k_c:.1f}")
a.axhline(osc.K, color="tab:gray", ls="--", lw=2, label=f"true = {osc.K}")
a.set_title("Recovering  k  (stiffness)"); a.set_xlabel("optimiser iteration"); a.set_ylabel("k")
a.legend(fontsize=8); a.grid(alpha=0.3)

fig.suptitle("Inverse problem, nothing known:  classical least squares  vs  PINN", fontsize=13)
fig.tight_layout()
osc.save_figure(fig, "case3_inverse.png")
