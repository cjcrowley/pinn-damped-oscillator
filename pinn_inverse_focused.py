"""
========================================================================
  ONE CAREFUL DEMO:  inverse problem with NOTHING known
========================================================================
Scenario: we are handed sparse, noisy measurements of a damped oscillator
and told only its *structure* -- a linear 2nd-order ODE

        u'' + mu*u' + k*u = 0        (mu, k UNKNOWN)

equivalently, the solution is a decaying sinusoid

        u(t) = e^{-d t} (A cos(w t) + B sin(w t)),   d = mu/2,  w = sqrt(k - d^2)

We recover the physics TWO ways, each knowing NO coefficients up front:

  (A) CLASSICAL least squares  -- fit d, A, B, w   (all 4 free)
                                  => mu = 2d,  k = w^2 + d^2
  (B) PINN                     -- a neural net u(t), with mu and k as
                                  trainable unknowns in the ODE residual.

Both are given exactly the same data and the same structural knowledge.
We compare: recovered mu, recovered k, and solution error vs the truth.

Run:  .\.venv\Scripts\python.exe pinn_inverse_focused.py
Out:  pinn_inverse_focused_result.png
========================================================================
"""

CFG = {
    # ---- the TRUE system (used only to generate data + to grade) -----
    "d":     2.0,    # true damping   -> mu_true = 2*d = 4.0
    "w0":    20.0,   # true stiffness -> k_true  = w0^2 = 400.0
    "t_max": 1.0,

    # ---- the measurements --------------------------------------------
    "n_data": 20,    # number of noisy sensor readings
    "noise":  0.05,  # stdev of Gaussian measurement noise
    "seed":   1,

    # ---- the PINN ----------------------------------------------------
    "hidden": 32, "layers": 3,
    "adam_epochs": 12000, "lr": 5e-3, "lbfgs_steps": 2000,
    "n_collocation": 200,
    "data_weight":    30.0,  # lambda_data: how hard to pull the net onto the data points
    "physics_weight": 30.0,  # lambda_phys: how hard to enforce the ODE. RAISE THIS to kill
                             # the phase slippage (forces the net toward a true constant-w solution).
                             # Effective physics/data ratio is physics_weight/data_weight,
                             # further scaled by 1/K_SCALE^2 from the residual normalisation.
    "mu_init": 1.0,          # PINN's (wrong) starting guess for mu   (true 4.0)
    "k_init":  100.0,        # PINN's (wrong) starting guess for k     (true 400.0)

    "device": "auto",
}

# ======================================================================
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

dev = torch.device("cuda" if (CFG["device"] == "auto" and torch.cuda.is_available())
                   else (CFG["device"] if CFG["device"] != "auto" else "cpu"))
torch.manual_seed(CFG["seed"])
rng = np.random.default_rng(CFG["seed"])
print(f"Using device: {dev}")

mu_true = 2 * CFG["d"]
k_true = CFG["w0"] ** 2
tmax = CFG["t_max"]
# fixed scales used ONLY to precondition the optimiser (order-of-magnitude
# guesses, not knowledge of the answer). k varies over ~hundreds, mu over ~ones.
K_SCALE = 100.0


# ---- ground truth via RK4 (data generator + grader) ------------------
def reference_solution(t_grid, mu, k):
    dt = 1e-4
    n = int(tmax / dt) + 1
    ts = np.linspace(0, tmax, n)
    xs = np.empty(n)
    x, v = 1.0, 0.0
    for i in range(n):
        xs[i] = x
        ax_, av_ = v, -mu * v - k * x
        x2, v2 = x + 0.5*dt*ax_, v + 0.5*dt*av_
        ax2, av2 = v2, -mu * v2 - k * x2
        x3, v3 = x + 0.5*dt*ax2, v + 0.5*dt*av2
        ax3, av3 = v3, -mu * v3 - k * x3
        x4, v4 = x + dt*ax3, v + dt*av3
        ax4, av4 = v4, -mu * v4 - k * x4
        x += dt/6*(ax_ + 2*ax2 + 2*ax3 + ax4)
        v += dt/6*(av_ + 2*av2 + 2*av3 + av4)
    return np.interp(t_grid, ts, xs)


# ---- make the noisy, sparse data -------------------------------------
t_data_np = np.sort(rng.uniform(0, tmax, CFG["n_data"]))
u_data_np = reference_solution(t_data_np, mu_true, k_true) + CFG["noise"] * rng.standard_normal(CFG["n_data"])

t_data = torch.tensor(t_data_np, dtype=torch.float32, device=dev).view(-1, 1)
u_data = torch.tensor(u_data_np, dtype=torch.float32, device=dev).view(-1, 1)
t_phys = torch.linspace(0, tmax, CFG["n_collocation"], device=dev).view(-1, 1).requires_grad_(True)
t_test = torch.linspace(0, tmax, 500, device=dev).view(-1, 1)
t_test_np = t_test.cpu().numpy().flatten()
u_truth = reference_solution(t_test_np, mu_true, k_true)


# ======================================================================
#  (A) CLASSICAL: fit d, A, B, w  (all free).  A,B are linear given d,w,
#      so we grid-search (d, w) and solve A,B by least squares (2-stage).
# ======================================================================
def _grid(t, u, ds, ws):
    best = (np.inf, None, None, None)
    for d in ds:
        e = np.exp(-d * t)
        for w in ws:
            Phi = np.stack([e * np.cos(w * t), e * np.sin(w * t)], axis=1)
            coef, *_ = np.linalg.lstsq(Phi, u, rcond=None)
            r = float(np.sum((Phi @ coef - u) ** 2))
            if r < best[0]:
                best = (r, d, w, coef)
    return best

def classical_fit(t, u):
    _, d, w, _ = _grid(t, u, np.linspace(0.05, 8, 160), np.linspace(8, 32, 260))     # coarse
    _, d, w, coef = _grid(t, u, np.linspace(max(0.01, d-0.25), d+0.25, 100),
                                np.linspace(w-0.4, w+0.4, 100))                       # refine
    return d, w, coef

d_c, w_c, coef_c = classical_fit(t_data_np, u_data_np)
mu_c, k_c = 2 * d_c, w_c**2 + d_c**2
u_c = (np.stack([np.exp(-d_c*t_test_np)*np.cos(w_c*t_test_np),
                 np.exp(-d_c*t_test_np)*np.sin(w_c*t_test_np)], axis=1) @ coef_c)
mse_c = float(np.mean((u_c - u_truth)**2))
def pct(val, true):
    return f"{100 * (val - true) / true:+.2f}%"

print(f"\n(A) CLASSICAL LSQ  : mu={mu_c:.4f} ({pct(mu_c, mu_true)}) | "
      f"k={k_c:.2f} ({pct(k_c, k_true)}) | MSE {mse_c:.3e}")


# ======================================================================
#  (B) PINN: network u(t), with mu and k as trainable unknowns.
# ======================================================================
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        h, L = CFG["hidden"], CFG["layers"]
        layers = [nn.Linear(1, h), nn.Tanh()]
        for _ in range(L - 1):
            layers += [nn.Linear(h, h), nn.Tanh()]
        layers += [nn.Linear(h, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, t):
        return self.net(t)

# Warm start (warm_start.py): this was the only training in the process, so it
# always ran on the GPU's cold path. Five throwaway steps first; the RNG state
# is restored, so the network below starts from exactly the same weights.
from warm_start import warm_start
warm_start(lambda: Net().to(dev), t_phys, mu=CFG["mu_init"], k=CFG["k_init"], k_scale=K_SCALE,
           t_data=t_data, u_data=u_data)

net = Net().to(dev)
mu_p = nn.Parameter(torch.tensor(CFG["mu_init"], device=dev))          # trainable mu
kp = nn.Parameter(torch.tensor(CFG["k_init"] / K_SCALE, device=dev))   # trainable k (preconditioned)
params = list(net.parameters()) + [mu_p, kp]

def k_val():                 # actual k = K_SCALE * kp
    return K_SCALE * kp

def total_loss():
    # DATA: match measurements
    loss = CFG["data_weight"] * torch.mean((net(t_data) - u_data) ** 2)
    # PHYSICS: residual of u'' + mu*u' + k*u, normalised by K_SCALE for conditioning
    u = net(t_phys)
    u_t = torch.autograd.grad(u, t_phys, torch.ones_like(u), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t_phys, torch.ones_like(u_t), create_graph=True)[0]
    r = (u_tt + mu_p * u_t + k_val() * u) / K_SCALE
    return loss + CFG["physics_weight"] * torch.mean(r ** 2)

opt = torch.optim.Adam(params, lr=CFG["lr"])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["adam_epochs"], eta_min=1e-4)
mu_hist, k_hist = [], []
print("\n(B) PINN training...")
for epoch in range(CFG["adam_epochs"] + 1):
    opt.zero_grad(); loss = total_loss(); loss.backward(); opt.step(); sched.step()
    mu_hist.append(mu_p.item()); k_hist.append(k_val().item())
    if epoch % (CFG["adam_epochs"] // 8) == 0:
        print(f"  epoch {epoch:6d} | loss {loss.item():.3e} | mu {mu_p.item():.3f} | k {k_val().item():.1f}")

lbfgs = torch.optim.LBFGS(params, max_iter=CFG["lbfgs_steps"], history_size=50,
                          tolerance_grad=1e-12, tolerance_change=1e-14, line_search_fn="strong_wolfe")
def closure():
    lbfgs.zero_grad(); loss = total_loss(); loss.backward()
    mu_hist.append(mu_p.item()); k_hist.append(k_val().item())
    return loss
lbfgs.step(closure)

with torch.no_grad():
    u_pinn = net(t_test).cpu().numpy().flatten()
mu_pinn, k_pinn = mu_p.item(), k_val().item()
mse_pinn = float(np.mean((u_pinn - u_truth)**2))
print(f"\n(B) PINN           : mu={mu_pinn:.4f} ({pct(mu_pinn, mu_true)}) | "
      f"k={k_pinn:.2f} ({pct(k_pinn, k_true)}) | MSE {mse_pinn:.3e}")


# ======================================================================
#  Compare
# ======================================================================
print("\n" + "="*74)
print(f"{'':16}{'mu (true 4.0)':>20}{'k (true 400)':>22}{'solution MSE':>16}")
print(f"{'CLASSICAL LSQ':16}{f'{mu_c:.3f} ({pct(mu_c, mu_true)})':>20}"
      f"{f'{k_c:.2f} ({pct(k_c, k_true)})':>22}{mse_c:>16.2e}")
print(f"{'PINN':16}{f'{mu_pinn:.3f} ({pct(mu_pinn, mu_true)})':>20}"
      f"{f'{k_pinn:.2f} ({pct(k_pinn, k_true)})':>22}{mse_pinn:>16.2e}")
print("="*74)

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
a.axhline(mu_true, color="tab:gray", ls="--", lw=2, label=f"true = {mu_true}")
a.set_title("Recovering  mu  (damping)"); a.set_xlabel("optimiser iteration"); a.set_ylabel("mu")
a.legend(fontsize=8); a.grid(alpha=0.3)

a = ax[1, 1]
a.plot(k_hist, color="tab:green", lw=2, label=f"PINN -> {k_pinn:.1f}")
a.axhline(k_c, color="tab:purple", ls="-.", lw=1.8, label=f"classical = {k_c:.1f}")
a.axhline(k_true, color="tab:gray", ls="--", lw=2, label=f"true = {k_true}")
a.set_title("Recovering  k  (stiffness)"); a.set_xlabel("optimiser iteration"); a.set_ylabel("k")
a.legend(fontsize=8); a.grid(alpha=0.3)

fig.suptitle("Inverse problem, nothing known:  classical least squares  vs  PINN", fontsize=13)
fig.tight_layout()
out = "pinn_inverse_focused_result.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"\nSaved plot to {out}")
import json
with open("pinn_inverse_focused.json", "w") as _f:
    json.dump(dict(classical=dict(mu=float(mu_c), k=float(k_c), mse=float(mse_c)),
                   pinn=dict(mu=mu_pinn, k=k_pinn, mse=mse_pinn,
                             mu_max=max(mu_hist), k_min=min(k_hist)),
                   adam_epochs=CFG["adam_epochs"], mu_hist=mu_hist, k_hist=k_hist,
                   env=dict(torch=torch.__version__, device=str(dev))), _f, indent=2)
print("Saved pinn_inverse_focused.json")
