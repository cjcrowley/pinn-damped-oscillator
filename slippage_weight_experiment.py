"""
Does weighting the physics term harder kill the phase slippage?
Sweep lambda_phys (physics_weight) and watch the PINN reconstruction,
the recovered (mu, k), the solution error, and the leftover ODE residual.

Self-contained; oscillating system (w0=20) so the slippage is visible.
Run:  python slippage_weight_experiment.py
Out:  slippage_weight_result.png
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- fixed setup (same data for every weight) ----
D_TRUE, W0_TRUE, TMAX = 2.0, 20.0, 1.0
MU_TRUE, K_TRUE = 2 * D_TRUE, W0_TRUE ** 2
N_DATA, NOISE, SEED = 20, 0.05, 1
DATA_WEIGHT, K_SCALE = 30.0, 100.0
ADAM_EPOCHS, LR, LBFGS_STEPS, N_COLLOC = 6000, 5e-3, 1000, 200

WEIGHTS = [1.0, 30.0, 300.0, 3000.0]   # the physics weights to compare

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(SEED)
print(f"Using device: {dev}")


def reference_solution(t_grid, mu, k):
    dt = 1e-4
    n = int(TMAX / dt) + 1
    ts = np.linspace(0, TMAX, n); xs = np.empty(n)
    x, v = 1.0, 0.0
    for i in range(n):
        xs[i] = x
        a1v = -mu*v - k*x
        x2, v2 = x+0.5*dt*v, v+0.5*dt*a1v;   a2v = -mu*v2 - k*x2
        x3, v3 = x+0.5*dt*v2, v+0.5*dt*a2v;  a3v = -mu*v3 - k*x3
        x4, v4 = x+dt*v3, v+dt*a3v;          a4v = -mu*v4 - k*x4
        x += dt/6*(v + 2*v2 + 2*v3 + v4)
        v += dt/6*(a1v + 2*a2v + 2*a3v + a4v)
    return np.interp(t_grid, ts, xs)


# ---- one fixed noisy dataset, shared across all runs ----
t_data_np = np.sort(rng.uniform(0, TMAX, N_DATA))
u_data_np = reference_solution(t_data_np, MU_TRUE, K_TRUE) + NOISE * rng.standard_normal(N_DATA)
t_data = torch.tensor(t_data_np, dtype=torch.float32, device=dev).view(-1, 1)
u_data = torch.tensor(u_data_np, dtype=torch.float32, device=dev).view(-1, 1)
t_phys = torch.linspace(0, TMAX, N_COLLOC, device=dev).view(-1, 1).requires_grad_(True)
t_test = torch.linspace(0, TMAX, 500, device=dev).view(-1, 1)
t_test_np = t_test.cpu().numpy().flatten()
u_truth = reference_solution(t_test_np, MU_TRUE, K_TRUE)


def make_net():
    layers = [nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(),
              nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1)]
    return nn.Sequential(*layers).to(dev)


def train(physics_weight):
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
        opt.zero_grad(); loss_fn().backward(); opt.step(); sch.step()
    lb = torch.optim.LBFGS(params, max_iter=LBFGS_STEPS, history_size=50,
                           tolerance_grad=1e-12, tolerance_change=1e-14, line_search_fn="strong_wolfe")
    def closure():
        lb.zero_grad(); l = loss_fn(); l.backward(); return l
    lb.step(closure)

    with torch.no_grad():
        u_pinn = net(t_test).cpu().numpy().flatten()
    # leftover ODE residual magnitude (how badly the net still breaks the ODE)
    u = net(t_phys)
    u_t = torch.autograd.grad(u, t_phys, torch.ones_like(u), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t_phys, torch.ones_like(u_t), create_graph=True)[0]
    resid = (u_tt + mu_p * u_t + K_SCALE * kp * u).detach().cpu().numpy().flatten()
    mse = float(np.mean((u_pinn - u_truth) ** 2))
    return dict(mu=mu_p.item(), k=(K_SCALE * kp).item(), mse=mse,
                u=u_pinn, resid_rms=float(np.sqrt(np.mean(resid ** 2))))


# Warm start (warm_start.py): physics_weight = 1 used to be the first training
# in the process and ran on the GPU's cold path; the other three always ran warm.
from warm_start import warm_start
warm_start(make_net, t_phys, t_data=t_data, u_data=u_data)

runs = []
for pw in WEIGHTS:
    r = train(pw)
    runs.append(r)
    print(f"physics_weight={pw:8.0f} | mu={r['mu']:.3f} | k={r['k']:6.1f} | "
          f"MSE={r['mse']:.2e} | residual_rms={r['resid_rms']:.2e}")


# ---- plot ----
fig, ax = plt.subplots(2, len(WEIGHTS), figsize=(4.3 * len(WEIGHTS), 8))
for j, (pw, r) in enumerate(zip(WEIGHTS, runs)):
    a = ax[0, j]
    a.plot(t_test_np, u_truth, color="tab:gray", lw=3, alpha=0.6, label="Truth")
    a.plot(t_test_np, r["u"], "--", color="tab:green", lw=2, label="PINN")
    a.scatter(t_data_np, u_data_np, s=18, color="black", zorder=5)
    a.set_title(f"physics_weight = {pw:.0f}\nk={r['k']:.1f} (true 400)  MSE={r['mse']:.1e}", fontsize=10)
    a.set_xlabel("t"); a.grid(alpha=0.3)
    if j == 0:
        a.set_ylabel("u(t)"); a.legend(fontsize=8)

# summary metrics vs physics_weight
ks = [r["k"] for r in runs]; mses = [r["mse"] for r in runs]; res = [r["resid_rms"] for r in runs]
mus = [r["mu"] for r in runs]
axk, axmse, axres, axmu = ax[1, 0], ax[1, 1], ax[1, 2], ax[1, 3]
axk.semilogx(WEIGHTS, ks, "o-", color="tab:green"); axk.axhline(K_TRUE, ls="--", color="tab:gray")
axk.set_title("recovered k -> 400"); axk.set_xlabel("physics_weight"); axk.set_ylabel("k"); axk.grid(alpha=0.3)
axmu.semilogx(WEIGHTS, mus, "o-", color="tab:blue"); axmu.axhline(MU_TRUE, ls="--", color="tab:gray")
axmu.set_title("recovered mu -> 4"); axmu.set_xlabel("physics_weight"); axmu.set_ylabel("mu"); axmu.grid(alpha=0.3)
axmse.loglog(WEIGHTS, mses, "o-", color="tab:red")
axmse.set_title("solution MSE vs truth"); axmse.set_xlabel("physics_weight"); axmse.set_ylabel("MSE"); axmse.grid(alpha=0.3)
axres.loglog(WEIGHTS, res, "o-", color="tab:purple")
axres.set_title("leftover ODE residual (RMS)"); axres.set_xlabel("physics_weight"); axres.set_ylabel("|residual|"); axres.grid(alpha=0.3)

fig.suptitle("Weighting the physics harder: does the slippage die? (data_weight fixed at 30)", fontsize=13)
fig.tight_layout()
fig.savefig("slippage_weight_result.png", dpi=120, bbox_inches="tight")
print("\nSaved plot to slippage_weight_result.png")
import json
with open("slippage_weight_experiment.json", "w") as _f:
    json.dump(dict(data_weight=DATA_WEIGHT,
                   runs=[dict(physics_weight=pw, mu=r["mu"], k=r["k"], mse=r["mse"],
                              resid_rms=r["resid_rms"], peak=float(np.max(np.abs(r["u"]))))
                         for pw, r in zip(WEIGHTS, runs)],
                   env=dict(torch=torch.__version__, device=str(dev))), _f, indent=2)
print("Saved slippage_weight_experiment.json")
