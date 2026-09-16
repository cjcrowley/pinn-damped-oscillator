"""
Case 1: solve the equation with no data at all (Figure 1).

The network sees no solution values, only collocation points and the demand
that the residual vanish at each of them. The initial conditions go in by
construction, u = 1 + t^2 N(t), unless hard_constraint is turned off.

This is also the playground. Every knob is in CONFIG; set PRESET to one of
the named experiments to see a specific failure, then change one thing and
run it again. "relu_fail" shows a network that cannot represent the problem,
and "soft_collapse" shows it satisfying the equation with u = 0.

Writes results/case1_forward.json and results/case1_forward.png.
"""
CONFIG = {
    # the physics
    "d": 2.0,                  # damping; 0 never stops.          try 0, 1, 2, 5, 10
    "w0": 20.0,                # frequency; higher is harder.      try 5, 20, 40
    "t_max": 1.0,              # length of the time window.        try 0.5, 1, 2
    # the network
    "hidden": 64,              # units per layer.                  try 16, 32, 64, 128
    "layers": 4,               # hidden layers.                    try 2, 3, 4, 5
    "activation": "tanh",      # "tanh", "sin", or "relu" (which cannot work: u'' = 0)
    # training
    "epochs": 50000,           # Adam steps.                       try 5000, 50000
    "lr": 1e-3,
    "n_collocation": 300,      # where the physics is enforced.    try 20, 100, 300, 1000
    "lr_decay": True,          # cosine schedule down to 1e-5
    "lbfgs_steps": 2000,       # L-BFGS polish after Adam; 0 turns it off
    # the initial conditions
    "hard_constraint": True,   # False: enforce them with a penalty instead
    "ic_weight": 0.01,         # the penalty, when hard_constraint is False
    "live_plot": False,        # True: watch it learn in a window
    "seed": 0,
}

PRESET = None
PRESETS = {
    "easy":           {"w0": 5.0, "epochs": 15000},
    "hard":           {"w0": 40.0, "epochs": 80000, "hidden": 128},
    "undamped":       {"d": 0.0},
    "overdamped":     {"d": 30.0, "w0": 20.0},
    "tiny_net":       {"hidden": 8, "layers": 2},
    "too_few_points": {"n_collocation": 20},
    "relu_fail":      {"activation": "relu"},
    "soft_collapse":  {"hard_constraint": False, "ic_weight": 0.1},
}

import matplotlib
import numpy as np
import torch

cfg = dict(CONFIG, **(PRESETS[PRESET] if PRESET else {}))
if not cfg["live_plot"]:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import oscillator as osc

mu, k = 2 * cfg["d"], cfg["w0"] ** 2


def make_model():
    net = osc.mlp(cfg["hidden"], cfg["layers"], cfg["activation"])
    return osc.HardIC(net) if cfg["hard_constraint"] else net


torch.manual_seed(cfg["seed"])
model = make_model()
print(f"{cfg['layers']} hidden layers x {cfg['hidden']} {cfg['activation']} units, "
      f"{sum(p.numel() for p in model.parameters())} parameters, on {osc.DEVICE}")

t_phys = osc.grid(cfg["n_collocation"], requires_grad=True, t_max=cfg["t_max"])
t0 = torch.zeros(1, 1, device=osc.DEVICE, requires_grad=True)
t_test = osc.grid(400, t_max=cfg["t_max"])
t_np = t_test.cpu().numpy().flatten()
u_ref = osc.rk4(t_np, mu=mu, k=k, t_max=cfg["t_max"])


def loss_fn():
    loss = osc.physics_loss(model, t_phys, mu=mu, k=k)
    if not cfg["hard_constraint"]:
        u0 = model(t0)
        v0 = torch.autograd.grad(u0, t0, torch.ones_like(u0), create_graph=True)[0]
        loss = loss + cfg["ic_weight"] * ((u0 - 1.0) ** 2 + v0 ** 2).squeeze()
    return loss


live = None
if cfg["live_plot"]:
    plt.ion()
    fig_live, ax_live = plt.subplots(figsize=(9, 5))
    ax_live.plot(t_np, u_ref, color="tab:gray", lw=3, alpha=0.6, label="True solution")
    (live,) = ax_live.plot(t_np, np.zeros_like(t_np), "--", color="tab:blue", lw=2, label="PINN")
    ax_live.legend(); ax_live.grid(alpha=0.3)

history = []


def record(loss):
    history.append(loss.item())
    step = len(history) - 1
    if step <= cfg["epochs"] and step % max(1, cfg["epochs"] // 10) == 0:
        print(f"  step {step:6d} | loss {history[-1]:.4e}", flush=True)
        if live is not None:
            live.set_ydata(osc.predict(model, t_test))
            ax_live.set_title(f"step {step} | loss {history[-1]:.2e}")
            fig_live.canvas.draw(); fig_live.canvas.flush_events(); plt.pause(0.001)


osc.warm_start(make_model, t_phys)
params = list(model.parameters())
osc.adam(params, loss_fn, cfg["epochs"], lr=cfg["lr"], eta_min=1e-5 if cfg["lr_decay"] else None, record=record)
mse_adam = float(np.mean((osc.predict(model, t_test) - u_ref) ** 2))
print(f"MSE after Adam alone: {mse_adam:.3e}")
osc.lbfgs(params, loss_fn, cfg["lbfgs_steps"], record=record)

u_pred = osc.predict(model, t_test)
abs_err = np.abs(u_pred - u_ref)
mse = float(np.mean((u_pred - u_ref) ** 2))
print(f"final MSE {mse:.3e}, max abs error {abs_err.max():.3e}")

osc.save_json("case1_forward.json", dict(mse_after_adam=mse_adam, mse=mse, max_abs_err=float(abs_err.max()),
                                         final_loss=history[-1], cfg=cfg))

if cfg["live_plot"]:
    plt.ioff()
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].plot(t_np, u_ref, color="tab:gray", lw=3, alpha=0.6, label="True (RK4)")
ax[0].plot(t_np, u_pred, "--", color="tab:blue", lw=2, label="PINN")
ax[0].scatter(t_phys.detach().cpu().numpy(), np.zeros(cfg["n_collocation"]), s=6, color="tab:red", zorder=5,
              label="Collocation pts")
ax[0].set_title(f"Solution  (d={cfg['d']}, w0={cfg['w0']})")
ax[0].set_xlabel("t"); ax[0].set_ylabel("u(t)"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].semilogy(history, color="tab:purple")
ax[1].set_title("Training loss (log scale)")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("loss"); ax[1].grid(alpha=0.3)
ax[2].semilogy(t_np, abs_err + 1e-12, color="tab:green")
ax[2].set_title(f"Pointwise |error|   (MSE={mse:.2e})")
ax[2].set_xlabel("t"); ax[2].set_ylabel("|PINN - true|"); ax[2].grid(alpha=0.3)
fig.suptitle(f"PINN Playground  [{PRESET or 'custom'}]  net={cfg['layers']}x{cfg['hidden']} {cfg['activation']}"
             f", epochs={cfg['epochs']}, hard_constraint={cfg['hard_constraint']}", fontsize=12)
fig.tight_layout()
osc.save_figure(fig, "case1_forward.png")
if cfg["live_plot"]:
    plt.show()
