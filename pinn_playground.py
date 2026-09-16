"""
========================================================================
  PINN PLAYGROUND  --  Damped Harmonic Oscillator
========================================================================
A single, tweak-friendly script for getting a *feel* for how a
Physics-Informed Neural Network behaves.

HOW TO USE
  1. Edit the CONFIG block below (every knob is commented).
  2. Run:   python pinn_playground.py
  3. Look at the saved plot  pinn_playground_result.png
  4. Change one thing, run again, see what happened. Repeat.

FAST START: set PRESET to one of the named experiments (see PRESETS
below) to instantly see a specific behaviour -- e.g. PRESET="relu_fail"
to watch a network that *cannot* solve this, or PRESET="soft_collapse"
to watch it cheat with the trivial u=0 solution.

THE PHYSICS
  We solve the ODE          u'' + mu*u' + k*u = 0
  with initial conditions   u(0) = 1,  u'(0) = 0.
  (mu = 2*d,  k = w0**2). This is a mass on a spring with friction.
========================================================================
"""

# ======================================================================
#  CONFIG  --  everything you can play with lives here
# ======================================================================
CONFIG = {
    # ---- The physics -------------------------------------------------
    "d":     2.0,   # damping.  0 = frictionless (never stops).  TRY: 0, 1, 2, 5, 10
    "w0":    20.0,  # stiffness/frequency. Higher = faster wiggles = HARDER.  TRY: 5, 20, 40
    "t_max": 1.0,   # length of the time window [0, t_max].  TRY: 0.5, 1.0, 2.0

    # ---- The neural network ------------------------------------------
    "hidden":     64,      # neurons per layer (network "width").  TRY: 16, 32, 64, 128
    "layers":     4,       # number of hidden layers ("depth").    TRY: 2, 3, 4, 5
    "activation": "tanh",  # "tanh" (best), "sin" (also great), "relu" (will FAIL -- see why!)

    # ---- Training ----------------------------------------------------
    "epochs":        50000,  # optimisation steps.  TRY: 5000 (fast/rough), 50000 (accurate)
    "lr":            1e-3,   # learning rate.  TRY: 1e-2 (jumpy), 1e-3, 1e-4 (slow)
    "n_collocation": 300,    # # of physics sample points.  TRY: 20 (too few!), 100, 300, 1000
    "lr_decay":      True,   # smoothly lower the learning rate toward the end (helps a lot)
    "lbfgs_steps":   2000,   # after Adam, polish with L-BFGS (a 2nd-order optimiser).
                             # THIS is what takes the error from ~1e-2 down to ~1e-5. 0 = off.

    # ---- The learning "trick" (this is the heart of a PINN) ----------
    "hard_constraint": True,  # True : u = 1 + t^2 * N(t)  -> initial conditions are EXACT.
                              # False: enforce them with a penalty instead. Set False +
                              #        ic_weight small to watch the net CHEAT (u -> 0).
    "ic_weight":        0.01, # penalty strength when hard_constraint=False. TRY: 1e-5, 1e-4, 1e-2
                              # (these are 1e-4 x the old values: the physics residual is now
                              #  divided by K_SCALE=100, so it shrank by 1e4 relative to this.)

    # ---- Runtime -----------------------------------------------------
    "device":    "auto",   # "auto", "cuda" (GPU), or "cpu"
    "live_plot": False,    # True = pop up a window and WATCH it learn in real time
    "seed":      0,        # change for a different random start
}

# One-word experiments. Set PRESET to a key below to override CONFIG.
# Set PRESET = None to just use CONFIG as written above.
PRESET = None

PRESETS = {
    "easy":          {"w0": 5.0,  "epochs": 15000},                 # low frequency, converges fast
    "hard":          {"w0": 40.0, "epochs": 80000, "hidden": 128},  # fast wiggles, needs a big net
    "undamped":      {"d": 0.0},                                    # frictionless, oscillates forever
    "overdamped":    {"d": 30.0, "w0": 20.0},                       # so much friction it never oscillates
    "tiny_net":      {"hidden": 8, "layers": 2},                    # too small to fit -> underfits
    "too_few_points":{"n_collocation": 20},                         # not enough physics samples
    "relu_fail":     {"activation": "relu"},                        # ReLU has zero 2nd-derivative -> can't work
    "soft_collapse": {"hard_constraint": False, "ic_weight": 0.1},  # watch it cheat with u=0
}

# ======================================================================
#  From here down is the machinery. Read it for the "tour", but you
#  only need to touch CONFIG / PRESET above to experiment.
# ======================================================================
import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt

# Merge a preset (if any) over the base CONFIG.
cfg = dict(CONFIG)
if PRESET is not None:
    if PRESET not in PRESETS:
        raise SystemExit(f"Unknown PRESET '{PRESET}'. Choose from: {list(PRESETS)}")
    cfg.update(PRESETS[PRESET])
    print(f"[preset] '{PRESET}' -> {PRESETS[PRESET]}")

# Pick backend: an interactive window for live plots, else headless file output.
if not cfg["live_plot"]:
    matplotlib.use("Agg")

# ---- device & reproducibility ----
if cfg["device"] == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(cfg["device"])
torch.manual_seed(cfg["seed"])
np.random.seed(cfg["seed"])
print(f"Using device: {device}")

# ---- derive the physical constants ----
mu = 2 * cfg["d"]
k = cfg["w0"] ** 2

# Fixed constant the ODE residual is divided by, so the physics term is O(1)
# and the cost weights mean the same thing in every script in this study.
# It is a CONSTANT, never the trainable k -- the inverse scripts do not know k.
K_SCALE = 100.0


# ----------------------------------------------------------------------
#  Ground truth: a classic numerical solver (RK4).
#  This is the "old way" -- we use it only to grade the PINN. It works
#  for ANY d / w0 (under-, over-, or un-damped), unlike a closed form.
# ----------------------------------------------------------------------
def reference_solution(t_grid):
    """4th-order Runge-Kutta integration of x'=v, v'=-mu*v-k*x."""
    dt = 1e-4
    n = int(cfg["t_max"] / dt) + 1
    x, v = 1.0, 0.0                       # initial conditions u(0)=1, u'(0)=0
    ts = np.linspace(0, cfg["t_max"], n)
    xs = np.empty(n)

    def deriv(x, v):
        return v, -mu * v - k * x

    for i in range(n):
        xs[i] = x
        k1x, k1v = deriv(x, v)
        k2x, k2v = deriv(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v)
        k3x, k3v = deriv(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v)
        k4x, k4v = deriv(x + dt * k3x, v + dt * k3v)
        x += dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        v += dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return np.interp(t_grid, ts, xs)


# ----------------------------------------------------------------------
#  The neural network:  a function  t -> u(t)
# ----------------------------------------------------------------------
class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


def make_activation(name):
    return {"tanh": nn.Tanh, "relu": nn.ReLU, "sin": Sin}[name]()


class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        act = cfg["activation"]
        h, L = cfg["hidden"], cfg["layers"]
        net = [nn.Linear(1, h), make_activation(act)]
        for _ in range(L - 1):
            net += [nn.Linear(h, h), make_activation(act)]
        net += [nn.Linear(h, 1)]
        self.net = nn.Sequential(*net)

    def forward(self, t):
        if cfg["hard_constraint"]:
            # u = 1 + t^2 * N(t): forces u(0)=1 and u'(0)=0 exactly, for any N.
            return 1.0 + t ** 2 * self.net(t)
        # Soft mode: raw network; initial conditions are only *encouraged* by a penalty.
        return self.net(t)


model = PINN().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Network: {cfg['layers']} hidden layers x {cfg['hidden']} units "
      f"({cfg['activation']}), {n_params} trainable parameters")

optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-5)
             if cfg["lr_decay"] else None)

# Collocation points: where we check that the physics holds.
t_phys = torch.linspace(0, cfg["t_max"], cfg["n_collocation"],
                        device=device).view(-1, 1).requires_grad_(True)
t0 = torch.zeros(1, 1, device=device, requires_grad=True)  # for the soft-constraint IC penalty


# ----------------------------------------------------------------------
#  Optional live plot setup
# ----------------------------------------------------------------------
t_test = torch.linspace(0, cfg["t_max"], 400, device=device).view(-1, 1)
t_np = t_test.detach().cpu().numpy().flatten()
u_ref = reference_solution(t_np)

live = None
if cfg["live_plot"]:
    try:
        plt.ion()
        figL, axL = plt.subplots(figsize=(9, 5))
        axL.plot(t_np, u_ref, color="tab:gray", lw=3, alpha=0.6, label="True solution")
        (lineL,) = axL.plot(t_np, np.zeros_like(t_np), "--", color="tab:blue", lw=2, label="PINN")
        axL.set_xlabel("t"); axL.set_ylabel("u(t)"); axL.legend(); axL.grid(alpha=0.3)
        live = (figL, axL, lineL)
    except Exception as e:
        print(f"[live_plot disabled: {e}]")
        live = None


# ----------------------------------------------------------------------
#  Training loop  --  this is where the PINN actually "solves" the ODE
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
#  Warm start (warm_start.py). On a GPU the first training in a process
#  takes a different arithmetic route from every one after it; this was
#  the only training in the process, so it always ran cold. Five throwaway
#  steps first put the measured run on the reproducible warm route.
# ----------------------------------------------------------------------
from warm_start import warm_start
warm_start(lambda: PINN().to(device), t_phys, mu=mu, k=k, k_scale=K_SCALE)

loss_history = []
print("\nTraining...")
for epoch in range(cfg["epochs"] + 1):
    optimizer.zero_grad()

    # 1) PHYSICS LOSS: the ODE residual should be zero everywhere.
    #    We get u' and u'' by differentiating the network with autograd.
    u = model(t_phys)
    u_t = torch.autograd.grad(u, t_phys, torch.ones_like(u), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t_phys, torch.ones_like(u_t), create_graph=True)[0]
    residual = (u_tt + mu * u_t + k * u) / K_SCALE
    loss = torch.mean(residual ** 2)

    # 2) INITIAL-CONDITION LOSS (only in soft mode; hard mode bakes it in).
    if not cfg["hard_constraint"]:
        u0 = model(t0)
        u0_t = torch.autograd.grad(u0, t0, torch.ones_like(u0), create_graph=True)[0]
        loss = loss + cfg["ic_weight"] * ((u0 - 1.0) ** 2 + u0_t ** 2).squeeze()

    loss.backward()
    optimizer.step()
    if scheduler:
        scheduler.step()

    loss_history.append(loss.item())
    if epoch % max(1, cfg["epochs"] // 10) == 0:
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  epoch {epoch:6d} | loss {loss.item():.4e} | lr {lr_now:.2e}")
        if live is not None:
            with torch.no_grad():
                lineL.set_ydata(model(t_test).cpu().numpy().flatten())
            live[1].set_title(f"epoch {epoch} | loss {loss.item():.2e}")
            live[0].canvas.draw(); live[0].canvas.flush_events(); plt.pause(0.001)

# ----------------------------------------------------------------------
#  Phase 2: L-BFGS polish (2nd-order optimiser).
#  Adam gets us into the right neighbourhood; L-BFGS uses curvature
#  information to drive the residual down by orders of magnitude.
# ----------------------------------------------------------------------
def physics_loss():
    u = model(t_phys)
    u_t = torch.autograd.grad(u, t_phys, torch.ones_like(u), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t_phys, torch.ones_like(u_t), create_graph=True)[0]
    l = torch.mean(((u_tt + mu * u_t + k * u) / K_SCALE) ** 2)
    if not cfg["hard_constraint"]:
        u0 = model(t0)
        u0_t = torch.autograd.grad(u0, t0, torch.ones_like(u0), create_graph=True)[0]
        l = l + cfg["ic_weight"] * ((u0 - 1.0) ** 2 + u0_t ** 2).squeeze()
    return l

# How good is the Adam stage ON ITS OWN? The whole point of the L-BFGS stage is
# the gap between this number and the final one, so report it rather than infer it.
with torch.no_grad():
    mse_adam = float(np.mean((model(t_test).cpu().numpy().flatten() - u_ref) ** 2))
print(f"\nMSE vs reference after Adam alone: {mse_adam:.3e}")

if cfg["lbfgs_steps"] > 0:
    print(f"Polishing with L-BFGS ({cfg['lbfgs_steps']} iterations)...")
    lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=cfg["lbfgs_steps"],
                              history_size=50, tolerance_grad=1e-12,
                              tolerance_change=1e-14, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        l = physics_loss()
        l.backward()
        loss_history.append(l.item())
        return l

    lbfgs.step(closure)
    print(f"  L-BFGS final loss {loss_history[-1]:.4e}")

# ----------------------------------------------------------------------
#  Evaluate & grade against the reference solution
# ----------------------------------------------------------------------
model.eval()
with torch.no_grad():
    u_pred = model(t_test).cpu().numpy().flatten()
abs_err = np.abs(u_pred - u_ref)
mse = float(np.mean((u_pred - u_ref) ** 2))
print(f"\nFinal MSE vs reference solution: {mse:.3e}   (max abs error {abs_err.max():.3e})")

# ----------------------------------------------------------------------
#  Three-panel diagnostic figure
# ----------------------------------------------------------------------
if cfg["live_plot"]:
    plt.ioff()
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
tag = PRESET if PRESET else "custom"

ax[0].plot(t_np, u_ref, color="tab:gray", lw=3, alpha=0.6, label="True (RK4)")
ax[0].plot(t_np, u_pred, "--", color="tab:blue", lw=2, label="PINN")
ax[0].scatter(t_phys.detach().cpu().numpy(), np.zeros(cfg["n_collocation"]),
              s=6, color="tab:red", zorder=5, label="Collocation pts")
ax[0].set_title(f"Solution  (d={cfg['d']}, w0={cfg['w0']})")
ax[0].set_xlabel("t"); ax[0].set_ylabel("u(t)"); ax[0].legend(); ax[0].grid(alpha=0.3)

ax[1].semilogy(loss_history, color="tab:purple")
ax[1].set_title("Training loss (log scale)")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("loss"); ax[1].grid(alpha=0.3)

ax[2].semilogy(t_np, abs_err + 1e-12, color="tab:green")
ax[2].set_title(f"Pointwise |error|   (MSE={mse:.2e})")
ax[2].set_xlabel("t"); ax[2].set_ylabel("|PINN - true|"); ax[2].grid(alpha=0.3)

fig.suptitle(f"PINN Playground  [{tag}]  net={cfg['layers']}x{cfg['hidden']} {cfg['activation']}"
             f", epochs={cfg['epochs']}, hard_constraint={cfg['hard_constraint']}", fontsize=12)
fig.tight_layout()
out = "pinn_playground_result.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"Saved plot to {out}")
import json
with open("pinn_playground.json", "w") as _f:
    json.dump(dict(mse_after_adam=mse_adam, mse=mse, max_abs_err=float(abs_err.max()),
                   final_loss=loss_history[-1], cfg=dict(cfg),
                   env=dict(torch=torch.__version__, device=str(device))), _f, indent=2)
print("Saved pinn_playground.json")
if cfg["live_plot"]:
    print("Close the plot window to exit.")
    plt.show()
