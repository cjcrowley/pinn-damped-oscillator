"""
========================================================================
  PINN with DATA  --  the way people usually mean it
========================================================================
Companion to pinn_playground.py. The playground solved the equation with
NO data (the "forward" problem). This script shows the two data-driven
modes that people usually have in mind:

  DEMO 1  "data + physics"  (data assimilation / physics-regularised fit)
     You have a KNOWN equation and some NOISY, SPARSE measurements.
     Goal: the best curve that BOTH fits the data AND obeys the equation.
     We compare it against a naive fit that uses the data ONLY -- and
     watch the physics term rescue it from overfitting the noise and
     from wandering across a gap in the data.

  DEMO 2  "inverse"  (parameter discovery)
     Same noisy data, but now we PRETEND WE DON'T KNOW the damping mu.
     We make mu a trainable unknown and let the PINN recover it from the
     data. This is the thing PINNs are genuinely prized for.

The one idea that unifies everything (including the playground):

     Total loss = data_weight    * (fit the data)       <-- L_data
                + physics_weight * (obey the equation)  <-- L_physics

  * playground / forward :  L_data switched OFF
  * demo 1 / data+physics:  BOTH on, mu known
  * demo 2 / inverse      :  BOTH on, mu is a trainable unknown

Run:  .\.venv\Scripts\python.exe pinn_data_inverse.py
Out:  pinn_data_inverse_result.png
========================================================================
"""

DEMO = "both"   # "data", "inverse", or "both"

CFG = {
    # ---- TRUE physics (the data is generated from these) -------------
    "d":     2.0,    # true damping is mu_true = 2*d = 4.0
    "w0":    20.0,   # stiffness (k = w0**2). We assume k is KNOWN.
    "t_max": 1.0,

    # ---- the measurements --------------------------------------------
    "n_data":  15,            # how many noisy sensor readings we get
    "noise":   0.05,          # stdev of Gaussian measurement noise
    "gap":     (0.45, 0.65),  # a stretch of time with NO sensor (tests gap-filling)

    # ---- network & training ------------------------------------------
    "hidden": 32, "layers": 3, "activation": "tanh",
    "adam_epochs": 10000, "lr": 5e-3, "lbfgs_steps": 1500,
    "n_collocation": 200,
    "data_weight": 20.0,      # how hard to pull the fit toward the data points
    "physics_weight": 20.0,   # lambda_phys, equal to data_weight. That is the obvious
                              # choice with nothing else to go on. case2_weight_study.py
                              # finds it is not the best one -- 50 is, just below the
                              # collapse -- but it lands within 25% of the classical fit.
                              # For a long time this term carried an implicit 1.0 while
                              # the residual was divided by
                              # k=400 instead of K_SCALE=100 -- which is 0.0625 in these
                              # units, three thousandths of the data weight. That was an
                              # accident of the normalisation, not a decision, and it cost
                              # a factor of seventeen.

    # ---- inverse-problem setup ---------------------------------------
    "mu_init_guess": 1.0,     # our deliberately-wrong starting guess for mu (true = 4.0)

    "device": "auto", "seed": 1,
}

# ======================================================================
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device("cuda" if (CFG["device"] == "auto" and torch.cuda.is_available())
                      else (CFG["device"] if CFG["device"] != "auto" else "cpu"))
torch.manual_seed(CFG["seed"])
rng = np.random.default_rng(CFG["seed"])
print(f"Using device: {device}")

mu_true = 2 * CFG["d"]
k = CFG["w0"] ** 2
tmax = CFG["t_max"]

# Fixed constant the ODE residual is divided by, so the physics term is O(1)
# and the cost weights mean the same thing in every script in this study.
# It is a CONSTANT, never the trainable k -- the inverse scripts do not know k.
K_SCALE = 100.0


# ----------------------------------------------------------------------
#  Ground truth (RK4) -- used to GENERATE the data and to GRADE results.
# ----------------------------------------------------------------------
def reference_solution(t_grid, mu, k):
    dt = 1e-4
    n = int(tmax / dt) + 1
    ts = np.linspace(0, tmax, n)
    xs = np.empty(n)
    x, v = 1.0, 0.0
    for i in range(n):
        xs[i] = x
        def deriv(x, v): return v, -mu * v - k * x
        k1x, k1v = deriv(x, v)
        k2x, k2v = deriv(x + 0.5*dt*k1x, v + 0.5*dt*k1v)
        k3x, k3v = deriv(x + 0.5*dt*k2x, v + 0.5*dt*k2v)
        k4x, k4v = deriv(x + dt*k3x, v + dt*k3v)
        x += dt/6*(k1x + 2*k2x + 2*k3x + k4x)
        v += dt/6*(k1v + 2*k2v + 2*k3v + k4v)
    return np.interp(t_grid, ts, xs)


# ----------------------------------------------------------------------
#  Make the noisy, sparse "sensor" data (with a gap in the middle).
# ----------------------------------------------------------------------
lo, hi = CFG["gap"]
cand = rng.uniform(0, tmax, 2000)
cand = cand[(cand < lo) | (cand > hi)]          # punch out the gap
t_data_np = np.sort(cand[:CFG["n_data"]])
u_clean = reference_solution(t_data_np, mu_true, k)
u_data_np = u_clean + CFG["noise"] * rng.standard_normal(len(t_data_np))

# The band drawn on the plots should show the hole in the RECORD, not the window
# the sampler was forbidden from. They are not the same interval: the sampler is
# barred from (0.45, 0.65), but the nearest points actually landed further out,
# so the real gap is wider than the exclusion zone.
_gi = int(np.argmax(np.diff(t_data_np)))
gap_lo, gap_hi = float(t_data_np[_gi]), float(t_data_np[_gi + 1])

t_data = torch.tensor(t_data_np, dtype=torch.float32, device=device).view(-1, 1)
u_data = torch.tensor(u_data_np, dtype=torch.float32, device=device).view(-1, 1)

# dense grids for physics + evaluation
t_phys = torch.linspace(0, tmax, CFG["n_collocation"], device=device).view(-1, 1).requires_grad_(True)
t_test = torch.linspace(0, tmax, 400, device=device).view(-1, 1)
t_test_np = t_test.cpu().numpy().flatten()
u_truth = reference_solution(t_test_np, mu_true, k)


# ----------------------------------------------------------------------
#  HONEST CLASSICAL BASELINE: fit the data to the KNOWN functional form
#  of the solution,   u(t) = e^{-d t} (A cos(w t) + B sin(w t)),  w=sqrt(k-d^2).
#  This is what a physicist who knows the equation would actually do.
#  It is *linear* in (A, B), so fitting them is ordinary least squares.
# ----------------------------------------------------------------------
def form_basis(t, d, w):
    e = np.exp(-d * t)
    return np.stack([e * np.cos(w * t), e * np.sin(w * t)], axis=1)  # columns: the two basis funcs

def fit_form_known_d(t, u, d, k):
    """mu (hence d) known -> only A, B are free -> linear least squares."""
    w = np.sqrt(max(k - d * d, 1e-12))
    coef, *_ = np.linalg.lstsq(form_basis(t, d, w), u, rcond=None)
    return coef, w

def fit_form_recover_d(t, u, k):
    """mu (hence d) unknown but k KNOWN -> profile over d; A,B linear.
       Separable nonlinear least squares -- no scipy needed. (3 params: d,A,B)"""
    ds = np.linspace(1e-3, np.sqrt(k) * 0.999, 4000)   # underdamped region d < sqrt(k)
    best = (np.inf, None, None, None)
    for d in ds:
        w = np.sqrt(k - d * d)
        Phi = form_basis(t, d, w)
        coef, *_ = np.linalg.lstsq(Phi, u, rcond=None)
        r = float(np.sum((Phi @ coef - u) ** 2))
        if r < best[0]:
            best = (r, d, w, coef)
    _, d, w, coef = best
    return coef, d, w


def _best_on_grid(t, u, ds, ws):
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

def fit_form_free_omega(t, u):
    """d, A, B, AND omega ALL free (4 params) -- assumes we DON'T even know k.
       Two-stage grid (coarse then refine) over (d, omega); A,B stay linear."""
    _, d, w, _ = _best_on_grid(t, u, np.linspace(0.05, 8.0, 160), np.linspace(10.0, 30.0, 220))
    _, d, w, coef = _best_on_grid(t, u, np.linspace(max(0.01, d - 0.25), d + 0.25, 90),
                                        np.linspace(w - 0.3, w + 0.3, 90))
    return coef, d, w


# ----------------------------------------------------------------------
#  Network:  t -> u(t).  No hard constraint here -- the DATA pins the
#  solution down (we don't even assume we know the initial conditions).
# ----------------------------------------------------------------------
class Sin(nn.Module):
    def forward(self, x): return torch.sin(x)

def make_net():
    act = {"tanh": nn.Tanh, "relu": nn.ReLU, "sin": Sin}[CFG["activation"]]
    h, L = CFG["hidden"], CFG["layers"]
    layers = [nn.Linear(1, h), act()]
    for _ in range(L - 1):
        layers += [nn.Linear(h, h), act()]
    layers += [nn.Linear(h, 1)]
    return nn.Sequential(*layers).to(device)


def train(model, use_physics, mu_value, trainable_mu=False, tag=""):
    """Generic trainer. Loss = data_weight*L_data + [L_physics].
       If trainable_mu, mu becomes an unknown recovered from the data."""
    mu_param = None
    params = list(model.parameters())
    if trainable_mu:
        mu_param = nn.Parameter(torch.tensor(float(mu_value), device=device))
        params = params + [mu_param]

    def compute_loss():
        # DATA TERM: match the noisy measurements.
        loss = CFG["data_weight"] * torch.mean((model(t_data) - u_data) ** 2)
        # PHYSICS TERM: obey the ODE at the collocation points.
        if use_physics:
            mu_now = mu_param if trainable_mu else mu_value
            u = model(t_phys)
            u_t = torch.autograd.grad(u, t_phys, torch.ones_like(u), create_graph=True)[0]
            u_tt = torch.autograd.grad(u_t, t_phys, torch.ones_like(u_t), create_graph=True)[0]
            residual = (u_tt + mu_now * u_t + k * u) / K_SCALE   # normalised -> O(1), stable
            loss = loss + CFG["physics_weight"] * torch.mean(residual ** 2)
        return loss

    opt = torch.optim.Adam(params, lr=CFG["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["adam_epochs"], eta_min=1e-4)
    hist, mu_hist = [], []
    for epoch in range(CFG["adam_epochs"] + 1):
        opt.zero_grad()
        loss = compute_loss()
        loss.backward()
        opt.step(); sched.step()
        hist.append(loss.item())
        if mu_param is not None:
            mu_hist.append(mu_param.item())

    if CFG["lbfgs_steps"] > 0:
        lbfgs = torch.optim.LBFGS(params, max_iter=CFG["lbfgs_steps"], history_size=50,
                                  tolerance_grad=1e-12, tolerance_change=1e-14,
                                  line_search_fn="strong_wolfe")
        def closure():
            lbfgs.zero_grad()
            loss = compute_loss()
            loss.backward()
            hist.append(loss.item())
            if mu_param is not None:
                mu_hist.append(mu_param.item())
            return loss
        lbfgs.step(closure)

    with torch.no_grad():
        u_pred = model(t_test).cpu().numpy().flatten()
    mse = float(np.mean((u_pred - u_truth) ** 2))
    print(f"  [{tag}] final loss {hist[-1]:.3e} | MSE-vs-truth {mse:.3e}"
          + (f" | recovered mu {mu_param.item():.4f} (true {mu_true})" if mu_param is not None else ""))
    return u_pred, mse, mu_hist, (mu_param.item() if mu_param is not None else None)


# ----------------------------------------------------------------------
#  Run the demos
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
#  Warm start (warm_start.py). DEMO 1's PINN used to be the first training
#  in the process and ran on the GPU's cold path; the data-only network and
#  DEMO 2 after it always ran warm. Five throwaway steps first put all three
#  on the same reproducible route.
# ----------------------------------------------------------------------
from warm_start import warm_start
warm_start(make_net, t_phys, mu=mu_true, k=k, k_scale=K_SCALE, t_data=t_data, u_data=u_data)

results = {}

if DEMO in ("data", "both"):
    print("\nDEMO 1: data+physics  vs  data-only NN  vs  known-form least squares")
    torch.manual_seed(CFG["seed"])
    u_phys, mse_phys, _, _ = train(make_net(), use_physics=True,  mu_value=mu_true, tag="data+physics")
    torch.manual_seed(CFG["seed"])
    u_naive, mse_naive, _, _ = train(make_net(), use_physics=False, mu_value=mu_true, tag="data-only NN")
    # honest classical baseline: fit the known solution form (mu known -> linear LSQ)
    coef, w = fit_form_known_d(t_data_np, u_data_np, mu_true / 2, k)
    u_form = form_basis(t_test_np, mu_true / 2, w) @ coef
    mse_form = float(np.mean((u_form - u_truth) ** 2))
    print(f"  [known-form LSQ] MSE-vs-truth {mse_form:.3e}   (2 params: A, B)")
    results["data"] = (u_phys, mse_phys, u_naive, mse_naive, u_form, mse_form)

if DEMO in ("inverse", "both"):
    print(f"\nDEMO 2: inverse -- recover mu (start guess {CFG['mu_init_guess']}, true {mu_true})")
    torch.manual_seed(CFG["seed"])
    u_inv, mse_inv, mu_hist, mu_rec = train(make_net(), use_physics=True,
                                            mu_value=CFG["mu_init_guess"], trainable_mu=True, tag="inverse PINN")
    # classical baseline A: k known, recover mu (3 params d,A,B) -- same info as the PINN
    coef2, d_form, w_form = fit_form_recover_d(t_data_np, u_data_np, k)
    mu_form = 2 * d_form
    u_form_inv = form_basis(t_test_np, d_form, w_form) @ coef2
    mse_form_inv = float(np.mean((u_form_inv - u_truth) ** 2))
    print(f"  [LSQ, k known ]  mu {mu_form:.4f}  (true {mu_true}) | MSE-vs-truth {mse_form_inv:.3e}   (3 params)")
    # classical baseline B: d,A,B,omega ALL free (4 params) -- k NOT assumed known
    coef3, d_free, w_free = fit_form_free_omega(t_data_np, u_data_np)
    mu_free, k_free = 2 * d_free, w_free ** 2 + d_free ** 2
    u_free = form_basis(t_test_np, d_free, w_free) @ coef3
    mse_free = float(np.mean((u_free - u_truth) ** 2))
    print(f"  [LSQ, k free  ]  mu {mu_free:.4f}  (true {mu_true}) | MSE-vs-truth {mse_free:.3e}   "
          f"(4 params; implied k={k_free:.1f}, true {k})")
    results["inverse"] = (u_inv, mse_inv, mu_hist, mu_rec, u_form_inv, mse_form_inv, mu_form,
                          u_free, mse_free, mu_free)


# ----------------------------------------------------------------------
#  Plot: 2 rows (data demo on top, inverse demo on bottom)
# ----------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(14, 9))

# --- Row 1: data+physics vs free-form NN vs known-form least squares ---
if "data" in results:
    u_phys, mse_phys, u_naive, mse_naive, u_form, mse_form = results["data"]
    a = ax[0, 0]
    a.axvspan(gap_lo, gap_hi, color="tab:orange", alpha=0.08)
    a.text((gap_lo+gap_hi)/2, -0.9, "no data\n(gap)", ha="center", va="bottom", fontsize=8, color="tab:orange")
    a.plot(t_test_np, u_truth, color="tab:gray", lw=3, alpha=0.6, label="Truth (unknown)")
    a.plot(t_test_np, u_naive, ":",  color="tab:red",   lw=2, label=f"Free-form NN, data only (MSE {mse_naive:.1e})")
    a.plot(t_test_np, u_phys,  "--", color="tab:blue",  lw=2, label=f"PINN: data + physics (MSE {mse_phys:.1e})")
    a.plot(t_test_np, u_form,  "-",  color="tab:green", lw=1.6, label=f"Known-form LSQ (MSE {mse_form:.1e})")
    a.scatter(t_data_np, u_data_np, s=35, color="black", zorder=5, label="Noisy sensors")
    a.set_ylim(-2.2, 3.4)
    a.set_title("DEMO 1  Fair fight: NN vs PINN vs known-form least squares")
    a.set_xlabel("t"); a.set_ylabel("u(t)"); a.legend(fontsize=8, loc="upper right"); a.grid(alpha=0.3)

    a = ax[0, 1]
    a.semilogy(t_test_np, np.abs(u_naive - u_truth) + 1e-12, ":",  color="tab:red",   lw=2, label="Free-form NN")
    a.semilogy(t_test_np, np.abs(u_phys  - u_truth) + 1e-12, "--", color="tab:blue",  lw=2, label="PINN (data+physics)")
    a.semilogy(t_test_np, np.abs(u_form  - u_truth) + 1e-12, "-",  color="tab:green", lw=1.6, label="Known-form LSQ")
    a.axvspan(gap_lo, gap_hi, color="tab:orange", alpha=0.08)
    a.set_title("Error vs truth (log scale) -- lower is better")
    a.set_xlabel("t"); a.set_ylabel("|prediction - truth|"); a.legend(fontsize=8); a.grid(alpha=0.3)
else:
    ax[0, 0].axis("off"); ax[0, 1].axis("off")

# --- Row 2: inverse (PINN vs two classical recoveries) ---
if "inverse" in results:
    (u_inv, mse_inv, mu_hist, mu_rec, u_form_inv, mse_form_inv, mu_form,
     u_free, mse_free, mu_free) = results["inverse"]
    a = ax[1, 0]
    a.plot(t_test_np, u_truth, color="tab:gray", lw=3, alpha=0.6, label="Truth")
    a.plot(t_test_np, u_inv, "--", color="tab:green", lw=2, label=f"PINN, k known (MSE {mse_inv:.1e})")
    a.plot(t_test_np, u_form_inv, "-", color="tab:purple", lw=1.4, label=f"LSQ, k known (MSE {mse_form_inv:.1e})")
    a.plot(t_test_np, u_free, "-", color="tab:orange", lw=1.4, label=f"LSQ, k free (MSE {mse_free:.1e})")
    a.scatter(t_data_np, u_data_np, s=35, color="black", zorder=5, label="Noisy sensors")
    a.set_title("DEMO 2  Solution recovered while mu was unknown")
    a.set_xlabel("t"); a.set_ylabel("u(t)"); a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1, 1]
    a.plot(mu_hist, color="tab:green", lw=2, label=f"PINN (k known) -> {mu_rec:.3f}")
    a.axhline(mu_form, color="tab:purple", ls="-.", lw=1.8, label=f"LSQ, k known = {mu_form:.3f}")
    a.axhline(mu_free, color="tab:orange", ls="-.", lw=1.8, label=f"LSQ, k free = {mu_free:.3f}")
    a.axhline(mu_true, color="tab:gray", ls="--", lw=2, label=f"true mu = {mu_true}")
    a.set_title(f"Recovering mu (true {mu_true}):  all three noise-limited")
    a.set_xlabel("optimiser iteration"); a.set_ylabel("mu estimate"); a.legend(fontsize=8); a.grid(alpha=0.3)
else:
    ax[1, 0].axis("off"); ax[1, 1].axis("off")

fig.suptitle("PINNs with data:  fit + physics (top)   and   parameter discovery (bottom)", fontsize=13)
fig.tight_layout()
# Dump the curves too, so claims about how far apart these reconstructions sit
# can be checked rather than eyeballed off the plot.
if "data" in results:
    import json
    u_phys, mse_phys, u_naive, mse_naive, u_form, mse_form = results["data"]
    json.dump(dict(t=t_test_np.tolist(), u_truth=u_truth.tolist(),
                   u_pinn=u_phys.tolist(), u_freeform=u_naive.tolist(),
                   u_classical=u_form.tolist(),
                   t_data=t_data_np.tolist(), u_data=u_data_np.tolist(),
                   gap=[gap_lo, gap_hi], noise=CFG["noise"],
                   mse=dict(pinn=mse_phys, freeform=mse_naive, classical=mse_form),
                   weights=dict(data=CFG["data_weight"], phys=CFG["physics_weight"]),
                   env=dict(torch=torch.__version__, device=str(device))),
              open("pinn_data_inverse.json", "w"), indent=2)
    print("Saved pinn_data_inverse.json")
if "inverse" in results:
    import json
    (u_inv, mse_inv, mu_hist, mu_rec, u_form_inv, mse_form_inv, mu_form,
     u_free, mse_free, mu_free) = results["inverse"]
    with open("pinn_data_inverse_demo2.json", "w") as _f:
        json.dump(dict(mu_pinn=mu_rec, mu_lsq_k_known=float(mu_form), mu_lsq_k_free=float(mu_free),
                       mse_pinn=mse_inv, mse_lsq_k_known=mse_form_inv, mse_lsq_k_free=mse_free,
                       adam_epochs=CFG["adam_epochs"], mu_hist=mu_hist,
                       env=dict(torch=torch.__version__, device=str(device))), _f, indent=2)
    print("Saved pinn_data_inverse_demo2.json")

out = "pinn_data_inverse_result.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"\nSaved plot to {out}")
