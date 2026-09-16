"""
Case 2 across five seeds (the seeds table).

The weight study ran one seed and found its best setting just below a cliff,
which is exactly where one seed is worth nothing. This repeats the interesting
weights on seeds 1 to 5. A seed redraws both the fifteen noisy points and the
network's starting weights, so each seed is an independent repeat of the whole
experiment. The classical fit is redone on each seed's data, and the number
that matters is the ratio of the PINN's error to it.

Writes results/case2_seeds.json.
"""
import numpy as np
import torch

import oscillator as osc

LAM_DATA = 20.0
LAM_PHYS = [0.0625, 1.25, 6.25, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0]
SEEDS = [1, 2, 3, 4, 5]

t_phys = osc.grid(200, requires_grad=True)
t_test = osc.grid(400)
t_test_np = t_test.cpu().numpy().flatten()
u_truth = osc.rk4(t_test_np)


def train(t_data, u_data, seed, lam_phys):
    torch.manual_seed(seed)
    net = osc.mlp()
    osc.fit(list(net.parameters()),
            lambda: LAM_DATA * osc.data_loss(net, t_data, u_data)
            + lam_phys * osc.physics_loss(net, t_phys),
            10000, 1500)
    u = osc.predict(net, t_test)
    return dict(mse=float(np.mean((u - u_truth) ** 2)), peak=float(np.max(np.abs(u))))


t_first, u_first = osc.gapped_data(SEEDS[0])
osc.warm_start(osc.mlp, t_phys, osc.column(t_first), osc.column(u_first))

lsq, runs = {}, {}
for seed in SEEDS:
    t_np, u_np = osc.gapped_data(seed)
    t_data, u_data = osc.column(t_np), osc.column(u_np)
    u_lsq = osc.decay_basis(t_test_np, osc.D, osc.OMEGA) @ osc.fit_amplitudes(t_np, u_np, osc.D, osc.OMEGA)
    lsq[seed] = float(np.mean((u_lsq - u_truth) ** 2))
    print(f"\nseed {seed}: known-form least squares MSE {lsq[seed]:.3e}", flush=True)
    for lp in LAM_PHYS:
        r = runs[(seed, lp)] = train(t_data, u_data, seed, lp)
        print(f"  lambda_phys={lp:>7g}  MSE={r['mse']:.3e}  {r['mse'] / lsq[seed]:>8.2f}x least squares"
              f"  peak |u|={r['peak']:.3f}", flush=True)

print("\nPINN error / least-squares error, lambda_data = 20")
print(f"{'lambda_phys':>12}" + "".join(f"{'seed ' + str(s):>9}" for s in SEEDS) + f"{'median':>9}")
for lp in LAM_PHYS:
    ratios = [runs[(s, lp)]["mse"] / lsq[s] for s in SEEDS]
    print(f"{lp:>12g}" + "".join(f"{x:>9.2f}" for x in ratios) + f"{float(np.median(ratios)):>9.2f}")
best_per_seed = {s: min(LAM_PHYS, key=lambda lp: runs[(s, lp)]["mse"]) for s in SEEDS}
print("best lambda_phys per seed: " + ", ".join(f"{s}: {v:g}" for s, v in best_per_seed.items()))

osc.save_json("case2_seeds.json", dict(
    lam_data=LAM_DATA, seeds=SEEDS, lam_phys=LAM_PHYS,
    lsq={str(s): v for s, v in lsq.items()},
    runs={f"{s}_{lp:g}": v for (s, lp), v in runs.items()},
    best_per_seed={str(s): v for s, v in best_per_seed.items()}))
