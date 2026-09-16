"""
Is run-to-run scatter a bug, or the hardware?

Trains one Case 2 configuration several times in one process. Every repeat
redraws the data and the initial weights from the seed and fingerprints them,
so a bug that changed either would show up rather than be assumed away.

What it found: the fingerprints are identical everywhere. On a GPU the first
training in a process disagrees with every one after it, and repeats 2 to n are
bit-identical. Forcing deterministic kernels (--mode cuda-det) does not change
that; warming up first (--warmup) removes it. The first pass reaches cuBLAS
before the process has a CUDA context (PyTorch warns as much, once per process)
and takes a different route to the same sums, and where it lands depends on
what the process did first. At equal weights, lambda_phys = 20, this script's
cold repeat gives 2.492e-4, the same as the original cold Case 2 run; an earlier
version of it, which assembled the loss in a different order, gave 2.660e-4.
Every warm repeat gives 2.707e-4. That is why every study script calls
warm_start() before its first measured training.

Run:  python determinism_check.py --mode cuda --configs 20            first repeat cold
      python determinism_check.py --mode cuda --configs 20 --warmup   warm throughout
      python determinism_check.py --mode cuda-det                     deterministic kernels
      python determinism_check.py --mode cpu
"""
import argparse
import hashlib
import os

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["cpu", "cuda", "cuda-det"], required=True)
parser.add_argument("--reps", type=int, default=3)
parser.add_argument("--configs", default="0.0625,20",
                    help="comma-separated lambda_phys values, trained in this order")
parser.add_argument("--warmup", action="store_true", help="warm start before the first repeat")
args = parser.parse_args()

# Both have to be set before torch is imported.
if args.mode == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
if args.mode == "cuda-det":
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch

import oscillator as osc

if args.mode == "cuda-det":
    torch.use_deterministic_algorithms(True)
print(f"mode={args.mode}  device={osc.DEVICE}  torch={torch.__version__}  repeats={args.reps}")

t_phys = osc.grid(200, requires_grad=True)
t_test = osc.grid(400)
u_truth = osc.rk4(t_test.cpu().numpy().flatten())


def fingerprint(*tensors):
    h = hashlib.sha256()
    for x in tensors:
        h.update(np.ascontiguousarray(x.detach().cpu().numpy()).tobytes())
    return h.hexdigest()[:12]


def train(lam_phys):
    t_np, u_np = osc.gapped_data()
    t_data, u_data = osc.column(t_np), osc.column(u_np)
    torch.manual_seed(1)
    net = osc.mlp()
    init = fingerprint(*net.parameters())
    osc.fit(list(net.parameters()),
            lambda: 20.0 * osc.data_loss(net, t_data, u_data) + lam_phys * osc.physics_loss(net, t_phys),
            10000, 1500)
    u = osc.predict(net, t_test)
    return dict(data=fingerprint(t_data, u_data), init=init, mse=float(np.mean((u - u_truth) ** 2)))


if args.warmup:
    t_np, u_np = osc.gapped_data()
    osc.warm_start(osc.mlp, t_phys, osc.column(t_np), osc.column(u_np))
    print("(warm start first: no repeat is the first training in this process)")

for lam_phys in (float(x) for x in args.configs.split(",")):
    rows = [train(lam_phys) for _ in range(args.reps)]
    mses = [r["mse"] for r in rows]
    print(f"\nlambda_phys = {lam_phys:g}")
    print(f"  data fingerprint : {'identical' if len({r['data'] for r in rows}) == 1 else 'DIFFERS'}")
    print(f"  initial weights  : {'identical' if len({r['init'] for r in rows}) == 1 else 'DIFFERS'}")
    print(f"  MSE per repeat   : {', '.join(f'{m:.6e}' for m in mses)}")
    print(f"  bit-identical    : {'yes' if len(set(mses)) == 1 else 'no'}"
          f"   (spread {(max(mses) - min(mses)) / min(mses) * 100:.2f}%)")
