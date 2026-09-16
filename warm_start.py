"""
========================================================================
  warm_start.py  --  take the GPU off its cold path before measuring
========================================================================
On a CUDA device the first training in a process runs before cuBLAS
holds a handle, and it takes a different arithmetic route to the same
sums. Every training after it takes the warm route, and warm results are
bit-identical from run to run. determinism_check.py found the effect,
and it is not confined to badly weighted problems: Case 2 at
lambda_phys = 20 gives a solution MSE of 2.660e-4 cold and 2.707e-4 warm.

So every study script calls warm_start() once, before its first
measured training. It runs a few optimiser steps of a physics-informed
loss on a throwaway network of the caller's own architecture, on the
caller's own collocation grid, then restores every random-number state
it touched, so the measured runs cannot tell it happened.

Five steps is enough. With them, a single Case 2 run at lambda_phys = 20
reproduces the published warm run (which was sixth in its process) bit
for bit, in every recorded quantity.
========================================================================
"""
import torch


def warm_start(make_net, t_phys, mu=4.0, k=400.0, k_scale=100.0, steps=5, lr=5e-3,
               t_data=None, u_data=None):
    """Run `steps` discarded optimiser steps so cuBLAS is initialised.

    make_net : zero-argument callable returning a network on the target device
    t_phys   : the caller's collocation tensor (requires_grad=True)
    t_data, u_data : optional measurements, so the data-term kernels warm too

    Does nothing on a CPU, where there is no cold path.
    """
    if t_phys.device.type != "cuda":
        return
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all()

    net = make_net()
    opt = torch.optim.Adam(list(net.parameters()), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        u = net(t_phys)
        u_t = torch.autograd.grad(u, t_phys, torch.ones_like(u), create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t_phys, torch.ones_like(u_t), create_graph=True)[0]
        loss = torch.mean(((u_tt + mu * u_t + k * u) / k_scale) ** 2)
        if t_data is not None:
            loss = loss + torch.mean((net(t_data) - u_data) ** 2)
        loss.backward()
        opt.step()
    del net, opt, u, u_t, u_tt, loss

    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state_all(cuda_state)
    print(f"(warm start: {steps} discarded optimiser steps, RNG state restored)", flush=True)
