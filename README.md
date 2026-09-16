# PINN on a damped oscillator

The code, numbers and figures behind my write-up on physics-informed neural
networks: **<https://christophercrowley.com/project-pinn.html>**

A physics-informed neural network (PINN) is trained on a differential equation
instead of on examples of its solution. Pointed at a damped harmonic
oscillator, where every right answer is known, it recovers the coefficients
more accurately than a classical least-squares fit on the same data, while
drawing a curve that is not a solution of the equation it just recovered.
Nothing the method reports from the inside flags it. The write-up makes that
argument; this repository holds everything needed to check it.

![The network's curve against the truth and the closest damped sinusoid, the phase lag at each zero crossing, and the structured residual no choice of parameters removes](results/case3_slip.png)

## The problem

Every script solves, fits, or identifies

```
u'' + mu*u' + k*u = 0,    u(0) = 1,  u'(0) = 0
```

with `d = 2` and `omega_0 = 20`, so `mu = 2d = 4` and `k = omega_0^2 = 400`:
about three and a sixth oscillations over `t` in `[0, 1]`, decaying to an
eighth of the starting amplitude. The reference solution everywhere is
fourth-order Runge–Kutta at `dt = 1e-4`. It generates the synthetic
measurements, and it grades every result.

The forward solve, data assimilation, and parameter discovery are not three
algorithms. They are one cost function with terms switched on and off:

| Case | data weight | physics weight | mu, k |
|---|---|---|---|
| 1. Forward solve | 0 | > 0 | known, fixed |
| 2. Data assimilation | > 0 | > 0 | known, fixed |
| 3. Inverse / discovery | > 0 | > 0 | **unknown, trainable** |

## What is here

| File | In the write-up | What it runs |
|---|---|---|
| `oscillator.py` | | Everything the scripts share: the reference solution, the measurements, the network, the loss, the training schedule, the classical fits, and the warm start. |
| `case1_forward.py` | Figure 1 | The equation alone, no data. It is also the playground: every knob sits in a config block at the top, with named presets for the interesting failures. |
| `case2_assimilation.py` | Figure 2 | Fifteen noisy points with a gap: the PINN against a free-form network and known-form least squares. Then the same points with `mu` unknown. |
| `case2_weights.py` | Figure 3 | The physics weight swept, with the weights a person could choose without an answer key marked separately from the one graded against it. |
| `case2_kscale.py` | Figure 4 | The same sweep with the residual divided by 100, 200, 400, and 300 as a control: does the best weight belong to the problem or to the constant? |
| `case2_seeds.py` | the seeds table | The same weights across five seeds. |
| `case3_inverse.py` | Figure 5, the Case 3 table | Both coefficients unknown: the PINN against classical least squares on identical data and identical prior knowledge. |
| `case3_weights.py` | Figures 6, 7 and 8, and their tables | The Case 3 physics-weight sweep; then the phase lag, the closest-sinusoid fit, the local-amplitude errors, and the drift from its own equation, on two of its runs. |
| `determinism_check.py` | | Whether run-to-run scatter is a bug or the hardware. |
| `results/` | | Every figure and JSON file the scripts write. They are committed, so a number can be checked without running anything. |

The diagnostics in `case3_weights.py`, and all of `determinism_check.py`, exist
because of questions I could not answer. The phase-slip and best-case
diagnostics were first worked out by hand and never folded back into code,
which meant the two parts of the write-up carrying the most weight were the two
nobody could check, including me. Writing them as code confirmed most of what
was there and caught one thing that was backwards: the frequency-error row of
the lag table had the wrong sign, predicting an early arrival for a network
that runs slow. Re-running everything warm, on a finer grid of weights, caught
two more: the free-form network's training loss had been reported as `1e-11`
when it is `1e-3`, and the half-decade grid had hidden that equal weights is
not the best setting.

## Reproducibility

The published numbers come from PyTorch 2.5.1 with CUDA 12.1 and NumPy 2.4.6,
under Python 3.12, on a 4 GB NVIDIA T400 (a Turing card). Other versions, other
GPUs, and CPUs give slightly different numbers. Near a collapse, where rounding
can decide which way a run goes, the differences can be large.

Two things make one run comparable with another.

**Every residual is divided by the same constant, `K_SCALE = 100`.** The
constant gets squared into the physics weight (only `lambda_phys / K_SCALE^2`
ever reaches the optimizer), so a weight means nothing until you say what the
residual was divided by. The first version of this study divided by a
different constant in one script, and that turned out to be most of the story
of Case 2. `case2_kscale.py` shows why.

**Every script calls `warm_start()` before its first measured training.** On a
GPU, the first training in a process takes a different arithmetic route from
every training after it: the first pass happens before cuBLAS holds a handle.
That is not a rounding curiosity, and it is not even a fixed alternative: where
the first run lands depends on what the process did before it. At equal weights
in Case 2, one script's cold run came out two percent below the warm answer and
another's eight percent below; the warm answer never moves. `warm_start()` runs
a few optimizer steps on a throwaway network and then restores the
random-number state, so repeated runs are bit-identical. On a CPU it does
nothing.

## Running it

Python 3.12 with PyTorch, NumPy, and Matplotlib. Everything runs on a
laptop-class GPU, and falls back to the CPU on its own when there is no CUDA
device.

```
pip install numpy matplotlib
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Pin `torch==2.5.1+cu121` and `numpy==2.4.6` only if you want the published
digits. Install torch from the PyTorch index alone: give pip a fallback to PyPI
and it will quietly hand you the CPU wheel instead.

Each script takes no arguments, runs from any directory, and writes into
`results/`:

```
python case1_forward.py
python case2_assimilation.py
python case2_weights.py
python case2_kscale.py
python case2_seeds.py
python case3_inverse.py
python case3_weights.py
```

The two long ones are `case2_kscale.py` and `case2_seeds.py`, at 41 and 45
trainings; the scripts are independent, so they can run side by side on one
GPU. To check that everything runs before committing to that, set
`PINN_QUICK=1` (in PowerShell, `$env:PINN_QUICK=1`): every schedule is cut to a
twentieth, and the output goes to `results_quick/` instead, leaving the
published results alone.

The cold start, and the warm start that removes it:

```
python determinism_check.py --mode cuda --configs 20            # first repeat cold, the rest warm
python determinism_check.py --mode cuda --configs 20 --warmup   # warm throughout
python determinism_check.py --mode cpu
```

Then change one thing and run it again.
