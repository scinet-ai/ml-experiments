# Mechanism-agnostic progress measures for grokking

**Problem (SciNet `e6efc81d`).** Nanda et al. (2023) defined *task-specific* progress
measures (restricted / excluded loss read off the known Fourier circuit) that rise
before test accuracy jumps and predict grokking on modular addition — but they require
the circuit to be known in advance. **Question:** is there a *mechanism-agnostic*
progress measure — computed only from weights/activations, with no task-specific circuit
knowledge — that crosses a stated threshold **before** test accuracy crosses its
grokking marker, with **positive lead time**, across modular addition, modular
multiplication, **and** sparse parity?

This directory reproduces grokking on all three tasks (≥5 seeds each) and evaluates six
mechanism-agnostic candidate measures.

## Tasks & models
| task | data | model | grok marker |
|------|------|-------|-------------|
| `add`    | `(a+b) mod p`, p=97, 40% train | 1-layer transformer (d=128, 4 heads) | test acc > 0.5 |
| `mul`    | `(a*b) mod p`, p=97, 40% train | 1-layer transformer (d=128, 4 heads) | test acc > 0.5 |
| `parity` | parity of k=3 fixed bits of n=30 | 1-hidden-layer MLP (width 256), Omnigrok-style large-ish init | test acc > 0.75 |

All trained full-batch with AdamW (lr 1e-3, β=(0.9,0.98)) and weight decay (the driver of
grokking). For the **binary** parity task, test acc = 0.5 is *chance*, so the meaningful
generalization transition is to high accuracy; we use the midpoint 0.75 as the grok
marker (documented, `analyze.py:GROK_THR_BY_TASK`). Modular tasks (chance ≈ 1/97) use 0.5.

## Mechanism-agnostic measures (no circuit knowledge)
Logged every N steps from weights/activations only:
- `weight_l2` — total L2 norm of all trainable parameters.
- `w_eff_rank` — mean participation-ratio effective rank of the 2-D weight matrices
  (`PR = (Σσ²)² / Σσ⁴` over singular values).
- `act_eff_rank` — effective rank of the hidden-activation covariance on a fixed probe batch.
- `act_sparsity` — fraction of near-zero post-nonlinearity activations.
- `act_kurtosis` — excess kurtosis of post-nonlinearity activations.
- `gzip_bytes` — MDL / compression proxy: gzip size of per-tensor int8-quantized weights.

## Pre-registered threshold & lead time (mechanism-agnostic, no test-acc peeking)
For a measure `M(t)` with reference `m0 = M(0)` and converged value `m_final` (mean of the
last few logged values), define normalised progress toward the converged value
`frac(t) = (M(t) − m0)/(m_final − m0)` (goes 0→1 whether M rises or falls). The
**threshold-cross step** is the first `t` with `frac(t) ≥ 0.5` (measure halfway to its
converged value). Then

    lead_time = grok_step − cross_step         (POSITIVE ⇒ measure LEADS the grok)

We also report **Spearman** correlations: `sp_conc = ρ(M(t), test_acc(t))` and a predictive
`sp_future = ρ(M(t), test_acc(t+1500 steps))`. (A measure that falls as test rises has
negative ρ; |ρ| is the tracking strength.)

## Usage
```bash
uv venv && source .venv/bin/activate && uv pip install -r requirements.txt

# zero-download smoke reproduction (~1-3 min, trains one small add seed, prints lead time)
bash reproduce.sh              # or: python3 verify.py

# full sweep (3 tasks x 5 seeds) -- writes csv/<task>_s<seed>.csv
python3 grok.py --task add    --seed 0 --steps 18000 --log_every 200 --train_frac 0.4 --out_csv csv/add_s0.csv
python3 grok.py --task mul    --seed 0 --steps 18000 --log_every 200 --train_frac 0.4 --out_csv csv/mul_s0.csv
python3 grok.py --task parity --seed 0 --steps 20000 --log_every 200 --n 30 --k 3 --wd 0.5 --init_scale 1 --n_train 1000 --out_csv csv/parity_s0.csv
# (repeat for seeds 1..4)

# analysis: lead times + Spearman + verdict + curve plots
python3 analyze.py
```
Set `GROK_DEVICE=mps` (Apple Silicon) or `cuda` to accelerate; default `cpu` is portable.

## Files
- `grok.py` — data, models, training loop, per-step measure logging.
- `analyze.py` — grok step, threshold crossings, lead times, Spearman, summary tables, plots.
- `verify.py` / `reproduce.sh` — zero-download smoke repro (headline: a positive lead time).
- `requirements.txt` — pinned environment (`method.env_lock`).
- `results_per_run.csv`, `results_summary.csv`, `curves_*.png` — generated outputs.

## Results

*(populated by `analyze.py`; see RESULTS section below and `results_summary.csv`.)*

<!-- RESULTS_TABLE -->
