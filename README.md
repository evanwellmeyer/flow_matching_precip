# flow_matching_precip

Conditional flow matching model for global precipitation change prediction. Given a historical
climatology and a short-period trend field, the model generates a probabilistic dPdP field
(normalized future precipitation change) by learning a velocity field that transports the
source distribution to the target.

---

## Overview

**Task:** Predict `dPdP = (future_precip - hist_clim) / global_mean(dP)` conditioned on:
- `clim`: historical climatology (mm/yr)
- `x0`: short-period linear trend field (dimensionless)

**Method:** Conditional flow matching. The model learns a velocity field `v(xt, clim, t)` where
`xt = (1-t)*x0 + t*x1` interpolates between source and target. At inference, integrating from
`t=0` to `t=1` (or `t=DT`) maps a trend field to a dPdP prediction.

**Network:** `Unet6R` — a 6-level U-Net with custom geo-aware padding (reflect in latitude,
circular in longitude), sinusoidal time embedding, residual blocks with GroupNorm + Mish
activation, and optional dropout.

---

## Files

### Core

| File | Purpose |
|------|---------|
| `flow_models.py` | `Unet6R` architecture: 6-level U-Net with sinusoidal time conditioning |
| `flow_datasets.py` | `FlowDataset`: two-level hierarchical sampler with augmentation |
| `cmip6_split.py` | Train/val/test splits for CMIP6 by family, exported as `TRAIN_GROUPS`, `VAL_GROUPS`, `TEST_GROUPS` |

### Training

| File | Purpose |
|------|---------|
| `train_flow_base.py` | Pre-train on HadGEM3-PPE + CESM2-PPE + AMIP |
| `train_flow_finetune.py` | Finetune base model on CMIP6 by-model data |

### Data Processing

| File | Purpose |
|------|---------|
| `process_amip.py` | AMIP-PD + amip-future4K → clim/trend/dPdP .nc files (11 models) |
| `process_cmip6.py` | CMIP6 historical + SSP585 → per-model clim/trend/dPdP .nc files |
| `process_hadgem3-ppe.py` | HadGEM3-PPE → per-member clim/trend/dPdP .nc files (1500 members) |
| `process_cesm2_lens2.py` | CESM2-LENS2 zarr → clim/trend/dPdP .nc files (cmip6 + smbb forcings) |
| `make_pr.py` | Utility: combine PRECC + PRECL zarr stores |

### Diagnostics

| File | Purpose |
|------|---------|
| `check_flow_base.ipynb` | Evaluate base model on AMIP test set |
| `check_flow_finetune.ipynb` | Evaluate finetuned model on CMIP6 test set; compare vs baselines |
| `infer_gpcp.ipynb` | Apply model to GPCP-observed trend to produce real-world dPdP predictions |
| `make_poster_figures.ipynb` | Publication-quality figure generation |
| `analyze_cmip6_split.py` | Inspect and summarize the CMIP6 train/val/test split |
| `check_files.py` | Verify all expected processed .nc files are present |

---

## Pipeline

```
RAW DATA
├── AMIP (historical + amip-future4K, 11 models)
├── HadGEM3-PPE (1500 members, no historical trend)
├── CESM2-PPE (262 members, no historical trend)
└── CESM2-LENS2 (zarr, cmip6 + smbb forcings)

    ↓  process_*.py

PROCESSED .nc FILES  (128×192 Gaussian grid, lon −180→180)
├── AMIP_{model}_clim/trend/dPdP.nc
├── HadGEM3-PPE_{member}_clim/trend/dPdP.nc
├── CESM2-LENS2_{forcing}_clim/trend/dPdP.nc
└── CMIP6/{model}_clim/trend/dPdP.nc

    ↓  train_flow_base.py

BASE MODEL  (5 ensemble members)
└── weights/flow_base_.../best_member{i}.pth

    ↓  train_flow_finetune.py

FINETUNED MODEL  (5 ensemble members)
└── weights/flow_finetune_.../best_member{i}.pth

    ↓  check_flow_finetune.ipynb / infer_gpcp.ipynb

DIAGNOSTICS & PREDICTIONS
```

---

## Model Architecture

**Unet6R** (`flow_models.py`)

- Input: `[xt, clim]` concatenated → `(B, 2, H, W)` + scalar `t`
- Output: velocity field `v` → `(B, 1, H, W)`
- 6 encoder levels: channels scale as `[1, 2, 4, 8, 16, 32, 64] × base_channels`
- Bottleneck: sinusoidal time embedding injected additively
- Custom padding: reflect (latitude) + circular (longitude) for globe topology
- Residual blocks: Conv → GroupNorm → Mish → optional Dropout
- Typical size: `base_channels=16` (standard GPU), `base_channels=4` (dev/MPS)

**Flow Matching Formulation**

```
x0      historical trend (source)
x1      dPdP target
xt      (1-t)*x0 + t*x1     interpolated field
v*      x1 - x0              training target velocity
```

---

## Training

### Base Model (`train_flow_base.py`)

Pre-trains on three data sources using `FlowDataset`:

| Source | Members | x0 Strategy |
|--------|---------|-------------|
| HadGEM3-PPE | 1500 | Sample from HadAM3 AMIP trend pool |
| CESM2-PPE | 262 | Sample from CCM0B AMIP trend pool |
| AMIP (11 models) | 1–3 | Own trend / augmentation / noise / zeros |

**Sampling:** Two-level hierarchical — pick source uniformly, then pick member uniformly.
Prevents large PPE ensembles from dominating gradient updates.

**x0 Augmentation** (mutually exclusive, AMIP only):

| Mode | Probability | Description |
|------|-------------|-------------|
| Own trend | 0.25 | Use model's own historical trend |
| Cross-model aug | 0.25 | Swap with random trend from same GCM family pool |
| Gaussian noise | 0.25 | Replace x0 with noise at random scale 0.5–2.0 |
| Zeros | 0.25 | Replace x0 with zeros (climatology-only mode) |

**Loss:**
```
loss = masked_MSE(v_pred, v_target, land_mask)
     + ENDPOINT_PENALTY_WEIGHT × endpoint_velocity_penalty(v_pred, land_mask, t)
```

- Land mask: OCEAN_WEIGHT=0.5, LAND_WEIGHT=1.0
- Endpoint penalty: penalizes residual velocity for t > 0.8, ramping linearly to t=1.
  Encourages the velocity field to decay at the target endpoint so integration
  does not blow up.

**Time sampling:** Beta(0.5, 0.5) instead of Uniform — concentrates samples near t=0
and t=1 to improve boundary behavior.

**Key hyperparameters:**
```
BASE_CHANNELS = 16
LR = 1e-2  (cosine annealing → LR*0.01)
BATCH_SIZE = 100, N_EPOCHS = 1000, PATIENCE = 50
ENDPOINT_PENALTY_WEIGHT = 0.05, ENDPOINT_PENALTY_START = 0.8
T_BETA_ALPHA = T_BETA_BETA = 0.5
```

### Finetuned Model (`train_flow_finetune.py`)

Transfers the base model to CMIP6 by-model data using the `cmip6_split.py` holdout strategy.

**CMIP6 Data Split** (by GCM family — one largest per family held out for test):

| Split | Groups | Description |
|-------|--------|-------------|
| Train | 43 | All remaining models + PPE subsets |
| Val | 9 | Next-largest per family (≥3 members) |
| Test | 10 | Largest per family (≥5 members) — never seen during training |

Test models: CNRM-CM6-1, CESM2-LENS2_cmip6, CanESM5-1_p1, EC-Earth3, MPI-ESM1-2-LR,
ACCESS-ESM1-5, INM-CM5-0, IPSL-CM6A-LR, MIROC6, GISS-E2-1-G_p1

**Key differences from base training:**
- Initializes from saved base model weights
- Lower LR: 1e-3 → 1e-6 (cosine annealing)
- OCEAN_WEIGHT = 1.0 (equal land/ocean weighting)
- 2000 train / 1000 val virtual samples per epoch
- Same augmentation strategy (p_aug=p_noise=p_zero=0.25)

---

## Inference

**Integration** (`integrate` in diagnostic notebooks):
```python
def integrate(model, x0, clim, steps=20, dt_end=1.0):
    x = x0.clone()
    dt = dt_end / steps
    for i in range(steps):
        t = i * dt
        v = model(x, clim, t)
        x = x + v * dt
    return x
```

- `DT` (default 1.0): integration endpoint. Set `DT < 1.0` to stop early and examine
  partial-flow predictions.
- `ODE_STEPS` (default 50): number of Euler steps.

**Ensemble prediction:** Run all non-garbage members independently and average.

**Member filtering** (in `check_flow_finetune.ipynb`): Before evaluation, each member is
run solo across all test models. Members with mean RMSE > `MEMBER_RMSE_THRESHOLD × median`
are dropped. Default threshold: 3.0×.

---

## Baselines

| Baseline | Description |
|----------|-------------|
| Train mean | Mean dPdP across all training-set models — the simplest multi-model average |
| Base model | Pre-trained (not finetuned) model predictions |

Key evaluation question: does the flow model RMSE on the test set beat the training mean baseline?

---

## Data Format

All processed files are on a **128×192 Gaussian grid** (matching the HadGEM3-PPE reference),
with **longitude −180 to 180**.

| Variable | Units | Description |
|----------|-------|-------------|
| `pr_clim` | mm/yr | 1979–2014 historical mean precipitation |
| `pr_trend` | dimensionless | Linear trend normalized by global-mean trend |
| `dPdP` | dimensionless | (future − hist) normalized by global-mean future change |

---

## Outputs

Weights saved to `~/Documents/research/weights/{EXPT_NAME}/`:
- `best_member{i}.pth` — model weights + metadata (epoch, val_loss, normalization stats, augmentation params)
- `log_member{i}.json` — per-epoch metrics: train loss, val loss, endpoint loss, learning rate
- `clim_stats.json` — global clim mean/std for normalization
- `split.json` — train/val/test group names (finetune only)
- `diagnostics/` — saved plots from check notebooks
