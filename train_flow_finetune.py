"""
train_flow_finetune.py
======================
Finetunes the base flow matching model on CMIP6 by-model data.

Differences from base training
------------------------------
  - Data: CMIP6 processed_by_model .nc files (real per-member trends)
  - Split: hardcoded train/val/test from cmip6_split.py
  - Validation: held-out *models*, not random split of same data
  - Trend augmentation: cross-model within same GCM family
  - Lower learning rate (LR_INIT) for finetuning
  - Harder ocean downweighting (OCEAN_WEIGHT = 0.1)
  - Loads pretrained base model weights

Data per model group
--------------------
  {group}_clim.nc   — (lat, lon)           ensemble-mean climatology
  {group}_trend.nc  — (member, lat, lon)   per-member normalized trend
                       or (lat, lon)        if single member
  {group}_dPdP.nc   — (lat, lon)           ensemble-mean forced change

Run
---
  python train_flow_finetune.py
"""

import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import xarray as xr

from flow_models import Unet6R
from cmip6_split import (TRAIN_GROUPS, VAL_GROUPS, TEST_GROUPS, group_family, TRAIN_SET, VAL_SET)

# ── configuration ──────────────────────────────────────────────────────────────

CMIP6_DIR      = Path("/Users/ewellmeyer/Documents/research/CMIP6/processed_by_model")
LANDMASK_PATH  = Path("/Users/ewellmeyer/Documents/research/HadGEM/hadgem_landmask_rg128.nc")
WEIGHTS_DIR    = Path("/Users/ewellmeyer/Documents/research/weights")

# base model to load (directory containing best_member*.pth and clim_stats.json)
BASE_CHANNELS  = 16
BASE_P_AUG     = 0.8
BASE_P_NOISE   = 0.05
BASE_P_ZERO    = 0.05
BASE_T_BETA_ALPHA = 0.5
BASE_T_BETA_BETA  = 0.5
BASE_ENDPOINT_PENALTY_WEIGHT = 0.05
BASE_EXPT      = (
    f"flow_base_unet6R_ch{BASE_CHANNELS}_land10_oce0.3"
    f"_aug{BASE_P_AUG}_noise{BASE_P_NOISE}_zero{BASE_P_ZERO}"
    f"_tbeta{BASE_T_BETA_ALPHA}_{BASE_T_BETA_BETA}"
    f"_vend{BASE_ENDPOINT_PENALTY_WEIGHT}"
)
BASE_DIR       = WEIGHTS_DIR / BASE_EXPT

# finetuning hyperparameters
N_ENSEMBLE     = 5
LR_INIT        = 1e-3      # lower than base (1e-3) for finetuning
LR_MIN         = 1e-6
N_EPOCHS       = 5000
PATIENCE       = 30
BATCH_SIZE     = 100
GRAD_CLIP      = 1.0
P_DROP         = 0.05
P_AUG          = 0.25       # cross-model trend augmentation probability
P_NOISE        = 0.25      # replace x0 with random Gaussian noise
P_ZERO         = 0.25      # replace x0 with zeros
LAND_WEIGHT    = 1.0
OCEAN_WEIGHT   = 1.0       
T_BETA_ALPHA   = 0.5       # <1 biases t samples toward 0 and 1
T_BETA_BETA    = 0.5
ENDPOINT_PENALTY_WEIGHT = 0.05
ENDPOINT_PENALTY_START  = 0.8
USE_AMP        = True
TRAIN_LEN      = 2000     # virtual epoch length (samples per epoch)
VAL_LEN        = 1000

if P_AUG + P_NOISE + P_ZERO > 1.0:
    raise ValueError("P_AUG + P_NOISE + P_ZERO must be <= 1.0")
if min(T_BETA_ALPHA, T_BETA_BETA) <= 0.0:
    raise ValueError("Beta time-sampling parameters must be positive")
if not 0.0 <= ENDPOINT_PENALTY_START < 1.0:
    raise ValueError("ENDPOINT_PENALTY_START must lie in [0, 1)")

EXPT_NAME = (
    f"flow_finetune_unet6R_ch{BASE_CHANNELS}_oce{OCEAN_WEIGHT}"
    f"_aug{P_AUG}_noise{P_NOISE}_zero{P_ZERO}"
    f"_tbeta{T_BETA_ALPHA}_{T_BETA_BETA}"
    f"_vend{ENDPOINT_PENALTY_WEIGHT}"
)
SAVE_DIR  = WEIGHTS_DIR / EXPT_NAME
SAVE_DIR.mkdir(parents=True, exist_ok=True)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

NUM_WORKERS = 0 if DEVICE.type in ("mps", "cpu") else 4
PIN_MEMORY  = DEVICE.type == "cuda"


# ── data loading ───────────────────────────────────────────────────────────────

def load_group(name, data_dir):
    """
    Load a single model group's clim, trend, and dPdP from netCDF.
    Returns dict with numpy arrays, or None if files are missing.

    trend is always returned as (N_members, H, W).
    clim and dpdp are (H, W).
    """
    clim_path  = data_dir / f"{name}_clim.nc"
    trend_path = data_dir / f"{name}_trend.nc"
    dpdp_path  = data_dir / f"{name}_dPdP.nc"

    if not all(p.exists() for p in [clim_path, trend_path, dpdp_path]):
        return None

    try:
        ds_c = xr.open_dataset(clim_path)
        clim = ds_c["pr_clim"].values.astype(np.float32)  # (H, W)
        ds_c.close()

        ds_t = xr.open_dataset(trend_path)
        trend = ds_t["pr_trend"].values.astype(np.float32)
        ds_t.close()
        # ensure 3D: (member, H, W)
        if trend.ndim == 2:
            trend = trend[np.newaxis, :, :]

        ds_d = xr.open_dataset(dpdp_path)
        dpdp = ds_d["dPdP"].values.astype(np.float32)  # (H, W)
        ds_d.close()

        return dict(clim=clim, trends=trend, dpdp=dpdp,
                    n_members=trend.shape[0])

    except Exception as e:
        print(f"  WARN: failed to load {name}: {e}")
        return None


def load_split(group_list, data_dir):
    """
    Load all groups in a split list.
    Returns list of dicts: {name, family, clim, trends, dpdp, n_members}
    """
    loaded = []
    for name, family in group_list:
        data = load_group(name, data_dir)
        if data is not None:
            data["name"]   = name
            data["family"] = family
            loaded.append(data)
        else:
            print(f"  SKIP: {name} (missing files)")
    return loaded


# ── dataset ────────────────────────────────────────────────────────────────────

class CMIP6FlowDataset(Dataset):
    """
    Flow matching dataset for CMIP6 finetuning.

    Sampling strategy:
      1. Pick a model group uniformly at random
      2. Pick a member uniformly within that group
      3. x0 = member's trend, x1 = group's dPdP, c = group's clim
      4. With probability p_aug, replace x0 with a random trend
         from the same GCM family's trend pool

    All fields are returned as (1, H, W) tensors.
    """

    def __init__(self, groups, clim_mean, clim_std, land_mask,
                 p_aug=0.0, p_noise=0.0, p_zero=0.0, length=10000):
        if p_aug + p_noise + p_zero > 1.0:
            raise ValueError("p_aug + p_noise + p_zero must be <= 1.0")
        self.groups    = groups
        self.clim_mean = clim_mean
        self.clim_std  = clim_std
        self.p_aug     = p_aug
        self.p_noise   = p_noise
        self.p_zero    = p_zero
        self.length    = length

        # (1, H, W) mask tensor
        self.mask = torch.from_numpy(land_mask)[None, :, :]

        # build family trend pools for augmentation
        self.family_pools = defaultdict(list)
        for g in groups:
            for m in range(g["n_members"]):
                self.family_pools[g["family"]].append(g["trends"][m])

        # convert pool lists to arrays for fast random indexing
        for fam in self.family_pools:
            self.family_pools[fam] = np.stack(self.family_pools[fam], axis=0)

        n_total = sum(g["n_members"] for g in groups)
        print(f"  CMIP6FlowDataset: {len(groups)} groups, {n_total} total members, "
              f"{len(self.family_pools)} family pools")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # pick a random group
        g = random.choice(self.groups)

        # pick a random member's trend as x0
        m_idx = random.randint(0, g["n_members"] - 1)
        x0 = g["trends"][m_idx]  # (H, W)

        # trend augmentation: mutually exclusive branches
        r = random.random()
        if r < self.p_aug:
            # cross-family swap
            pool = self.family_pools[g["family"]]
            x0 = pool[random.randint(0, len(pool) - 1)]
        elif r < self.p_aug + self.p_noise:
            # replace with Gaussian noise at a random scale
            noise_std = random.uniform(0.5, 2.0)
            x0 = np.random.randn(*x0.shape).astype(np.float32) * noise_std
        elif r < self.p_aug + self.p_noise + self.p_zero:
            # replace with zeros
            x0 = np.zeros_like(x0)

        x1   = g["dpdp"]   # (H, W)
        clim = g["clim"]   # (H, W)

        # normalize clim
        clim = (clim - self.clim_mean) / self.clim_std

        # convert to (1, H, W) tensors
        clim_t = torch.from_numpy(clim)[None, :, :]
        x0_t   = torch.from_numpy(x0.copy())[None, :, :]
        x1_t   = torch.from_numpy(x1.copy())[None, :, :]

        return clim_t, x0_t, x1_t, self.mask


# ── loss ───────────────────────────────────────────────────────────────────────

def masked_mse(v_pred, v_target, mask):
    return (mask * (v_pred - v_target) ** 2).mean()


def sample_times(batch_size, device):
    """Sample t with extra mass near the boundaries when alpha,beta < 1."""
    if T_BETA_ALPHA == 1.0 and T_BETA_BETA == 1.0:
        return torch.rand(batch_size, device=device)
    dist = torch.distributions.Beta(T_BETA_ALPHA, T_BETA_BETA)
    return dist.sample((batch_size,)).to(device=device)


def endpoint_velocity_penalty(v_pred, mask, t):
    """Penalize residual velocity near t=1 so integration settles cleanly."""
    gate = F.relu((t - ENDPOINT_PENALTY_START) / (1.0 - ENDPOINT_PENALTY_START))
    gate = gate[:, None, None, None]
    return (gate * mask * v_pred.square()).mean()


# ── main ───────────────────────────────────────────────────────────────────────

def main():

    # =========================================================================
    # load land mask
    # =========================================================================

    print("Loading land mask...")
    ds_mask   = xr.open_dataset(LANDMASK_PATH)
    mask_var  = list(ds_mask.data_vars)[0]
    land_frac = ds_mask[mask_var].values.astype(np.float32)
    land_mask = OCEAN_WEIGHT + (LAND_WEIGHT - OCEAN_WEIGHT) * land_frac
    print(f"  shape: {land_mask.shape}  "
          f"weight range: [{land_mask.min():.2f}, {land_mask.max():.2f}]")

    # =========================================================================
    # load clim normalization from base training
    # =========================================================================

    stats_path = BASE_DIR / "clim_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Cannot find {stats_path}. Train the base model first.")
    with open(stats_path) as f:
        stats = json.load(f)
    clim_mean = stats["clim_mean"]
    clim_std  = stats["clim_std"]
    print(f"Clim normalization (from base): mean={clim_mean:.2f} std={clim_std:.2f}")

    # =========================================================================
    # load CMIP6 data
    # =========================================================================

    print(f"\nLoading CMIP6 data from {CMIP6_DIR}")
    print(f"  Train groups: {len(TRAIN_GROUPS)}")
    train_data = load_split(TRAIN_GROUPS, CMIP6_DIR)
    print(f"  -> loaded {len(train_data)} train groups, "
          f"{sum(g['n_members'] for g in train_data)} total members")

    print(f"  Val groups: {len(VAL_GROUPS)}")
    val_data = load_split(VAL_GROUPS, CMIP6_DIR)
    print(f"  -> loaded {len(val_data)} val groups, "
          f"{sum(g['n_members'] for g in val_data)} total members")

    if not train_data:
        raise RuntimeError("No training data loaded!")
    if not val_data:
        raise RuntimeError("No validation data loaded!")

    # =========================================================================
    # build datasets and dataloaders
    # =========================================================================

    train_ds = CMIP6FlowDataset(
        train_data, clim_mean, clim_std, land_mask,
        p_aug=P_AUG, p_noise=P_NOISE, p_zero=P_ZERO, length=TRAIN_LEN)
    val_ds = CMIP6FlowDataset(
        val_data, clim_mean, clim_std, land_mask,
        p_aug=0.0, length=VAL_LEN)   # no augmentation for val

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    print(f"\nTrain: {TRAIN_LEN} samples ({len(train_loader)} batches)")
    print(f"Val:   {VAL_LEN} samples ({len(val_loader)} batches)")
    print(f"Device: {DEVICE}  Workers: {NUM_WORKERS}")

    # =========================================================================
    # ensemble training loop
    # =========================================================================

    for member_idx in range(0, N_ENSEMBLE):
        print(f"\n{'='*60}")
        print(f"Ensemble member {member_idx}")
        print(f"{'='*60}")

        torch.manual_seed(member_idx + 1000)  # offset from base seeds
        np.random.seed(member_idx + 1000)
        random.seed(member_idx + 1000)

        # ── build model and load base weights ────────────────────────────────
        model = Unet6R(input_channels=2, output_channels=1,
                       base_channels=BASE_CHANNELS,
                       p_drop=P_DROP).to(DEVICE)

        base_ckpt_path = BASE_DIR / f"best_member{member_idx}.pth"
        if base_ckpt_path.exists():
            ckpt = torch.load(base_ckpt_path, map_location=DEVICE)
            model.load_state_dict(ckpt["state_dict"])
            print(f"  loaded base weights from {base_ckpt_path.name} "
                  f"(base val_loss={ckpt['val_loss']:.5f})")
        else:
            print(f"  WARN: no base checkpoint at {base_ckpt_path.name}, "
                  f"training from scratch")

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR_INIT,
                                       weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=N_EPOCHS, eta_min=LR_MIN)
        scaler    = (torch.cuda.amp.GradScaler()
                     if USE_AMP and DEVICE.type == "cuda" else None)

        save_path  = SAVE_DIR / f"best_member{member_idx}.pth"
        log_path   = SAVE_DIR / f"log_member{member_idx}.json"
        best_val       = float("inf")
        patience_count = 0
        log            = []

        # ── resume from finetune checkpoint if it exists ─────────────────────
        start_epoch = 1
        if save_path.exists():
            ft_ckpt = torch.load(save_path, map_location=DEVICE)
            model.load_state_dict(ft_ckpt["state_dict"])
            best_val = ft_ckpt["val_loss"]
            print(f"  resuming finetune from epoch {ft_ckpt['epoch']} "
                  f"(val_loss={best_val:.5f})")
            if log_path.exists():
                with open(log_path) as f:
                    log = json.load(f)
                start_epoch = log[-1]["epoch"] + 1

        epoch_start = time.time()

        for epoch in range(start_epoch, N_EPOCHS + 1):

            # ── train ─────────────────────────────────────────────────────────
            model.train()
            train_loss = 0.0
            train_mse = 0.0
            train_endpoint = 0.0
            t_epoch    = time.time()

            for clim, x0, x1, mask in train_loader:
                clim, x0, x1, mask = (clim.to(DEVICE), x0.to(DEVICE),
                                       x1.to(DEVICE),  mask.to(DEVICE))
                B  = clim.shape[0]
                t  = sample_times(B, DEVICE)
                t4 = t[:, None, None, None]
                xt = (1 - t4) * x0 + t4 * x1

                optimizer.zero_grad()
                with torch.autocast(device_type=DEVICE.type,
                                    enabled=scaler is not None):
                    v_pred = model(xt, clim, t)
                    mse_loss = masked_mse(v_pred, x1 - x0, mask)
                    endpoint_loss = endpoint_velocity_penalty(v_pred, mask, t)
                    loss = mse_loss + ENDPOINT_PENALTY_WEIGHT * endpoint_loss

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()

                train_loss += loss.item()
                train_mse += mse_loss.item()
                train_endpoint += endpoint_loss.item()

            train_loss /= len(train_loader)
            train_mse /= len(train_loader)
            train_endpoint /= len(train_loader)

            # ── validate ──────────────────────────────────────────────────────
            model.eval()
            val_loss = 0.0
            val_mse = 0.0
            val_endpoint = 0.0
            with torch.no_grad():
                for clim, x0, x1, mask in val_loader:
                    clim, x0, x1, mask = (clim.to(DEVICE), x0.to(DEVICE),
                                           x1.to(DEVICE),  mask.to(DEVICE))
                    B  = clim.shape[0]
                    t  = sample_times(B, DEVICE)
                    t4 = t[:, None, None, None]
                    xt = (1 - t4) * x0 + t4 * x1
                    v_pred = model(xt, clim, t)
                    mse_loss = masked_mse(v_pred, x1 - x0, mask)
                    endpoint_loss = endpoint_velocity_penalty(v_pred, mask, t)
                    val_loss += (mse_loss + ENDPOINT_PENALTY_WEIGHT * endpoint_loss).item()
                    val_mse += mse_loss.item()
                    val_endpoint += endpoint_loss.item()
            val_loss /= len(val_loader)
            val_mse /= len(val_loader)
            val_endpoint /= len(val_loader)

            scheduler.step()
            epoch_time = time.time() - t_epoch
            log.append({
                "epoch": epoch,
                "train": train_loss,
                "train_mse": train_mse,
                "train_endpoint": train_endpoint,
                "val": val_loss,
                "val_mse": val_mse,
                "val_endpoint": val_endpoint,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_time": epoch_time,
            })

            # print progress
            elapsed = time.time() - epoch_start
            epochs_done = epoch - start_epoch + 1
            eta = elapsed / epochs_done * (N_EPOCHS - epoch)
            improved = "  *" if val_loss < best_val else ""
            print(f"  epoch {epoch:3d}/{N_EPOCHS}  "
                  f"train={train_loss:.5f} (mse={train_mse:.5f}, end={train_endpoint:.5f})  "
                  f"val={val_loss:.5f} (mse={val_mse:.5f}, end={val_endpoint:.5f})  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                  f"{epoch_time:.1f}s  ETA {eta/60:.1f}min{improved}")

            if val_loss < best_val:
                best_val       = val_loss
                patience_count = 0
                torch.save({
                    "epoch":        epoch,
                    "state_dict":   model.state_dict(),
                    "val_loss":     val_loss,
                    "clim_mean":    clim_mean,
                    "clim_std":     clim_std,
                    "base_expt":    BASE_EXPT,
                    "ocean_weight": OCEAN_WEIGHT,
                    "p_aug":        P_AUG,
                    "p_noise":      P_NOISE,
                    "p_zero":       P_ZERO,
                    "t_beta_alpha": T_BETA_ALPHA,
                    "t_beta_beta":  T_BETA_BETA,
                    "endpoint_penalty_weight": ENDPOINT_PENALTY_WEIGHT,
                    "endpoint_penalty_start":  ENDPOINT_PENALTY_START,
                }, save_path)
            else:
                patience_count += 1
                if patience_count >= PATIENCE:
                    print(f"  early stopping at epoch {epoch} "
                          f"(no improvement for {PATIENCE} epochs)")
                    break

        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

        print(f"  member {member_idx} done.  best val = {best_val:.5f}  "
              f"stopped at epoch {log[-1]['epoch']}")

    # save the split used for this run
    split_info = {
        "train": [name for name, _ in TRAIN_GROUPS],
        "val":   [name for name, _ in VAL_GROUPS],
        "test":  [name for name, _ in TEST_GROUPS],
    }
    with open(SAVE_DIR / "split.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print("\nAll ensemble members finetuned.")
    print(f"Weights saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
