"""
train_flow_base.py
==================
Base model training for conditional flow matching.

Flow matching formulation
-------------------------
  x0      historical trend (normalized, dimensionless)  — noisy source
  x1      dPdP (future precip change, normalized)        — target
  c       historical climatology (mm/yr, normalized)     — conditioning field
  xt      (1-t)*x0 + t*x1                               — interpolated field
  v*      x1 - x0                                       — training target

Network input  : [xt, c] concatenated -> (B, 2, H, W), plus scalar t
Network output : predicted velocity v ~ x1 - x0

Loss
----
Weighted MSE with a land mask: ocean pixels are downweighted to focus
the model on land precipitation where predictions matter and the physics
are more constrained. Loss = mean( mask * (v_pred - v_target)^2 ).

Training data
-------------
  HadGEM3-PPE  1500 members, no trend  -> x0 from HadAM3 AMIP trend pool
  CESM2-PPE     262 members, no trend  -> x0 from CCM0B AMIP trend pool
  AMIP           11 models, real trend -> x0 = own trend (or augmented)

Run this script
---------------
  python train_flow_base.py
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import xarray as xr

from flow_models   import Unet6R
from flow_datasets import (FlowDataset, model_family, load_nc_field)

# configuration

HADGEM_PPE_DIR = Path("/Users/ewellmeyer/Documents/research/HadGEM")
CESM2_PPE_DIR  = Path("/Users/ewellmeyer/Documents/research/CESM2")
AMIP_DIR       = Path("/Users/ewellmeyer/Documents/research/AMIP/processed")
LANDMASK_PATH  = Path("/Users/ewellmeyer/Documents/research/HadGEM/hadgem_landmask_rg128.nc")
WEIGHTS_DIR    = Path("/Users/ewellmeyer/Documents/research/weights")


N_ENSEMBLE   = 5
PATIENCE     = 30
BATCH_SIZE   = 100
LR           = 1e-3
N_EPOCHS     = 1000
GRAD_CLIP    = 1.0
P_DROP       = 0.0
P_AUG        = 0.8   # probability of cross-model trend augmentation (AMIP)
LAND_WEIGHT  = 1.0   # loss weight for land pixels
OCEAN_WEIGHT = 0.3   # loss weight for ocean pixels (soft downweight for base)
USE_AMP      = True
DEVICE       = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

BASE_CHANNELS = 8
EXPT_NAME   = f"flow_base_unet6R_ch{BASE_CHANNELS}_land10_oce{OCEAN_WEIGHT}_aug{P_AUG}"
WEIGHTS_DIR = WEIGHTS_DIR / EXPT_NAME
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# MPS does not support pin_memory or multi-process DataLoader reliably
NUM_WORKERS = 0 if DEVICE.type in ("mps", "cpu") else 4
PIN_MEMORY  = DEVICE.type == "cuda"

def masked_mse(v_pred, v_target, mask):
    """
    Pixel-weighted MSE.
      v_pred, v_target : (B, 1, H, W)
      mask             : (B, 1, H, W)  per-pixel weights
    """
    return (mask * (v_pred - v_target) ** 2).mean()


def main():

    # =========================================================================
    # load land mask
    # =========================================================================

    print("Loading land mask...")
    ds_mask   = xr.open_dataset(LANDMASK_PATH)
    mask_var  = list(ds_mask.data_vars)[0]
    land_frac = ds_mask[mask_var].values.astype(np.float32)   # (H, W) in [0,1]

    land_mask_weights = OCEAN_WEIGHT + (LAND_WEIGHT - OCEAN_WEIGHT) * land_frac
    print(f"  shape: {land_mask_weights.shape}  "
          f"land fraction: {land_frac.mean():.3f}  "
          f"weight range: [{land_mask_weights.min():.2f}, {land_mask_weights.max():.2f}]")

    # =========================================================================
    # load training data
    # =========================================================================

    print("\nLoading training data...")
    sources = []

    # HadGEM3-PPE atmospheric PPE (1500 members, no historical trend)
    try:
        hg_clim = load_nc_field(HADGEM_PPE_DIR / "GA789_PR_his_rg128.nc", "PR")
        hg_x1   = load_nc_field(HADGEM_PPE_DIR / "GA789_dPdP_rg128.nc",   "dPdP")
        sources.append(dict(name="HadGEM3-PPE", family="HadAM3",
                            clim=hg_clim, x1=hg_x1, x0=None, is_ppe=True))
        print(f"  HadGEM3-PPE      : {hg_clim.shape[0]} members")
    except Exception as e:
        print(f"  WARN HadGEM3-PPE : {e}")

    # CESM2-PPE (262 members, no historical trend)
    try:
        c2_clim = load_nc_field(CESM2_PPE_DIR / "CESM2_PR_his_rg128.nc",  "PR")
        c2_x1   = load_nc_field(CESM2_PPE_DIR / "CESM2_dPdP_rg128.nc",    "dPdP")
        sources.append(dict(name="CESM2-PPE", family="CCM0B",
                            clim=c2_clim, x1=c2_x1, x0=None, is_ppe=True))
        print(f"  CESM2-PPE        : {c2_clim.shape[0]} members")
    except Exception as e:
        print(f"  WARN CESM2-PPE   : {e}")

    # AMIP models — now reading .nc files instead of .zarr
    for clim_path in sorted(AMIP_DIR.glob("AMIP_*_clim.nc")):
        model      = clim_path.name.replace("AMIP_", "").replace("_clim.nc", "")
        trend_path = AMIP_DIR / f"AMIP_{model}_trend.nc"
        dpdp_path  = AMIP_DIR / f"AMIP_{model}_dPdP.nc"
        if not trend_path.exists() or not dpdp_path.exists():
            print(f"  WARN AMIP {model}: missing trend or dPdP, skipping")
            continue
        try:
            clim  = load_nc_field(clim_path,  "pr_clim")
            trend = load_nc_field(trend_path, "pr_trend")
            dpdp  = load_nc_field(dpdp_path,  "dPdP")
            fam   = model_family(model)
            sources.append(dict(name=model, family=fam,
                                clim=clim, x1=dpdp, x0=trend, is_ppe=False))
            print(f"  AMIP {model:<20} ({fam}): {clim.shape[0]} member(s)")
        except Exception as e:
            print(f"  WARN AMIP {model}: {e}")

    print(f"\nTotal sources loaded: {len(sources)}")

    # =========================================================================
    # compute clim normalization stats from all training data
    # =========================================================================

    all_clim = np.concatenate([s["clim"].reshape(-1) for s in sources])
    all_clim = all_clim[np.isfinite(all_clim)]
    clim_mean, clim_std = float(all_clim.mean()), float(all_clim.std())
    print(f"Clim normalization:  mean={clim_mean:.2f}  std={clim_std:.2f} mm/yr")

    with open(WEIGHTS_DIR / "clim_stats.json", "w") as f:
        json.dump({"clim_mean": clim_mean, "clim_std": clim_std}, f)

    # =========================================================================
    # build dataset and dataloaders
    # =========================================================================

    dataset = FlowDataset(
        sources   = sources,
        clim_mean = clim_mean,
        clim_std  = clim_std,
        p_aug     = P_AUG,
        length    = 10000,
        land_mask = land_mask_weights,
    )

    n_val   = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    print(f"\nDataset: {n_train} train  {n_val} val  "
          f"({len(train_loader)} / {len(val_loader)} batches)  "
          f"device: {DEVICE}  workers: {NUM_WORKERS}")

    # =========================================================================
    # ensemble training loop
    # =========================================================================

    for member_idx in range(N_ENSEMBLE):
        print(f"\n{'='*60}")
        print(f"Ensemble member {member_idx}")
        print(f"{'='*60}")

        torch.manual_seed(member_idx)
        np.random.seed(member_idx)
        random.seed(member_idx)

        model     = Unet6R(input_channels=2, output_channels=1,
                           base_channels=BASE_CHANNELS,
                           p_drop=P_DROP).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=N_EPOCHS, eta_min=LR * 0.01)
        scaler    = (torch.cuda.amp.GradScaler()
                     if USE_AMP and DEVICE.type == "cuda" else None)

        save_path = WEIGHTS_DIR / f"best_member{member_idx}.pth"
        log_path  = WEIGHTS_DIR / f"log_member{member_idx}.json"
        best_val      = float("inf")
        patience_count = 0
        log           = []

        start_epoch = 1
        if save_path.exists():
            ckpt = torch.load(save_path, map_location=DEVICE)
            model.load_state_dict(ckpt["state_dict"])
            best_val = ckpt["val_loss"]
            print(f"  loaded checkpoint from epoch {ckpt['epoch']}  "
                f"(val_loss={best_val:.5f})")
            if log_path.exists():
                with open(log_path) as f:
                    log = json.load(f)
                start_epoch = log[-1]["epoch"] + 1
                print(f"  resuming from epoch {start_epoch}")

        epoch_start = time.time()

        for epoch in range(start_epoch, N_EPOCHS + 1):

            # ── train ─────────────────────────────────────────────────────────
            model.train()
            train_loss = 0.0
            n_batches  = len(train_loader)
            t_epoch    = time.time()

            for batch_idx, (clim, x0, x1, mask) in enumerate(train_loader):
                clim, x0, x1, mask = (clim.to(DEVICE), x0.to(DEVICE),
                                       x1.to(DEVICE),  mask.to(DEVICE))
                B  = clim.shape[0]
                t  = torch.rand(B, device=DEVICE)
                t4 = t[:, None, None, None]
                xt = (1 - t4) * x0 + t4 * x1

                optimizer.zero_grad()
                with torch.autocast(device_type=DEVICE.type,
                                    enabled=scaler is not None):
                    v_pred = model(xt, clim, t)
                    loss   = masked_mse(v_pred, x1 - x0, mask)

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

            train_loss /= len(train_loader)

            # ── validate ──────────────────────────────────────────────────────
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for clim, x0, x1, mask in val_loader:
                    clim, x0, x1, mask = (clim.to(DEVICE), x0.to(DEVICE),
                                           x1.to(DEVICE),  mask.to(DEVICE))
                    B  = clim.shape[0]
                    t  = torch.rand(B, device=DEVICE)
                    t4 = t[:, None, None, None]
                    xt = (1 - t4) * x0 + t4 * x1
                    v_pred    = model(xt, clim, t)
                    val_loss += masked_mse(v_pred, x1 - x0, mask).item()
            val_loss /= len(val_loader)

            scheduler.step()
            epoch_time = time.time() - t_epoch
            log.append({"epoch": epoch, "train": train_loss, "val": val_loss,
                        "epoch_time": epoch_time})

            if epoch % 1 == 0:
                elapsed_total = time.time() - epoch_start
                epochs_done = epoch - start_epoch + 1
                eta = elapsed_total / epochs_done * (N_EPOCHS - epoch)
                print(f"  epoch {epoch:3d}/{N_EPOCHS}  "
                      f"train={train_loss:.5f}  val={val_loss:.5f}  "
                      f"{epoch_time:.1f}s/epoch  "
                      f"ETA {eta/60:.1f}min")

            if val_loss < best_val:
                best_val       = val_loss
                patience_count = 0
                torch.save({
                    "epoch":      epoch,
                    "state_dict": model.state_dict(),
                    "val_loss":   val_loss,
                    "clim_mean":  clim_mean,
                    "clim_std":   clim_std,
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

    print("\nAll ensemble members trained.")


if __name__ == "__main__":
    main()