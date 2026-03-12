"""
flow_datasets.py
================
Dataset classes and data loading utilities for flow matching training.

Exports
-------
  GCM_FAMILIES   : dict mapping family name -> list of model names
  model_family   : look up family for a model name
  TrendPool      : collection of trend fields for family-level x0 sampling
  FlowDataset    : PyTorch Dataset for (clim, x0, x1) flow matching samples
  load_nc_field  : load variable from NetCDF as (members, lat, lon) float32
  load_zarr_field: load variable from zarr as (members, lat, lon) float32
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr


# ── family mapping ─────────────────────────────────────────────────────────────
# Used to route PPE members to the correct AMIP trend pool for x0 sampling,
# and for cross-model trend augmentation within AMIP training data.

GCM_FAMILIES = {
    "CCM0B":  ["BCC-CSM2-MR", "CESM2", "E3SM-1-0", "TaiESM1",
               "NorESM2-LM", "NorESM2-MM", "CESM2-WACCM", "FIO-ESM-2-0",
               "CAS-ESM2-0", "FGOALS-f3-L", "FGOALS-g3", "CIESM"],
    "ECHAM0": ["CAMS-CSM1-0", "MPI-ESM1-2-HR", "MPI-ESM1-2-LR",
               "AWI-CM-1-1-MR", "NESM3"],
    "ARPEGE": ["CNRM-CM6-1", "CNRM-CM6-1-HR", "CNRM-ESM2-1"],
    "EC-Earth3": ["EC-Earth3", "EC-Earth3-CC", "EC-Earth3-Veg", "EC-Earth3-Veg-LR"],
    "GFDL":   ["GFDL-CM4", "GFDL-ESM4", "KIOST-ESM"],
    "MIROC":  ["MIROC6", "MIROC-ES2L"],
    "UCLA":   ["MRI-ESM2-0", "GISS-E2-1-G", "GISS-E2-1-G-CC", "GISS-E2-1-H"],
    "HadAM3": ["HadGEM3-GC31-LL", "HadGEM3-GC31-MM", "UKESM1-0-LL",
               "KACE-1-0-G", "ACCESS-CM2", "ACCESS-ESM1-5"],
    "CanAM3": ["CanESM5", "CanESM5-1", "CanESM5-CanOE"],
    "INM":    ["INM-CM4-8", "INM-CM5-0"],
    "IPSL":   ["IPSL-CM6A-LR", "IPSL-CM6A-ATM", "IPSL-CM6A-ESM",
               "IPSL-CM5A2-INCA"],
}

def model_family(model_name):
    for fam, members in GCM_FAMILIES.items():
        if model_name in members:
            return fam
    return "unknown"


# ── file loading ───────────────────────────────────────────────────────────────

def load_nc_field(path, var):
    """
    Load `var` from a NetCDF file.
    Returns float32 array of shape (members, lat, lon).
    Adds a member dim if the variable is 2D.
    """
    ds = xr.open_dataset(path)
    da = ds[var].values.astype(np.float32)
    if da.ndim == 2:
        da = da[np.newaxis]
    return da


def load_zarr_field(path, var):
    """
    Load `var` from a zarr store.
    Returns float32 array of shape (members, lat, lon).
    """
    ds = xr.open_zarr(path)
    da = ds[var].values.astype(np.float32)
    if da.ndim == 2:
        da = da[np.newaxis]
    return da


# ── trend pool ─────────────────────────────────────────────────────────────────

class TrendPool:
    """
    Holds a collection of normalized trend fields for a GCM family.
    Used to sample x0 for PPE members (which have no real trend) and for
    cross-model augmentation of AMIP training samples.
    """
    def __init__(self):
        self.trends = []

    def add(self, trend_array):
        """Add trend fields. trend_array: (members, lat, lon) or (lat, lon)."""
        if trend_array.ndim == 2:
            trend_array = trend_array[np.newaxis]
        for i in range(trend_array.shape[0]):
            self.trends.append(trend_array[i])

    def sample(self):
        return random.choice(self.trends)

    def __len__(self):
        return len(self.trends)


# ── dataset ────────────────────────────────────────────────────────────────────

class FlowDataset(Dataset):
    """
    Infinite-style dataset for conditional flow matching training.
    Each __getitem__ call returns a randomly sampled (clim, x0, x1) triple.

    Sampling strategy
    -----------------
    Two-level:
      1. Sample a source (model/dataset) uniformly from the sources list.
         This prevents large ensembles from dominating training.
      2. Sample a member uniformly from that source.

    x0 selection
    ------------
    For AMIP sources (is_ppe=False):
      With probability p_aug, replace x0 with a trend sampled from the
      same family's trend pool (cross-model augmentation).
      Otherwise use the member's own trend.
    For PPE sources (is_ppe=True):
      Always sample x0 from the family trend pool since the PPE runs are
      too short to compute meaningful trends.
      Falls back to the global pool if no family pool exists.

    Parameters
    ----------
    sources : list of dicts, each containing:
        'name'   (str)           model/dataset name
        'family' (str)           GCM family name
        'clim'   (N, H, W)       historical climatology in mm/yr
        'x1'     (N, H, W)       dPdP (flow target)
        'x0'     (N, H, W)|None  historical trend; None for PPE sources
        'is_ppe' (bool)          True if source has no real trend
    clim_mean : float  global mean of clim across all training data (mm/yr)
    clim_std  : float  global std  of clim across all training data (mm/yr)
    p_aug     : float  probability of cross-model trend augmentation
    length    : int    virtual dataset length (samples per epoch)
    land_mask : (H, W) float32 array of pixel weights, or None for uniform
    """

    def __init__(self, sources, clim_mean, clim_std,
                 p_aug=0.8, length=10000, land_mask=None):
        self.sources   = sources
        self.clim_mean = clim_mean
        self.clim_std  = clim_std
        self.p_aug     = p_aug
        self._len      = length
        self.land_mask = land_mask   # (H, W) float32 or None

        # build per-family trend pools from AMIP sources with real trends
        self.family_pools = {}
        self.global_pool  = TrendPool()
        for src in sources:
            if not src["is_ppe"] and src["x0"] is not None:
                fam = src["family"]
                if fam not in self.family_pools:
                    self.family_pools[fam] = TrendPool()
                self.family_pools[fam].add(src["x0"])
                self.global_pool.add(src["x0"])

    def _get_pool(self, family):
        pool = self.family_pools.get(family)
        if pool and len(pool) > 0:
            return pool
        return self.global_pool

    def __len__(self):
        return self._len

    def __getitem__(self, _):
        # level 1: uniform source sampling
        src = random.choice(self.sources)
        # level 2: uniform member sampling
        idx = random.randrange(src["clim"].shape[0])

        clim = src["clim"][idx]                                # (H, W)
        x1   = src["x1"][idx]                                  # (H, W)

        # normalize climatology
        clim_norm = (clim - self.clim_mean) / (self.clim_std + 1e-6)

        # x0 selection
        pool = self._get_pool(src["family"])
        if src["is_ppe"] or random.random() < self.p_aug:
            x0 = pool.sample()
        else:
            x0 = src["x0"][idx]

        # replace NaNs with zero (land-only models may have ocean NaNs)
        clim_norm = np.nan_to_num(clim_norm, nan=0.0)
        x0        = np.nan_to_num(x0,        nan=0.0)
        x1        = np.nan_to_num(x1,        nan=0.0)

        clim_t = torch.from_numpy(clim_norm[np.newaxis]).float()
        x0_t   = torch.from_numpy(x0[np.newaxis]).float()
        x1_t   = torch.from_numpy(x1[np.newaxis]).float()

        if self.land_mask is not None:
            mask_t = torch.from_numpy(self.land_mask[np.newaxis]).float()
            return clim_t, x0_t, x1_t, mask_t

        return clim_t, x0_t, x1_t