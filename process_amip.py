"""
process_amip.py

Processes AMIP historic and amip-future4K files for 11 models into:

  {out_dir}/AMIP_{MODEL}_clim.zarr    — 1980-2014 hist climatology (lat, lon)
  {out_dir}/AMIP_{MODEL}_trend.zarr   — 1980-2014 hist trend, normalized
                                         by global mean (dimensionless)
  {out_dir}/AMIP_{MODEL}_dPdP.zarr    — future4K - hist clim, normalized
                                         by global mean (dimensionless)

All outputs are on the native model grid regridded to 128x192 with
longitude ranging from -180 to 180.

Processing follows the same conventions as process_cmip6_by_model.py:
  - clim/trend computed on native grid, regridded as 2D fields
  - dPdP computed on regridded grid after regridding future and hist means
  - trend normalized by global_mean(trend)
  - dPdP normalized by global_mean(dP)

TARGET GRID
-----------
The target grid is read directly from an existing PPE file (HadGEM or CESM2)
so that AMIP outputs land on exactly the same lat/lon coordinates used by
the PPE and land-mask data.  This avoids subtle coordinate mismatches that
would otherwise appear when using np.linspace to approximate a Gaussian grid.
"""

import re
import shutil
from pathlib import Path

import numpy as np
import xarray as xr
import xesmf as xe

HIST_DIR         = Path("/Users/ewellmeyer/Documents/research/AMIP/PD/concatenated")
FUTR_DIR         = Path("/Users/ewellmeyer/Documents/research/AMIP/future4K/PR/cat")
OUT_DIR          = Path("/Users/ewellmeyer/Documents/research/AMIP/processed")
CONV             = 86400.0 * 365.25
HIST_Y0, HIST_Y1 = 1979, 2014
MIN_HIST_YRS     = 10
MIN_FUTR_YRS     = 10

# ---------------------------------------------------------------------------
# Target grid: read from an existing PPE file so coordinates match exactly.
# Fall back to the CESM2 file if HadGEM is missing, and vice versa.
# ---------------------------------------------------------------------------
_PPE_GRID_CANDIDATES = [
    Path("/Users/ewellmeyer/Documents/research/HadGEM/GA789_PR_his_rg128.nc"),
    Path("/Users/ewellmeyer/Documents/research/CESM2/CESM2_PR_his_rg128.nc"),
]


def _load_target_grid():
    """
    Read TARGET_LAT and TARGET_LON from the first available PPE file.

    The PPE files use coordinate names 'latitude'/'longitude', so we
    check for both naming conventions.  Returns (lat_array, lon_array).
    """
    for candidate in _PPE_GRID_CANDIDATES:
        if candidate.exists():
            ds = xr.open_dataset(candidate)
            # resolve coordinate names
            lat_name = "latitude" if "latitude" in ds.coords else "lat"
            lon_name = "longitude" if "longitude" in ds.coords else "lon"
            lat = ds.coords[lat_name].values.astype(np.float64)
            lon = ds.coords[lon_name].values.astype(np.float64)
            ds.close()
            print(f"Target grid loaded from {candidate.name}")
            print(f"  lat: [{lat.min():.4f}, {lat.max():.4f}]  n={len(lat)}")
            print(f"  lon: [{lon.min():.4f}, {lon.max():.4f}]  n={len(lon)}")
            return lat, lon

    raise FileNotFoundError(
        "No PPE reference grid found.  Ensure at least one of these exists:\n"
        + "\n".join(f"  {p}" for p in _PPE_GRID_CANDIDATES)
    )


TARGET_LAT, TARGET_LON = _load_target_grid()
N_LAT, N_LON = len(TARGET_LAT), len(TARGET_LON)

GCM_FAMILIES = {
    "CCM0B":   ["BCC-CSM2-MR", "CESM2", "E3SM-1-0", "TaiESM1",
                "NorESM2-LM", "NorESM2-MM"],
    "ECHAM0":  ["MPI-ESM1-2-HR", "MPI-ESM1-2-LR", "CNRM-CM6-1"],
    "ARPEGE":  ["CNRM-CM6-1", "CNRM-CM6-1-HR", "CNRM-ESM2-1"],
    "GFDL":    ["GFDL-CM4"],
    "MIROC":   ["MIROC6"],
    "UCLA":    ["MRI-ESM2-0"],
    "HadAM3":  ["HadGEM3-GC31-LL"],
    "CanAM3":  ["CanESM5"],
    "IPSL":    ["IPSL-CM6A-LR"],
}

def model_family(model_name):
    for fam, members in GCM_FAMILIES.items():
        if model_name in members:
            return fam
    return "unknown"


def model_from_filename(fname):
    """Extract model name from CMIP6-style AMIP filename."""
    m = re.match(r'pr_Amon_([^_]+)_amip', fname)
    return m.group(1) if m else None


def open_pr(path):
    ds = xr.open_dataset(path, use_cftime=True)
    pr = ds["pr"] * CONV
    pr.attrs["units"] = "mm/yr"
    renames = {}
    if "latitude"  in pr.dims: renames["latitude"]  = "lat"
    if "longitude" in pr.dims: renames["longitude"] = "lon"
    if renames:
        pr = pr.rename(renames)
    # convert to -180 to 180 if needed
    lon = pr.coords["lon"].values
    if lon.max() > 180:
        pr = pr.assign_coords(lon=((pr.coords["lon"] + 180) % 360) - 180)
        pr = pr.sortby("lon")
    pr = pr.sortby("time")
    _, idx = np.unique(pr.time.values, return_index=True)
    if len(idx) < len(pr.time):
        pr = pr.isel(time=idx)
    return pr


def annual_mean(da):
    return da.resample(time="YS").mean("time")


def sel_years(da, y0, y1):
    yrs = da.time.dt.year
    return da.isel(time=((yrs >= y0) & (yrs <= y1)))


def global_mean(da):
    weights = np.cos(np.deg2rad(da.lat))
    return float(da.weighted(weights).mean(("lat", "lon")))


def linear_trend_field(da):
    t = np.arange(len(da.time), dtype=float)
    da = da.assign_coords(time=t)
    p = da.polyfit(dim="time", deg=1)
    return p["polyfit_coefficients"].sel(degree=1).drop_vars("degree")


def make_regridder(da_in):
    """Build a bilinear regridder to the PPE-derived target grid."""
    ds_out = xr.Dataset({"lat": ("lat", TARGET_LAT),
                         "lon": ("lon", TARGET_LON)})
    return xe.Regridder(da_in, ds_out, method="bilinear",
                        periodic=True, reuse_weights=False)


def safe_zarr_write(ds, path):
    if path.exists():
        shutil.rmtree(path)
    ds.to_zarr(path, mode="w")
    print(f"  -> wrote {path.name}")


def find_file_pairs():
    """
    Match hist and future4K files by model name.
    Returns list of (model_name, hist_path, futr_path).
    """
    hist_files = {model_from_filename(f.name): f
                  for f in sorted(HIST_DIR.glob("pr_Amon_*_amip_*.nc"))
                  if model_from_filename(f.name)}

    futr_files = {model_from_filename(f.name): f
                  for f in sorted(FUTR_DIR.glob("pr_Amon_*_amip-future4K_*.nc"))
                  if model_from_filename(f.name)}

    pairs = []
    for model in sorted(set(hist_files) & set(futr_files)):
        pairs.append((model, hist_files[model], futr_files[model]))

    missing_futr = set(hist_files) - set(futr_files)
    missing_hist = set(futr_files) - set(hist_files)
    if missing_futr:
        print(f"WARN: no future file for: {sorted(missing_futr)}")
    if missing_hist:
        print(f"WARN: no hist file for: {sorted(missing_hist)}")

    return pairs


def process_model(model_name, hist_path, futr_path, out_dir):
    prefix = out_dir / f"AMIP_{model_name}"
    expected = [Path(str(prefix) + s)
                for s in ("_clim.zarr", "_trend.zarr", "_dPdP.zarr")]
    if all(p.exists() for p in expected):
        print(f"  [{model_name}] SKIP: all outputs already exist")
        return

    print(f"\n  [{model_name}]")
    print(f"    hist: {hist_path.name}")
    print(f"    futr: {futr_path.name}")

    try:
        pr_hist_raw = open_pr(hist_path)
    except Exception as e:
        print(f"    WARN: cannot open hist: {e}"); return

    pr_hist_ann = annual_mean(pr_hist_raw)
    pr_hist     = sel_years(pr_hist_ann, HIST_Y0, HIST_Y1)
    if len(pr_hist.time) < MIN_HIST_YRS:
        print(f"    WARN: only {len(pr_hist.time)} hist years -> skip"); return
    pr_hist = pr_hist.assign_coords(time=pr_hist.time.dt.year.values)
    print(f"    hist years: {len(pr_hist.time)}")

    try:
        pr_futr_raw = open_pr(futr_path)
    except Exception as e:
        print(f"    WARN: cannot open future: {e}"); return

    pr_futr_ann = annual_mean(pr_futr_raw)
    pr_futr     = sel_years(pr_futr_ann, HIST_Y0, HIST_Y1)
    if len(pr_futr.time) < MIN_FUTR_YRS:
        print(f"    WARN: only {len(pr_futr.time)} future years -> skip"); return
    print(f"    futr years: {len(pr_futr.time)}")

    try:
        regridder = make_regridder(pr_hist.isel(time=0))
    except Exception as e:
        print(f"    WARN: cannot build regridder: {e}"); return

    clim_native = pr_hist.mean("time")
    clim_rg     = regridder(clim_native)

    try:
        tr_native = linear_trend_field(pr_hist)
        tr_rg     = regridder(tr_native)
        gm_trend  = global_mean(tr_rg)
        if abs(gm_trend) > 1e-6:
            tr_rg = tr_rg / gm_trend
        else:
            print(f"    WARN: global mean trend ~ 0, saving unnormalized")
        trend_ok = True
    except Exception as e:
        print(f"    WARN: trend failed: {e}")
        trend_ok = False

    futr_native = pr_futr.mean("time")
    futr_rg     = regridder(futr_native)
    dP          = futr_rg - clim_rg
    gm          = global_mean(dP)
    if abs(gm) < 1e-6:
        print(f"    WARN: global mean dP ~ 0 -> skipping dPdP")
        dpdp_ok = False
    else:
        dPdP    = dP / gm
        dpdp_ok = True

    out_dir.mkdir(parents=True, exist_ok=True)
    family = model_family(model_name)

    ds_clim = xr.Dataset(
        {"pr_clim": clim_rg},
        attrs=dict(model=model_name, family=family,
                   period=f"{HIST_Y0}-{HIST_Y1}", units="mm/yr",
                   longitude_range="-180 to 180")
    )
    safe_zarr_write(ds_clim, Path(str(prefix) + "_clim.zarr"))

    if trend_ok:
        ds_trend = xr.Dataset(
            {"pr_trend": tr_rg},
            attrs=dict(model=model_name, family=family,
                       period=f"{HIST_Y0}-{HIST_Y1}",
                       units="dimensionless (normalized by global mean trend)",
                       longitude_range="-180 to 180")
        )
        safe_zarr_write(ds_trend, Path(str(prefix) + "_trend.zarr"))

    if dpdp_ok:
        ds_dpdp = xr.Dataset(
            {"dPdP": dPdP},
            attrs=dict(model=model_name, family=family,
                       hist_period=f"{HIST_Y0}-{HIST_Y1}",
                       futr_period=f"{HIST_Y0}-{HIST_Y1} +4K",
                       units="dimensionless",
                       longitude_range="-180 to 180")
        )
        safe_zarr_write(ds_dpdp, Path(str(prefix) + "_dPdP.zarr"))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = find_file_pairs()
    print(f"Found {len(pairs)} model pairs\n")
    for model_name, hist_path, futr_path in pairs:
        process_model(model_name, hist_path, futr_path, OUT_DIR)
    print("\n\nDone.")


if __name__ == "__main__":
    main()