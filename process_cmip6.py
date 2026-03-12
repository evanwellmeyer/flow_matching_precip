"""
process_cmip6_by_model.py
=========================
Processes CMIP6 pr files organized in /bsoden3/ewellmeyer/CMIP6/by_model/
into three zarr outputs per physics group per model:

  {out_dir}/{MODEL}_{pN}_clim.zarr   — 1980-2014 climatology (lat, lon)
  {out_dir}/{MODEL}_{pN}_trend.zarr  — per-member linear trend 1980-2014
                                       mm/yr per yr  (member, lat, lon)
  {out_dir}/{MODEL}_{pN}_dPdP.zarr   — forced change target 2066-2100
                                       ensemble-mean dP / global_mean(dP)
                                       (lat, lon)

Regridding strategy (for speed)
---------------------------------
Regridding is expensive so we minimize the number of regrid calls:

  Climatology : average all member time-means on native grid → regrid once
  dPdP target : average future member time-means on native grid → regrid once
                compute dP on 128x192, then divide by global_mean(dP)
  Trend       : unavoidably per-member (we want member-level output),
                but we regrid each member's trend field (lat x lon) rather
                than the full (time, lat, lon) cube — input is tiny.

Physics grouping rules
-----------------------
* The 'p' number in the variant label (rNiNpNfN) determines the group.
* Different 'f' values within the same 'p' group are pooled as separate
  realizations (f treated like a different seed, not different physics).
* Models that store physics variants in separate subdirectories
  (original_hist_p1, original_hist_p2, ...) are handled automatically.
* Models with a single 'original_hist' folder are one group; physics number
  inferred from filenames.

Special cases
-------------
* HadGEM3-PPE     - non-standard filenames -> skipped.
* GISS-E2-1-G-CC  - uses 'esm-ssp585'; treated as ssp5.
* Files not covering 1980-2014 excluded per member with warning.
* Future files not reaching 2066 skipped with warning.
"""

import re
import shutil
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import xarray as xr
import xesmf as xe

warnings.filterwarnings("ignore", category=FutureWarning)

# ── configuration ──────────────────────────────────────────────────────────────
BASE_DIR         = Path("/bsoden3/ewellmeyer/CMIP6/by_model")
OUT_DIR          = Path("/bsoden3/ewellmeyer/CMIP6/processed_by_model")
CONV             = 86400.0 * 365.25        # kg/m2/s -> mm/yr
HIST_Y0, HIST_Y1 = 1980, 2014
FUTR_Y0, FUTR_Y1 = 2066, 2100
SKIP_MODELS      = {"HadGEM3-PPE", "CESM2"}
MIN_HIST_YRS     = 10
MIN_FUTR_YRS     = 10

# target regrid grid
N_LAT, N_LON = 128, 192
TARGET_LAT = np.linspace(-90 + 180/N_LAT/2,  90 - 180/N_LAT/2, N_LAT)
TARGET_LON = np.linspace(  0 + 360/N_LON/2, 360 - 360/N_LON/2, N_LON)


# ── helpers ────────────────────────────────────────────────────────────────────

def parse_variant(fname):
    m = re.search(r'_r(\d+)i(\d+)p(\d+)f(\d+)_', fname)
    return tuple(int(x) for x in m.groups()) if m else None


def physics_from_variant(fname):
    v = parse_variant(fname)
    return v[2] if v else None


def member_id_from_variant(fname):
    m = re.search(r'(r\d+i\d+p\d+f\d+)', fname)
    return m.group(1) if m else None


def open_pr(path):
    """Open NetCDF, return pr DataArray in mm/yr with standardized coords."""
    ds = xr.open_dataset(path, use_cftime=True)
    pr = ds["pr"] * CONV
    pr.attrs["units"] = "mm/yr"
    # standardize coord names
    renames = {}
    if "latitude"  in pr.dims: renames["latitude"]  = "lat"
    if "longitude" in pr.dims: renames["longitude"] = "lon"
    if renames:
        pr = pr.rename(renames)
    # standardize lon to [0, 360)
    if float(pr.coords["lon"].min()) < 0:
        pr = pr.assign_coords(lon=(pr.coords["lon"] % 360))
        pr = pr.sortby("lon")
    # ensure time is monotonic (some models have out-of-order or duplicate steps)
    pr = pr.sortby("time")
    _, idx = np.unique(pr.time.values, return_index=True)
    if len(idx) < len(pr.time):
        n_dropped = len(pr.time) - len(idx)
        print(f"      INFO: dropped {n_dropped} duplicate time steps in {path.name}")
        pr = pr.isel(time=idx)
    return pr


def make_regridder(da_in):
    """Bilinear regridder from da_in grid to 128x192."""
    ds_out = xr.Dataset({"lat": TARGET_LAT, "lon": TARGET_LON})
    return xe.Regridder(da_in, ds_out, method="bilinear",
                        periodic=True, reuse_weights=False)


def annual_mean(da):
    return da.resample(time="YS").mean("time")


def sel_years(da, y0, y1):
    yrs = da.time.dt.year
    return da.isel(time=((yrs >= y0) & (yrs <= y1)))


def global_mean(da):
    """Area-weighted global mean of a (lat, lon) DataArray."""
    weights = np.cos(np.deg2rad(da.lat))
    return float(da.weighted(weights).mean(("lat", "lon")))


def linear_trend_field(da):
    """
    Per-pixel least-squares trend (mm/yr per yr) over time axis.
    da must be annual. Returns (lat, lon) array.
    """
    t = np.arange(len(da.time), dtype=float)
    da = da.assign_coords(time=t)
    p = da.polyfit(dim="time", deg=1)
    return p["polyfit_coefficients"].sel(degree=1).drop_vars("degree")


def safe_zarr_write(ds, path):
    if path.exists():
        shutil.rmtree(path)
    ds.to_zarr(path, mode="w")
    print(f"    -> wrote {path.name}")


# ── directory scanning ─────────────────────────────────────────────────────────

def scan_model_dir(model_dir):
    """
    Returns dict {p_int: {"hist_files": [...], "futr_files": [...]}}
    Physics key comes from subdirectory suffix (_p1, _p2 ...) or filename.
    """
    groups = defaultdict(lambda: {"hist_files": [], "futr_files": []})

    for subdir in sorted(model_dir.iterdir()):
        if not subdir.is_dir():
            continue
        name = subdir.name

        pm = re.search(r'_p(\d+)$', name)
        p_from_dir = int(pm.group(1)) if pm else None

        is_hist = "hist" in name
        is_futr = any(x in name for x in ("ssp", "rcp"))

        for f in sorted(subdir.glob("*.nc")):
            p_from_file = physics_from_variant(f.name)
            p_key = p_from_dir if p_from_dir is not None else (p_from_file or 1)
            if is_hist:
                groups[p_key]["hist_files"].append(f)
            elif is_futr:
                groups[p_key]["futr_files"].append(f)

    return groups


# ── native-grid loading ────────────────────────────────────────────────────────

def load_hist_member_native(path):
    """
    Load hist file, annual-mean, select 1980-2014, assign integer year coords.
    Returns (member_id, DataArray) or None.
    """
    mid = member_id_from_variant(path.name) or path.stem
    try:
        pr = open_pr(path)
    except Exception as e:
        print(f"      WARN: cannot open {path.name}: {e}")
        return None

    pr_ann = annual_mean(pr)
    pr_sel = sel_years(pr_ann, HIST_Y0, HIST_Y1)
    if len(pr_sel.time) < MIN_HIST_YRS:
        print(f"      WARN: {path.name} - only {len(pr_sel.time)} hist years -> skip")
        return None

    pr_sel = pr_sel.assign_coords(time=pr_sel.time.dt.year.values)
    return mid, pr_sel


def load_futr_member_native(path):
    """
    Load future file, annual-mean, select 2066-2100, return time-mean (lat,lon).
    Returns DataArray or None.
    """
    try:
        pr = open_pr(path)
    except Exception as e:
        print(f"      WARN: cannot open {path.name}: {e}")
        return None

    pr_ann = annual_mean(pr)
    pr_sel = sel_years(pr_ann, FUTR_Y0, FUTR_Y1)
    if len(pr_sel.time) < MIN_FUTR_YRS:
        print(f"      WARN: {path.name} - only {len(pr_sel.time)} future years -> skip")
        return None

    return pr_sel.mean("time")   # (lat, lon) on native grid


# ── main processing function ───────────────────────────────────────────────────

def process_physics_group(model_name, p_key, hist_files, futr_files,
                           out_dir, is_only_group):

    tag = model_name if is_only_group else f"{model_name}_p{p_key}"
    print(f"\n  [{tag}]  hist={len(hist_files)}  futr={len(futr_files)}")

    if not hist_files:
        print("    SKIP: no hist files"); return
    if not futr_files:
        print("    SKIP: no future files"); return

    # ── skip if all outputs already exist ────────────────────────────────────
    prefix = out_dir / tag
    expected = [Path(str(prefix) + s) for s in ("_clim.zarr", "_trend.zarr", "_dPdP.zarr")]
    if all(p.exists() for p in expected):
        print("    SKIP: all outputs already exist"); return

    # ── load all hist members on native grid ──────────────────────────────────
    hist_by_member = {}
    for f in hist_files:
        result = load_hist_member_native(f)
        if result is not None:
            mid, da = result
            hist_by_member[mid] = da

    if not hist_by_member:
        print("    SKIP: no valid hist members"); return

    # common year intersection
    common_years = sorted(
        set.intersection(*[set(da.time.values) for da in hist_by_member.values()])
    )
    if len(common_years) < MIN_HIST_YRS:
        print(f"    SKIP: only {len(common_years)} common hist years"); return

    member_ids = sorted(hist_by_member.keys())
    print(f"    hist members: {len(member_ids)},  common years: {len(common_years)}")

    # ── regridder cache keyed by native grid shape ───────────────────────────
    # Some models (e.g. EC-Earth3) mix grid resolutions across members.
    # We build one regridder per unique (nlat, nlon) shape and reuse it.
    regridder_cache = {}

    def get_regridder(da):
        shape = (da.sizes["lat"], da.sizes["lon"])
        if shape not in regridder_cache:
            regridder_cache[shape] = make_regridder(da)
            if len(regridder_cache) > 1:
                print(f"      INFO: built additional regridder for shape {shape}")
        return regridder_cache[shape]

    # ── climatology: regrid each member's time-mean (2D, cheap) then average ─
    # We regrid per-member here (not the ensemble mean) so that members on
    # different native grids are all correctly handled.
    clim_rg_list = []
    for m in member_ids:
        tm = hist_by_member[m].sel(time=list(common_years)).mean("time")
        try:
            clim_rg_list.append(get_regridder(tm)(tm))
        except Exception as e:
            print(f"      WARN: clim regrid failed for {m}: {e}")

    if not clim_rg_list:
        print("    SKIP: no members could be regridded"); return

    clim_rg = xr.concat(clim_rg_list, dim="member").mean("member")  # (lat, lon)

    # ── trend: per-member on native grid, regrid trend field (not the cube) ───
    # normalize each member's trend by its own global mean (same as dPdP)
    trend_rg_list       = []
    valid_trend_members = []
    for m in member_ids:
        da_m = hist_by_member[m].sel(time=list(common_years))
        try:
            tr_native = linear_trend_field(da_m)               # (lat, lon)
            tr_rg     = get_regridder(tr_native)(tr_native)    # (lat, lon) on 128x192
            gm_trend  = global_mean(tr_rg)
            if abs(gm_trend) > 1e-6:
                tr_rg = tr_rg / gm_trend
            else:
                print(f"      WARN: global mean trend ~ 0 for {m}, saving unnormalized")
            trend_rg_list.append(tr_rg)
            valid_trend_members.append(m)
        except Exception as e:
            print(f"      WARN: trend failed for {m}: {e}")

    if trend_rg_list:
        trend_stack = xr.concat(
            trend_rg_list,
            dim=xr.DataArray(valid_trend_members, dims=["member"])
        )   # (member, lat, lon)
    else:
        print("    WARN: no valid trends computed")
        trend_stack = None

    # ── future: regrid each member's time-mean then ensemble-average ─────────
    # Same per-member regrid approach as clim to handle mixed grids.
    futr_rg_list = []
    for f in futr_files:
        fm = load_futr_member_native(f)   # (lat, lon) on native grid or None
        if fm is not None:
            try:
                futr_rg_list.append(get_regridder(fm)(fm))
            except Exception as e:
                print(f"      WARN: futr regrid failed for {f.name}: {e}")

    if not futr_rg_list:
        print("    WARN: no valid future members -> skipping dPdP")
        dPdP = None
    else:
        print(f"    futr members used: {len(futr_rg_list)}")
        futr_ens_rg = xr.concat(futr_rg_list, dim="member").mean("member")

        # dP and normalization on regridded grid
        dP = futr_ens_rg - clim_rg
        gm = global_mean(dP)
        if abs(gm) < 1e-6:
            print("    WARN: global mean of dP ~ 0, cannot normalize -> skipping dPdP")
            dPdP = None
        else:
            dPdP = dP / gm

    # ── write outputs ─────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_clim = xr.Dataset(
        {"pr_clim": clim_rg},
        attrs=dict(model=model_name, physics=p_key,
                   period=f"{HIST_Y0}-{HIST_Y1}", units="mm/yr",
                   n_members=len(member_ids))
    )
    safe_zarr_write(ds_clim, Path(str(prefix) + "_clim.zarr"))

    if trend_stack is not None:
        ds_trend = xr.Dataset(
            {"pr_trend": trend_stack},
            attrs=dict(model=model_name, physics=p_key,
                       period=f"{HIST_Y0}-{HIST_Y1}", units="dimensionless (normalized by global mean trend)",
                       members=",".join(valid_trend_members))
        )
        safe_zarr_write(ds_trend, Path(str(prefix) + "_trend.zarr"))

    if dPdP is not None:
        ds_dpdp = xr.Dataset(
            {"dPdP": dPdP},
            attrs=dict(model=model_name, physics=p_key,
                       hist_period=f"{HIST_Y0}-{HIST_Y1}",
                       futr_period=f"{FUTR_Y0}-{FUTR_Y1}",
                       units="dimensionless",
                       n_futr_members=len(futr_rg_list))
        )
        safe_zarr_write(ds_dpdp, Path(str(prefix) + "_dPdP.zarr"))


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_dirs = sorted(d for d in BASE_DIR.iterdir() if d.is_dir())

    for model_dir in model_dirs:
        model_name = model_dir.name
        if model_name in SKIP_MODELS:
            print(f"\nSKIP: {model_name}")
            continue

        print(f"\n{'='*60}\nMODEL: {model_name}")
        groups = scan_model_dir(model_dir)
        if not groups:
            print("  no files found"); continue

        is_only_group = len(groups) == 1
        for p_key in sorted(groups.keys()):
            process_physics_group(
                model_name, p_key,
                groups[p_key]["hist_files"],
                groups[p_key]["futr_files"],
                OUT_DIR,
                is_only_group,
            )

    print("\n\nDone.")


if __name__ == "__main__":
    main()