"""
process_cesm2_lens2.py
======================
Processes CESM2 Large Ensemble v2 (LENS2) zarr files into three outputs
per forcing group (cmip6, smbb), following the same conventions as
process_cmip6_by_model.py:

  {out_dir}/CESM2-LENS2_cmip6_clim.zarr   — 1980-2014 climatology (lat, lon)
  {out_dir}/CESM2-LENS2_cmip6_trend.zarr  — per-member trend 1980-2014,
                                             normalized by global mean
                                             (member, lat, lon)
  {out_dir}/CESM2-LENS2_cmip6_dPdP.zarr  — ensemble-mean dP / global_mean(dP)
                                             2066-2100 vs 1980-2014
                                             (lat, lon)
  ... and same three for smbb

Input files (all in IN_DIR)
---------------------------
  cesm2LE-historical-{cmip6,smbb}-{PRECC,PRECL}.zarr
  cesm2LE-ssp370-{cmip6,smbb}-{PRECC,PRECL}.zarr

Variable notes
--------------
  PR = PRECC + PRECL  (convective + large-scale)
  Units: m/s  ->  mm/yr via * 365.25 * 86400 * 1000
  Calendar: noleap  ->  use_cftime=True when opening

Regridding strategy
-------------------
Same as process_cmip6_by_model.py:
  - Clim  : member time-means averaged on native grid -> regrid once
  - dPdP  : future member time-means averaged on native grid -> regrid once,
             then dP computed and normalized on 128x192
  - Trend : per-member 2D trend field computed on native grid -> regrid each
            (unavoidable since we want per-member output; 2D field is cheap)
  Both cmip6 and smbb are treated as separate physics groups.
"""

import shutil
from pathlib import Path

import numpy as np
import xarray as xr
import xesmf as xe

# ── configuration ──────────────────────────────────────────────────────────────
IN_DIR           = Path("/Users/ewellmeyer/Documents/research/CMIP6/data_by_model/CESM2")
OUT_DIR          = Path("/Users/ewellmeyer/Documents/research/CMIP6/processed_by_model")
CONV             = 365.25 * 86400 * 1000   # m/s -> mm/yr
HIST_Y0, HIST_Y1 = 1980, 2014
FUTR_Y0, FUTR_Y1 = 2066, 2100
MIN_YRS          = 10

# target regrid grid
N_LAT, N_LON = 128, 192
TARGET_LAT = np.linspace(-90 + 180/N_LAT/2,  90 - 180/N_LAT/2, N_LAT)
TARGET_LON = np.linspace(  0 + 360/N_LON/2, 360 - 360/N_LON/2, N_LON)

FORCINGS = ["cmip6", "smbb"]


# ── helpers ────────────────────────────────────────────────────────────────────

def open_pr_zarr(precc_path, precl_path):
    """
    Open PRECC and PRECL zarr stores, add them, convert to mm/yr.
    Returns DataArray (member, time, lat, lon).
    Uses .values to bypass xarray alignment issues with duplicate time coords.
    """
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    ds_c = xr.open_zarr(precc_path, decode_times=time_coder)
    ds_l = xr.open_zarr(precl_path, decode_times=time_coder)

    # get lat/lon coord names
    lat_name = "lat" if "lat" in ds_c.coords else "latitude"
    lon_name = "lon" if "lon" in ds_c.coords else "longitude"

    # add using .values to avoid xarray alignment on duplicate time index
    pr_vals = (ds_c["PRECC"].values + ds_l["PRECL"].values) * CONV
    pr = xr.DataArray(
        pr_vals,
        dims=ds_c["PRECC"].dims,
        coords=ds_c["PRECC"].coords,
        attrs={"units": "mm/yr"},
    )

    # standardize coord names
    renames = {}
    if lat_name != "lat": renames[lat_name] = "lat"
    if lon_name != "lon": renames[lon_name] = "lon"
    if renames:
        pr = pr.rename(renames)

    # standardize lon to [0, 360)
    if float(pr.coords["lon"].min()) < 0:
        pr = pr.assign_coords(lon=(pr.coords["lon"] % 360))
        pr = pr.sortby("lon")

    # sort time and drop duplicates
    pr = pr.sortby("time")
    _, idx = np.unique(pr.time.values, return_index=True)
    if len(idx) < pr.sizes["time"]:
        print(f"    INFO: dropped {pr.sizes['time'] - len(idx)} duplicate time steps")
        pr = pr.isel(time=idx)

    # rename member_id dim to member for consistency
    if 'member_id' in pr.dims:
        pr = pr.rename({'member_id': 'member'})
    return pr   # (member, time, lat, lon)


def annual_mean(da):
    return da.resample(time="YS").mean("time")


def sel_years(da, y0, y1):
    yrs = da.time.dt.year
    return da.isel(time=((yrs >= y0) & (yrs <= y1)))


def global_mean(da):
    weights = np.cos(np.deg2rad(da.lat))
    return float(da.weighted(weights).mean(("lat", "lon")))


def linear_trend_field(da):
    """Per-pixel least-squares trend over time. da: (time, lat, lon)."""
    t = np.arange(len(da.time), dtype=float)
    da = da.assign_coords(time=t)
    p = da.polyfit(dim="time", deg=1)
    return p["polyfit_coefficients"].sel(degree=1).drop_vars("degree")


def make_regridder(da_in):
    ds_out = xr.Dataset({"lat": TARGET_LAT, "lon": TARGET_LON})
    return xe.Regridder(da_in, ds_out, method="bilinear",
                        periodic=True, reuse_weights=False)


def safe_zarr_write(ds, path):
    if path.exists():
        shutil.rmtree(path)
    ds.to_zarr(path, mode="w")
    print(f"  -> wrote {path.name}")


# ── per-forcing processing ─────────────────────────────────────────────────────

def process_forcing(forcing, out_dir):
    tag = f"CESM2-LENS2_{forcing}"
    prefix = out_dir / tag

    expected = [Path(str(prefix) + s)
                for s in ("_clim.zarr", "_trend.zarr", "_dPdP.zarr")]
    if all(p.exists() for p in expected):
        print(f"\n[{tag}] SKIP: all outputs already exist")
        return

    print(f"\n{'='*60}\n[{tag}]")

    # ── locate input files ────────────────────────────────────────────────────
    hist_precc = IN_DIR / f"cesm2LE-historical-{forcing}-PRECC.zarr"
    hist_precl = IN_DIR / f"cesm2LE-historical-{forcing}-PRECL.zarr"
    futr_precc = IN_DIR / f"cesm2LE-ssp370-{forcing}-PRECC.zarr"
    futr_precl = IN_DIR / f"cesm2LE-ssp370-{forcing}-PRECL.zarr"

    for p in [hist_precc, hist_precl, futr_precc, futr_precl]:
        if not p.exists():
            print(f"  SKIP: missing {p.name}")
            return

    # ── load and select years ─────────────────────────────────────────────────
    print("  loading historical...")
    pr_hist_full = open_pr_zarr(hist_precc, hist_precl)
    pr_hist_ann  = annual_mean(pr_hist_full)
    pr_hist      = sel_years(pr_hist_ann, HIST_Y0, HIST_Y1)
    if pr_hist.sizes["time"] < MIN_YRS:
        print(f"  SKIP: only {pr_hist.sizes['time']} hist years"); return
    pr_hist = pr_hist.assign_coords(time=pr_hist.time.dt.year.values)
    n_members = pr_hist.sizes["member"]
    print(f"  hist: {n_members} members, {pr_hist.sizes['time']} years")

    print("  loading future...")
    pr_futr_full = open_pr_zarr(futr_precc, futr_precl)
    pr_futr_ann  = annual_mean(pr_futr_full)
    pr_futr      = sel_years(pr_futr_ann, FUTR_Y0, FUTR_Y1)
    if pr_futr.sizes["time"] < MIN_YRS:
        print(f"  SKIP: only {pr_futr.sizes['time']} future years"); return
    print(f"  futr: {pr_futr.sizes['member']} members, {pr_futr.sizes['time']} years")

    # ── build regridder from first member ────────────────────────────────────
    sample = pr_hist.isel(member=0, time=0)
    try:
        regridder = make_regridder(sample)
    except Exception as e:
        print(f"  SKIP: cannot build regridder: {e}"); return

    # ── climatology: member time-means averaged on native grid -> regrid once ─
    print("  computing climatology...")
    clim_native = pr_hist.mean("time").mean("member")   # (lat, lon)
    clim_rg     = regridder(clim_native)

    # ── trend: per-member on native grid -> regrid 2D field -> normalize ──────
    print("  computing trends...")
    trend_rg_list   = []
    valid_member_ids = []
    member_coords = (pr_hist.coords["member"].values
                     if "member" in pr_hist.coords
                     else np.arange(n_members))

    for i in range(n_members):
        da_m = pr_hist.isel(member=i)    # (time, lat, lon)
        try:
            tr_native = linear_trend_field(da_m)
            tr_rg     = regridder(tr_native)
            gm_trend  = global_mean(tr_rg)
            if abs(gm_trend) > 1e-6:
                tr_rg = tr_rg / gm_trend
            else:
                print(f"    WARN: member {i} global mean trend ~ 0, unnormalized")
            trend_rg_list.append(tr_rg.values)
            valid_member_ids.append(str(member_coords[i]))
        except Exception as e:
            print(f"    WARN: trend failed for member {i}: {e}")

    if not trend_rg_list:
        print("  WARN: no valid trends computed")
        trend_stack = None
    else:
        trend_arr = np.stack(trend_rg_list, axis=0)   # (member, lat, lon)
        trend_stack = xr.DataArray(
            trend_arr,
            dims=["member", "lat", "lon"],
            coords={"member": valid_member_ids,
                    "lat": TARGET_LAT, "lon": TARGET_LON},
        )

    # ── dPdP: future member time-means averaged on native grid -> regrid once ─
    print("  computing dPdP...")
    futr_native = pr_futr.mean("time").mean("member")   # (lat, lon)
    futr_rg     = regridder(futr_native)

    dP = futr_rg - clim_rg
    gm = global_mean(dP)
    if abs(gm) < 1e-6:
        print("  WARN: global mean dP ~ 0, skipping dPdP")
        dPdP = None
    else:
        dPdP = dP / gm

    # ── write outputs ─────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_clim = xr.Dataset(
        {"pr_clim": clim_rg},
        attrs=dict(model="CESM2-LENS2", forcing=forcing,
                   period=f"{HIST_Y0}-{HIST_Y1}", units="mm/yr",
                   n_members=n_members)
    )
    safe_zarr_write(ds_clim, Path(str(prefix) + "_clim.zarr"))

    if trend_stack is not None:
        ds_trend = xr.Dataset(
            {"pr_trend": trend_stack},
            attrs=dict(model="CESM2-LENS2", forcing=forcing,
                       period=f"{HIST_Y0}-{HIST_Y1}",
                       units="dimensionless (normalized by global mean trend)",
                       n_members=len(valid_member_ids))
        )
        safe_zarr_write(ds_trend, Path(str(prefix) + "_trend.zarr"))

    if dPdP is not None:
        ds_dpdp = xr.Dataset(
            {"dPdP": dPdP},
            attrs=dict(model="CESM2-LENS2", forcing=forcing,
                       hist_period=f"{HIST_Y0}-{HIST_Y1}",
                       futr_period=f"{FUTR_Y0}-{FUTR_Y1}",
                       units="dimensionless",
                       n_members=n_members)
        )
        safe_zarr_write(ds_dpdp, Path(str(prefix) + "_dPdP.zarr"))


# ── main ───────────────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)

for forcing in FORCINGS:
    process_forcing(forcing, OUT_DIR)

print("\nDone.")