"""
process_cesm2_lens2.py
======================
Processes CESM2 Large Ensemble v2 (LENS2) zarr files into three netCDF
outputs per forcing group (cmip6, smbb):

  {out_dir}/CESM2-LENS2_cmip6_clim.nc   — 1980-2014 climatology (lat, lon)
  {out_dir}/CESM2-LENS2_cmip6_trend.nc  — per-member trend (member, lat, lon)
  {out_dir}/CESM2-LENS2_cmip6_dPdP.nc   — ensemble-mean dP / global_mean(dP)
  ... and same three for smbb

Input files are zarr stores (read with xr.open_zarr).
Output files are netCDF (read with xr.open_dataset).

TARGET GRID
-----------
Read directly from an existing PPE file.  Longitude convention -180 to 180.
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

# ── target grid: read from PPE file ───────────────────────────────────────────
_PPE_GRID_CANDIDATES = [
    Path("/Users/ewellmeyer/Documents/research/HadGEM/GA789_PR_his_rg128.nc"),
    Path("/Users/ewellmeyer/Documents/research/CESM2/CESM2_PR_his_rg128.nc"),
]


def _load_target_grid():
    for candidate in _PPE_GRID_CANDIDATES:
        if candidate.exists():
            ds = xr.open_dataset(candidate)
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

FORCINGS = ["cmip6", "smbb"]


# ── helpers ────────────────────────────────────────────────────────────────────

def open_pr_zarr(precc_path, precl_path):
    """
    Open PRECC and PRECL zarr stores, add them, convert to mm/yr.
    Returns DataArray (member, time, lat, lon).
    """
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    ds_c = xr.open_zarr(precc_path, decode_times=time_coder)
    ds_l = xr.open_zarr(precl_path, decode_times=time_coder)

    lat_name = "lat" if "lat" in ds_c.coords else "latitude"
    lon_name = "lon" if "lon" in ds_c.coords else "longitude"

    pr_vals = (ds_c["PRECC"].values + ds_l["PRECL"].values) * CONV
    pr = xr.DataArray(
        pr_vals,
        dims=ds_c["PRECC"].dims,
        coords=ds_c["PRECC"].coords,
        attrs={"units": "mm/yr"},
    )

    renames = {}
    if lat_name != "lat": renames[lat_name] = "lat"
    if lon_name != "lon": renames[lon_name] = "lon"
    if renames:
        pr = pr.rename(renames)

    lon = pr.coords["lon"].values
    if lon.max() > 180:
        pr = pr.assign_coords(lon=((pr.coords["lon"] + 180) % 360) - 180)
        pr = pr.sortby("lon")

    pr = pr.sortby("time")
    _, idx = np.unique(pr.time.values, return_index=True)
    if len(idx) < pr.sizes["time"]:
        print(f"    INFO: dropped {pr.sizes['time'] - len(idx)} duplicate time steps")
        pr = pr.isel(time=idx)

    if 'member_id' in pr.dims:
        pr = pr.rename({'member_id': 'member'})
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
    ds_out = xr.Dataset({"lat": ("lat", TARGET_LAT),
                         "lon": ("lon", TARGET_LON)})
    return xe.Regridder(da_in, ds_out, method="bilinear",
                        periodic=True, reuse_weights=False)


def safe_nc_write(ds, path):
    if path.exists():
        path.unlink()
    ds.to_netcdf(path)
    print(f"  -> wrote {path.name}")


# ── per-forcing processing ─────────────────────────────────────────────────────

def process_forcing(forcing, out_dir):
    tag = f"CESM2-LENS2_{forcing}"
    prefix = out_dir / tag

    expected = [Path(str(prefix) + s)
                for s in ("_clim.nc", "_trend.nc", "_dPdP.nc")]
    if all(p.exists() for p in expected):
        print(f"\n[{tag}] SKIP: all outputs already exist")
        return

    print(f"\n{'='*60}\n[{tag}]")

    hist_precc = IN_DIR / f"cesm2LE-historical-{forcing}-PRECC.zarr"
    hist_precl = IN_DIR / f"cesm2LE-historical-{forcing}-PRECL.zarr"
    futr_precc = IN_DIR / f"cesm2LE-ssp370-{forcing}-PRECC.zarr"
    futr_precl = IN_DIR / f"cesm2LE-ssp370-{forcing}-PRECL.zarr"

    for p in [hist_precc, hist_precl, futr_precc, futr_precl]:
        if not p.exists():
            print(f"  SKIP: missing {p.name}")
            return

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

    sample = pr_hist.isel(member=0, time=0)
    try:
        regridder = make_regridder(sample)
    except Exception as e:
        print(f"  SKIP: cannot build regridder: {e}"); return

    print("  computing climatology...")
    clim_native = pr_hist.mean("time").mean("member")
    clim_rg     = regridder(clim_native)

    print("  computing trends...")
    trend_rg_list    = []
    valid_member_ids = []
    member_coords = (pr_hist.coords["member"].values
                     if "member" in pr_hist.coords
                     else np.arange(n_members))

    for i in range(n_members):
        da_m = pr_hist.isel(member=i)
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
        trend_arr = np.stack(trend_rg_list, axis=0)
        trend_stack = xr.DataArray(
            trend_arr,
            dims=["member", "lat", "lon"],
            coords={"member": valid_member_ids,
                    "lat": TARGET_LAT, "lon": TARGET_LON},
        )

    print("  computing dPdP...")
    futr_native = pr_futr.mean("time").mean("member")
    futr_rg     = regridder(futr_native)

    dP = futr_rg - clim_rg
    gm = global_mean(dP)
    if abs(gm) < 1e-6:
        print("  WARN: global mean dP ~ 0, skipping dPdP")
        dPdP = None
    else:
        dPdP = dP / gm

    out_dir.mkdir(parents=True, exist_ok=True)

    ds_clim = xr.Dataset(
        {"pr_clim": clim_rg},
        attrs=dict(model="CESM2-LENS2", forcing=forcing,
                   period=f"{HIST_Y0}-{HIST_Y1}", units="mm/yr",
                   n_members=n_members,
                   longitude_range="-180 to 180")
    )
    safe_nc_write(ds_clim, Path(str(prefix) + "_clim.nc"))

    if trend_stack is not None:
        ds_trend = xr.Dataset(
            {"pr_trend": trend_stack},
            attrs=dict(model="CESM2-LENS2", forcing=forcing,
                       period=f"{HIST_Y0}-{HIST_Y1}",
                       units="dimensionless (normalized by global mean trend)",
                       n_members=len(valid_member_ids),
                       longitude_range="-180 to 180")
        )
        safe_nc_write(ds_trend, Path(str(prefix) + "_trend.nc"))

    if dPdP is not None:
        ds_dpdp = xr.Dataset(
            {"dPdP": dPdP},
            attrs=dict(model="CESM2-LENS2", forcing=forcing,
                       hist_period=f"{HIST_Y0}-{HIST_Y1}",
                       futr_period=f"{FUTR_Y0}-{FUTR_Y1}",
                       units="dimensionless",
                       n_members=n_members,
                       longitude_range="-180 to 180")
        )
        safe_nc_write(ds_dpdp, Path(str(prefix) + "_dPdP.nc"))


# ── main ───────────────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)

for forcing in FORCINGS:
    process_forcing(forcing, OUT_DIR)

print("\nDone.")