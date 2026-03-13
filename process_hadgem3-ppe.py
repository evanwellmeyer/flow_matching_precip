"""
process_hadgem_ppe.py
=====================
Processes HadGEM3-PPE perturbed-physics ensemble files into three netCDF
outputs per member:

  {out_dir}/HadGEM3-PPE_{NN}_clim.nc   — 1980-2014 climatology (lat, lon)
  {out_dir}/HadGEM3-PPE_{NN}_trend.nc  — linear trend 1980-2014,
                                          normalized by global mean
  {out_dir}/HadGEM3-PPE_{NN}_dPdP.nc   — forced change 2066-2100,
                                          dP / global_mean(dP)

TARGET GRID
-----------
Read directly from an existing PPE file.  Longitude convention -180 to 180.
"""

import re
from pathlib import Path

import numpy as np
import xarray as xr
import xesmf as xe

# ── configuration ──────────────────────────────────────────────────────────────
IN_DIR           = Path("/bsoden3/ewellmeyer/CMIP6/by_model/HadGEM3-PPE")
OUT_DIR          = Path("/bsoden3/ewellmeyer/CMIP6/processed_by_model")
CONV             = 365.25
HIST_Y0, HIST_Y1 = 1980, 2014
FUTR_Y0, FUTR_Y1 = 2065, 2100
MIN_HIST_YRS     = 10
MIN_FUTR_YRS     = 10

# ── target grid: read from PPE file ───────────────────────────────────────────
_PPE_GRID_CANDIDATES = [
    Path("/bsoden3/ewellmeyer/CMIP6/GA789_PR_his_rg128.nc"),
    Path("/bsoden3/ewellmeyer/CMIP6/CESM2_PR_his_rg128.nc"),
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


# ── helpers ────────────────────────────────────────────────────────────────────

def member_id_from_filename(fname):
    m = re.search(r'_(\d{2})_mon_', fname)
    return m.group(1) if m else None


def open_pr(path):
    ds = xr.open_dataset(path, use_cftime=True)
    pr = ds["pr"]
    if "ensemble_member" in pr.dims:
        pr = pr.squeeze("ensemble_member", drop=True)
    pr = pr * CONV
    pr.attrs["units"] = "mm/yr"
    renames = {}
    if "latitude"  in pr.dims: renames["latitude"]  = "lat"
    if "longitude" in pr.dims: renames["longitude"] = "lon"
    if renames:
        pr = pr.rename(renames)
    lon = pr.coords["lon"].values
    if lon.max() > 180:
        pr = pr.assign_coords(lon=((pr.coords["lon"] + 180) % 360) - 180)
        pr = pr.sortby("lon")
    pr = pr.sortby("time")
    _, idx = np.unique(pr.time.values, return_index=True)
    if len(idx) < len(pr.time):
        print(f"  INFO: dropped {len(pr.time) - len(idx)} duplicate time steps")
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
    ds_out = xr.Dataset({"lat": ("lat", TARGET_LAT),
                         "lon": ("lon", TARGET_LON)})
    return xe.Regridder(da_in, ds_out, method="bilinear",
                        periodic=True, reuse_weights=False)


def safe_nc_write(ds, path):
    if path.exists():
        path.unlink()
    ds.to_netcdf(path)
    print(f"  -> wrote {path.name}")


# ── per-member processing ──────────────────────────────────────────────────────

def process_member(path, out_dir):
    mid = member_id_from_filename(path.name)
    if mid is None:
        print(f"  WARN: cannot parse member id from {path.name}, skipping")
        return

    tag = f"HadGEM3-PPE_{mid}"
    prefix = out_dir / tag

    expected = [Path(str(prefix) + s) for s in ("_clim.nc", "_trend.nc", "_dPdP.nc")]
    if all(p.exists() for p in expected):
        print(f"  [{tag}] SKIP: all outputs already exist")
        return

    print(f"\n  [{tag}]  {path.name}")

    try:
        pr = open_pr(path)
    except Exception as e:
        print(f"  WARN: cannot open {path.name}: {e}")
        return

    pr_ann = annual_mean(pr)

    pr_hist = sel_years(pr_ann, HIST_Y0, HIST_Y1)
    if len(pr_hist.time) < MIN_HIST_YRS:
        print(f"  WARN: only {len(pr_hist.time)} hist years -> skip")
        return
    pr_hist = pr_hist.assign_coords(time=pr_hist.time.dt.year.values)
    print(f"    hist years: {len(pr_hist.time)}")

    pr_futr = sel_years(pr_ann, FUTR_Y0, FUTR_Y1)
    if len(pr_futr.time) < MIN_FUTR_YRS:
        print(f"  WARN: only {len(pr_futr.time)} future years -> skip")
        return
    print(f"    futr years: {len(pr_futr.time)}")

    try:
        regridder = make_regridder(pr_hist.isel(time=0))
    except Exception as e:
        print(f"  WARN: cannot build regridder: {e}")
        return

    clim_native = pr_hist.mean("time")
    clim_rg     = regridder(clim_native)

    try:
        tr_native = linear_trend_field(pr_hist)
        tr_rg     = regridder(tr_native)
        gm_trend  = global_mean(tr_rg)
        if abs(gm_trend) > 1e-6:
            tr_rg = tr_rg / gm_trend
        else:
            print(f"  WARN: global mean trend ~ 0, saving unnormalized")
        trend_ok = True
    except Exception as e:
        print(f"  WARN: trend failed: {e}")
        trend_ok = False

    futr_native = pr_futr.mean("time")
    futr_rg     = regridder(futr_native)
    dP          = futr_rg - clim_rg
    gm          = global_mean(dP)
    if abs(gm) < 1e-6:
        print(f"  WARN: global mean dP ~ 0, skipping dPdP")
        dpdp_ok = False
    else:
        dPdP    = dP / gm
        dpdp_ok = True

    out_dir.mkdir(parents=True, exist_ok=True)

    ds_clim = xr.Dataset(
        {"pr_clim": clim_rg},
        attrs=dict(model="HadGEM3-PPE", member=mid,
                   period=f"{HIST_Y0}-{HIST_Y1}", units="mm/yr",
                   longitude_range="-180 to 180")
    )
    safe_nc_write(ds_clim, Path(str(prefix) + "_clim.nc"))

    if trend_ok:
        ds_trend = xr.Dataset(
            {"pr_trend": tr_rg},
            attrs=dict(model="HadGEM3-PPE", member=mid,
                       period=f"{HIST_Y0}-{HIST_Y1}",
                       units="dimensionless (normalized by global mean trend)",
                       longitude_range="-180 to 180")
        )
        safe_nc_write(ds_trend, Path(str(prefix) + "_trend.nc"))

    if dpdp_ok:
        ds_dpdp = xr.Dataset(
            {"dPdP": dPdP},
            attrs=dict(model="HadGEM3-PPE", member=mid,
                       hist_period=f"{HIST_Y0}-{HIST_Y1}",
                       futr_period=f"{FUTR_Y0}-{FUTR_Y1}",
                       units="dimensionless",
                       longitude_range="-180 to 180")
        )
        safe_nc_write(ds_dpdp, Path(str(prefix) + "_dPdP.nc"))


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(IN_DIR.glob("pr_rcp85_land-gcm_global_60km_*_mon_*.nc"))
    if not files:
        print(f"No files found in {IN_DIR}")
        return
    print(f"Found {len(files)} HadGEM3-PPE member files\n")
    for f in files:
        process_member(f, OUT_DIR)
    print("\n\nDone.")


if __name__ == "__main__":
    main()