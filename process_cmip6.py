"""
process_cmip6_by_model.py
=========================
Processes CMIP6 pr files organized in /bsoden3/ewellmeyer/CMIP6/by_model/
into three netCDF outputs per physics group per model:

  {out_dir}/{MODEL}_{pN}_clim.nc   — 1980-2014 climatology (lat, lon)
  {out_dir}/{MODEL}_{pN}_trend.nc  — per-member linear trend 1980-2014
                                      normalized (member, lat, lon)
  {out_dir}/{MODEL}_{pN}_dPdP.nc   — forced change target 2066-2100
                                      ensemble-mean dP / global_mean(dP)
                                      (lat, lon)

TARGET GRID
-----------
Read directly from an existing PPE file so all outputs share the same
Gaussian grid.  Longitude convention is -180 to 180.
"""

import re
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
CONV             = 86400.0 * 365.25
HIST_Y0, HIST_Y1 = 1980, 2014
FUTR_Y0, FUTR_Y1 = 2066, 2100
SKIP_MODELS      = {"HadGEM3-PPE", "CESM2"}
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
    ds = xr.open_dataset(path, use_cftime=True)
    pr = ds["pr"] * CONV
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
        n_dropped = len(pr.time) - len(idx)
        print(f"      INFO: dropped {n_dropped} duplicate time steps in {path.name}")
        pr = pr.isel(time=idx)
    return pr


def make_regridder(da_in):
    ds_out = xr.Dataset({"lat": ("lat", TARGET_LAT),
                         "lon": ("lon", TARGET_LON)})
    return xe.Regridder(da_in, ds_out, method="bilinear",
                        periodic=True, reuse_weights=False)


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


def safe_nc_write(ds, path):
    if path.exists():
        path.unlink()
    ds.to_netcdf(path)
    print(f"    -> wrote {path.name}")


# ── directory scanning ─────────────────────────────────────────────────────────

def scan_model_dir(model_dir):
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
    return pr_sel.mean("time")


# ── main processing function ───────────────────────────────────────────────────

def process_physics_group(model_name, p_key, hist_files, futr_files,
                           out_dir, is_only_group):

    tag = model_name if is_only_group else f"{model_name}_p{p_key}"
    print(f"\n  [{tag}]  hist={len(hist_files)}  futr={len(futr_files)}")

    if not hist_files:
        print("    SKIP: no hist files"); return
    if not futr_files:
        print("    SKIP: no future files"); return

    prefix = out_dir / tag
    expected = [Path(str(prefix) + s) for s in ("_clim.nc", "_trend.nc", "_dPdP.nc")]
    if all(p.exists() for p in expected):
        print("    SKIP: all outputs already exist"); return

    hist_by_member = {}
    for f in hist_files:
        result = load_hist_member_native(f)
        if result is not None:
            mid, da = result
            hist_by_member[mid] = da

    if not hist_by_member:
        print("    SKIP: no valid hist members"); return

    common_years = sorted(
        set.intersection(*[set(da.time.values) for da in hist_by_member.values()])
    )
    if len(common_years) < MIN_HIST_YRS:
        print(f"    SKIP: only {len(common_years)} common hist years"); return

    member_ids = sorted(hist_by_member.keys())
    print(f"    hist members: {len(member_ids)},  common years: {len(common_years)}")

    regridder_cache = {}

    def get_regridder(da):
        shape = (da.sizes["lat"], da.sizes["lon"])
        if shape not in regridder_cache:
            regridder_cache[shape] = make_regridder(da)
            if len(regridder_cache) > 1:
                print(f"      INFO: built additional regridder for shape {shape}")
        return regridder_cache[shape]

    clim_rg_list = []
    for m in member_ids:
        tm = hist_by_member[m].sel(time=list(common_years)).mean("time")
        try:
            clim_rg_list.append(get_regridder(tm)(tm))
        except Exception as e:
            print(f"      WARN: clim regrid failed for {m}: {e}")

    if not clim_rg_list:
        print("    SKIP: no members could be regridded"); return

    clim_rg = xr.concat(clim_rg_list, dim="member").mean("member")

    trend_rg_list       = []
    valid_trend_members = []
    for m in member_ids:
        da_m = hist_by_member[m].sel(time=list(common_years))
        try:
            tr_native = linear_trend_field(da_m)
            tr_rg     = get_regridder(tr_native)(tr_native)
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
        )
    else:
        print("    WARN: no valid trends computed")
        trend_stack = None

    futr_rg_list = []
    for f in futr_files:
        fm = load_futr_member_native(f)
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
        dP = futr_ens_rg - clim_rg
        gm = global_mean(dP)
        if abs(gm) < 1e-6:
            print("    WARN: global mean of dP ~ 0, cannot normalize -> skipping dPdP")
            dPdP = None
        else:
            dPdP = dP / gm

    out_dir.mkdir(parents=True, exist_ok=True)

    ds_clim = xr.Dataset(
        {"pr_clim": clim_rg},
        attrs=dict(model=model_name, physics=p_key,
                   period=f"{HIST_Y0}-{HIST_Y1}", units="mm/yr",
                   n_members=len(member_ids),
                   longitude_range="-180 to 180")
    )
    safe_nc_write(ds_clim, Path(str(prefix) + "_clim.nc"))

    if trend_stack is not None:
        ds_trend = xr.Dataset(
            {"pr_trend": trend_stack},
            attrs=dict(model=model_name, physics=p_key,
                       period=f"{HIST_Y0}-{HIST_Y1}",
                       units="dimensionless (normalized by global mean trend)",
                       members=",".join(valid_trend_members),
                       longitude_range="-180 to 180")
        )
        safe_nc_write(ds_trend, Path(str(prefix) + "_trend.nc"))

    if dPdP is not None:
        ds_dpdp = xr.Dataset(
            {"dPdP": dPdP},
            attrs=dict(model=model_name, physics=p_key,
                       hist_period=f"{HIST_Y0}-{HIST_Y1}",
                       futr_period=f"{FUTR_Y0}-{FUTR_Y1}",
                       units="dimensionless",
                       n_futr_members=len(futr_rg_list),
                       longitude_range="-180 to 180")
        )
        safe_nc_write(ds_dpdp, Path(str(prefix) + "_dPdP.nc"))


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