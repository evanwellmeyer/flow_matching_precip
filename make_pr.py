"""
Compute the total precip pr = precc + precl from the cesm2 lens zarr files
- saves one zarr per expiriment
"""


import xarray as xr
from pathlib import Path
import numcodecs.blosc as blosc

blosc.use_threads = False 

base_dir = Path("/bsoden3/ewellmeyer/CMIP6/by_model/CESM2")
hist_dir = base_dir / "original_hist"
ssp3_dir = base_dir / "original_ssp3"

SECONDS_PER_YEAR = 365.25 * 86400  # m/s -> m/yr
MM_PER_M = 1000                     # m/yr -> mm/yr
CONV = SECONDS_PER_YEAR * MM_PER_M  # m/s -> mm/yr


pairs = [
    (hist_dir / "cesm2LE-historical-cmip6-PRECC.zarr",  hist_dir / "cesm2LE-historical-cmip6-PRECL.zarr",  base_dir / "cesm2LE-historical-cmip6-PR.zarr"),
    (hist_dir / "cesm2LE-historical-smbb-PRECC.zarr",   hist_dir / "cesm2LE-historical-smbb-PRECL.zarr",   base_dir / "cesm2LE-historical-smbb-PR.zarr"),
    (ssp3_dir / "cesm2LE-ssp370-cmip6-PRECC.zarr",      ssp3_dir / "cesm2LE-ssp370-cmip6-PRECL.zarr",      base_dir / "cesm2LE-ssp370-cmip6-PR.zarr"),
    (ssp3_dir / "cesm2LE-ssp370-smbb-PRECC.zarr",       ssp3_dir / "cesm2LE-ssp370-smbb-PRECL.zarr",       base_dir / "cesm2LE-ssp370-smbb-PR.zarr"),
]

for precc_path, precl_path, out_path in pairs:
    print(f"Processing {out_path.name}...")
    ds_c = xr.open_zarr(precc_path)
    ds_l = xr.open_zarr(precl_path)

    pr = (ds_c["PRECC"] + ds_l["PRECL"]) * CONV
    pr = pr.rename("PR")
    pr.attrs.update({
        "long_name": "Total precipitation rate",
        "units": "mm/yr",
        "description": "PRECC + PRECL"
    })

    xr.Dataset({"PR": pr}).to_zarr(out_path, mode="w")
    print(f" Saved -> {out_path}")

print("Done.")

