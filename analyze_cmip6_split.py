"""
analyze_cmip6_split.py
======================
Surveys the processed_by_model directory and prints a breakdown of models
by family, member count, and data availability. Helps design a principled
train/val/test split for CMIP6 finetuning.

All processed data is now netCDF (.nc), read with xr.open_dataset.
"""

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# ── configuration ──────────────────────────────────────────────────────────────
BY_MODEL_DIR = Path("/Users/ewellmeyer/Documents/research/CMIP6/processed_by_model")
AMIP_DIR     = Path("/Users/ewellmeyer/Documents/research/AMIP/processed")
OUT_FIG      = BY_MODEL_DIR / "split_analysis.png"

MIN_MEMBERS_TEST = 5
MIN_MEMBERS_VAL  = 3

GCM_FAMILIES = {
    "CCM0B":    ["BCC-CSM2-MR", "CESM2", "CESM2-WACCM", "CESM2-LENS2",
                 "E3SM-1-0", "E3SM-1-1", "E3SM-1-1-ECA", "E3SM-2-0",
                 "FIO-ESM-2-0", "FGOALS-f3-L", "FGOALS-g3",
                 "NorESM2-LM", "NorESM2-MM", "TaiESM1"],
    "ECHAM0":   ["AWI-CM-1-1-MR", "CAMS-CSM1-0", "MPI-ESM1-2-HR",
                 "MPI-ESM1-2-LR", "NESM3"],
    "ARPEGE":   ["CNRM-CM6-1", "CNRM-ESM2-1", "CMCC-CM2-SR5", "CMCC-ESM2"],
    "EC-Earth3":["EC-Earth3", "EC-Earth3-AerChem", "EC-Earth3-CC",
                 "EC-Earth3-Veg", "EC-Earth3-Veg-LR"],
    "GFDL":     ["GFDL-ESM4", "KIOST-ESM"],
    "MIROC":    ["MIROC6", "MIROC-ES2L"],
    "UCLA":     ["MRI-ESM2-0", "GISS-E2-1-G", "GISS-E2-1-G-CC",
                 "GISS-E2-1-H", "GISS-E2-2-G", "MCM-UA-1-0"],
    "HadAM3":   ["HadGEM3-GC31-LL", "KACE-1-0-G", "ACCESS-CM2",
                 "ACCESS-ESM1-5", "UKESM1-0-LL", "UKESM1-1-LL",
                 "HadGEM3-PPE"],
    "CanAM3":   ["CanESM5", "CanESM5-1", "CanESM5-CanOE"],
    "INM":      ["INM-CM4-8", "INM-CM5-0"],
    "IPSL":     ["IPSL-CM5A2-INCA", "IPSL-CM6A-LR"],
}


def family_of(model_name):
    base = model_name.split("_")[0]
    for fam, members in GCM_FAMILIES.items():
        if base in members:
            return fam
    return "unknown"


# =============================================================================
# scan AMIP directory for model names (overlap check)
# =============================================================================

amip_models = set()
if AMIP_DIR.exists():
    # check both .nc and .zarr in case AMIP hasn't been reprocessed yet
    for pattern in ("AMIP_*_clim.nc", "AMIP_*_clim.zarr"):
        for p in AMIP_DIR.glob(pattern):
            parts = p.stem.replace("_clim", "").split("_", 1)
            if len(parts) == 2:
                amip_models.add(parts[1])
    if amip_models:
        print(f"AMIP models found ({len(amip_models)}): {sorted(amip_models)}")
else:
    print(f"WARN: AMIP directory not found at {AMIP_DIR}, skipping overlap check")


def amip_overlap(group_name):
    base = group_name.split("_")[0]
    candidates = {base}
    if "-LENS2" in base:
        candidates.add(base.replace("-LENS2", ""))
    return candidates & amip_models


# =============================================================================
# scan directory
# =============================================================================

def open_nc_safe(path):
    """Open a netCDF file; return dataset or None."""
    try:
        ds = xr.open_dataset(path)
        if "lat" not in ds.dims and "latitude" not in ds.dims:
            print(f"  WARN: {path.name} has no lat dimension")
            ds.close()
            return None
        return ds
    except Exception as e:
        print(f"  WARN: could not open {path.name}: {e}")
        return None


clim_files = sorted(BY_MODEL_DIR.glob("*_clim.nc"))

groups = {}
for p in clim_files:
    name = p.name.replace("_clim.nc", "")

    ds_clim = open_nc_safe(p)
    if ds_clim is None:
        print(f"  SKIP: {name} — clim file is corrupt or missing coords")
        continue
    ds_clim.close()

    trend_path = BY_MODEL_DIR / f"{name}_trend.nc"
    dpdp_path  = BY_MODEL_DIR / f"{name}_dPdP.nc"
    trend_ok   = trend_path.exists()
    dpdp_ok    = dpdp_path.exists()

    n_members = 1
    if trend_ok:
        ds_trend = open_nc_safe(trend_path)
        if ds_trend is not None:
            if "member" in ds_trend.dims:
                n_members = ds_trend.sizes["member"]
            elif "pr_trend" in ds_trend and ds_trend["pr_trend"].ndim == 3:
                n_members = ds_trend["pr_trend"].shape[0]
            ds_trend.close()
        else:
            trend_ok = False

    overlap = amip_overlap(name)

    groups[name] = dict(
        family       = family_of(name),
        n_members    = n_members,
        has_trend    = trend_ok,
        has_dpdp     = dpdp_ok,
        complete     = trend_ok and dpdp_ok,
        amip_overlap = overlap,
    )

# =============================================================================
# warn about unknown families
# =============================================================================

unknowns = [(n, i) for n, i in groups.items() if i["family"] == "unknown"]
if unknowns:
    print(f"\n{'!'*70}")
    print(f"WARNING: {len(unknowns)} group(s) mapped to 'unknown' family.")
    print("These models are not in GCM_FAMILIES — please assign them manually")
    print("to avoid accidental data leakage between train/val/test.")
    for n, i in unknowns:
        print(f"  {n}  (members={i['n_members']}, complete={i['complete']})")
    print(f"{'!'*70}\n")

# =============================================================================
# print per-family summary
# =============================================================================

by_family = defaultdict(list)
for name, info in groups.items():
    by_family[info["family"]].append((name, info))

print(f"\n{'='*70}")
print(f"{'FAMILY':<14} {'MODEL/GROUP':<30} {'MEMBERS':>7}  {'COMPLETE':<10} AMIP?")
print(f"{'='*70}")

family_summary = {}
for fam in sorted(by_family):
    entries  = sorted(by_family[fam], key=lambda x: x[0])
    n_groups = len(entries)
    n_total  = sum(e[1]["n_members"] for e in entries)
    n_complete = sum(1 for e in entries if e[1]["complete"])
    family_summary[fam] = dict(n_groups=n_groups, n_total=n_total,
                                n_complete=n_complete, entries=entries)
    for i, (name, info) in enumerate(entries):
        prefix  = fam if i == 0 else ""
        flag    = "ok" if info["complete"] else "MISSING"
        amip_flag = ",".join(info["amip_overlap"]) if info["amip_overlap"] else ""
        print(f"  {prefix:<12} {name:<30} {info['n_members']:>7}  {flag:<10} {amip_flag}")
    print(f"  {'':12} {'--- family total ---':<30} {n_total:>7}  "
          f"({n_complete}/{n_groups} complete)")
    print()

# =============================================================================
# summary counts
# =============================================================================

single     = [(n, i) for n, i in groups.items() if i["n_members"] == 1 and i["complete"]]
multi      = [(n, i) for n, i in groups.items() if i["n_members"] > 1  and i["complete"]]
incomplete = [(n, i) for n, i in groups.items() if not i["complete"]]

print(f"{'='*70}")
print(f"Complete groups:    {len(single) + len(multi)}")
print(f"  multi-member:     {len(multi)}")
print(f"  single-member:    {len(single)}")
print(f"Incomplete (skip):  {len(incomplete)}")
for n, _ in incomplete:
    print(f"  {n}")

# =============================================================================
# suggested split
# =============================================================================

print(f"\n{'='*70}")
print("SUGGESTED SPLIT")
print(f"  test  requires >= {MIN_MEMBERS_TEST} members")
print(f"  val   requires >= {MIN_MEMBERS_VAL} members")
print(f"{'='*70}")

suggested_test  = []
suggested_val   = []
suggested_train = []

for fam, summ in sorted(family_summary.items()):
    complete = [(n, i) for n, i in summ["entries"] if i["complete"]]
    if not complete:
        continue
    complete_sorted = sorted(complete, key=lambda x: -x[1]["n_members"])
    assigned_test = False
    assigned_val  = False
    for name, info in complete_sorted:
        if not assigned_test and info["n_members"] >= MIN_MEMBERS_TEST:
            suggested_test.append((name, fam))
            assigned_test = True
        elif not assigned_val and info["n_members"] >= MIN_MEMBERS_VAL:
            suggested_val.append((name, fam))
            assigned_val = True
        else:
            suggested_train.append((name, fam))

print(f"\nTEST  ({len(suggested_test)} groups):")
for n, fam in suggested_test:
    overlap = groups[n]["amip_overlap"]
    warn = f"  *** AMIP OVERLAP: {','.join(overlap)} ***" if overlap else ""
    print(f"  {n:<35} [{fam}]  members={groups[n]['n_members']}{warn}")

print(f"\nVAL   ({len(suggested_val)} groups):")
for n, fam in suggested_val:
    overlap = groups[n]["amip_overlap"]
    warn = f"  *** AMIP OVERLAP: {','.join(overlap)} ***" if overlap else ""
    print(f"  {n:<35} [{fam}]  members={groups[n]['n_members']}{warn}")

print(f"\nTRAIN ({len(suggested_train)} groups):")
for n, fam in suggested_train:
    overlap = groups[n]["amip_overlap"]
    note = f"  (also in AMIP)" if overlap else ""
    print(f"  {n:<35} [{fam}]  members={groups[n]['n_members']}{note}")

n_test  = sum(groups[n]["n_members"] for n, _ in suggested_test)
n_val   = sum(groups[n]["n_members"] for n, _ in suggested_val)
n_train = sum(groups[n]["n_members"] for n, _ in suggested_train)
print(f"\nTotal members — train: {n_train}  val: {n_val}  test: {n_test}")

# =============================================================================
# bar chart
# =============================================================================

color_map = {}
for n, _ in suggested_test:  color_map[n] = "tomato"
for n, _ in suggested_val:   color_map[n] = "gold"
for n, _ in suggested_train: color_map[n] = "steelblue"
for n, i in groups.items():
    if not i["complete"]:
        color_map[n] = "lightgray"

names   = sorted(groups.keys())
members = [groups[n]["n_members"] for n in names]
colors  = [color_map.get(n, "lightgray") for n in names]

fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.35), 6))
bars = ax.bar(range(len(names)), members, color=colors, edgecolor="white", lw=0.5)

for idx, n in enumerate(names):
    if groups[n]["amip_overlap"]:
        ax.plot(idx, members[idx] * 1.15, marker="D", color="black",
                markersize=4, zorder=5)

ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=75, ha="right", fontsize=7)
ax.set_ylabel("Number of members")
ax.set_title(
    "CMIP6 processed_by_model — member counts by group\n"
    f"blue=train  gold=val (>={MIN_MEMBERS_VAL})  "
    f"red=test (>={MIN_MEMBERS_TEST})  gray=incomplete  "
    u"\u25C6=AMIP overlap"
)
ax.set_yscale("log")
ax.grid(True, alpha=0.3, axis="y")

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend = [
    Patch(color="steelblue", label="train"),
    Patch(color="gold",      label=f"val (>={MIN_MEMBERS_VAL} members)"),
    Patch(color="tomato",    label=f"test (>={MIN_MEMBERS_TEST} members)"),
    Patch(color="lightgray", label="incomplete/skip"),
    Line2D([0], [0], marker="D", color="black", linestyle="None",
           markersize=4, label="AMIP overlap"),
]
ax.legend(handles=legend, fontsize=8, loc="upper left")
plt.tight_layout()
fig.savefig(OUT_FIG, dpi=120, bbox_inches="tight")
print(f"\nFigure saved to {OUT_FIG}")