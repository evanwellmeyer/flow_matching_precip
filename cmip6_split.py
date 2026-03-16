"""
cmip6_split.py
==============
Hardcoded train / val / test split for CMIP6 finetuning.

This split was designed by analyze_cmip6_split.py with the following rules:
  - TEST  : one group per family with >= 5 members (largest first)
  - VAL   : next-largest per family with >= 3 members
  - TRAIN : everything else that is complete
  - Families where no group clears thresholds go entirely to train

The split is deterministic and should not be randomized.

Usage
-----
    from cmip6_split import TRAIN_GROUPS, VAL_GROUPS, TEST_GROUPS
    from cmip6_split import group_family, ALL_FAMILIES
"""

# ── family mapping (same as flow_datasets.py / analyze_cmip6_split.py) ────────

ALL_FAMILIES = {
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


def group_family(group_name):
    """Map a group name (e.g. CanESM5_p1, HadGEM3-PPE_01) to its family."""
    base = group_name.split("_")[0]
    for fam, members in ALL_FAMILIES.items():
        if base in members:
            return fam
    return "unknown"


# ── hardcoded split ───────────────────────────────────────────────────────────
# format: list of (group_name, family)

TEST_GROUPS = [
    ("CNRM-CM6-1",        "ARPEGE"),
    ("CESM2-LENS2_cmip6", "CCM0B"),
    ("CanESM5-1_p1",      "CanAM3"),
    ("EC-Earth3",         "EC-Earth3"),
    ("MPI-ESM1-2-LR",    "ECHAM0"),
    ("ACCESS-ESM1-5",    "HadAM3"),
    ("INM-CM5-0",        "INM"),
    ("IPSL-CM6A-LR",     "IPSL"),
    ("MIROC6",           "MIROC"),
    ("GISS-E2-1-G_p1",   "UCLA"),
]

VAL_GROUPS = [
    ("CNRM-ESM2-1",      "ARPEGE"),
    ("CESM2-LENS2_smbb", "CCM0B"),
    ("CanESM5_p2",       "CanAM3"),
    ("EC-Earth3-Veg",    "EC-Earth3"),
    ("MPI-ESM1-2-HR",    "ECHAM0"),
    ("GFDL-ESM4",        "GFDL"),
    ("UKESM1-0-LL",      "HadAM3"),
    ("MIROC-ES2L",       "MIROC"),
    ("GISS-E2-1-H_p1",   "UCLA"),
]

TRAIN_GROUPS = [
    ("CMCC-CM2-SR5_p1",  "ARPEGE"),
    ("CMCC-ESM2",        "ARPEGE"),
    ("E3SM-2-0",         "CCM0B"),
    ("E3SM-1-0_p2",      "CCM0B"),
    ("FGOALS-g3",        "CCM0B"),
    ("BCC-CSM2-MR",      "CCM0B"),
    ("CESM2-WACCM",      "CCM0B"),
    ("FGOALS-f3-L",      "CCM0B"),
    ("FIO-ESM-2-0",      "CCM0B"),
    ("NorESM2-LM",       "CCM0B"),
    ("NorESM2-MM",       "CCM0B"),
    ("TaiESM1",          "CCM0B"),
    ("E3SM-1-1",         "CCM0B"),
    ("E3SM-1-1-ECA",     "CCM0B"),
    ("CanESM5-1_p2",     "CanAM3"),
    ("CanESM5_p1",       "CanAM3"),
    ("CanESM5-CanOE",    "CanAM3"),
    ("EC-Earth3-CC",     "EC-Earth3"),
    ("EC-Earth3-AerChem","EC-Earth3"),
    ("EC-Earth3-Veg-LR", "EC-Earth3"),
    ("AWI-CM-1-1-MR",    "ECHAM0"),
    ("NESM3",            "ECHAM0"),
    ("CAMS-CSM1-0",      "ECHAM0"),
    ("KIOST-ESM",        "GFDL"),
    ("ACCESS-CM2",       "HadAM3"),
    ("HadGEM3-GC31-LL",  "HadAM3"),
    ("KACE-1-0-G",       "HadAM3"),
    ("HadGEM3-PPE_01",   "HadAM3"),
    ("HadGEM3-PPE_02",   "HadAM3"),
    ("HadGEM3-PPE_03",   "HadAM3"),
    ("HadGEM3-PPE_04",   "HadAM3"),
    ("HadGEM3-PPE_05",   "HadAM3"),
    ("HadGEM3-PPE_06",   "HadAM3"),
    ("HadGEM3-PPE_07",   "HadAM3"),
    ("HadGEM3-PPE_08",   "HadAM3"),
    ("HadGEM3-PPE_09",   "HadAM3"),
    ("HadGEM3-PPE_10",   "HadAM3"),
    ("HadGEM3-PPE_11",   "HadAM3"),
    ("HadGEM3-PPE_12",   "HadAM3"),
    ("HadGEM3-PPE_13",   "HadAM3"),
    ("HadGEM3-PPE_14",   "HadAM3"),
    ("HadGEM3-PPE_15",   "HadAM3"),
    ("HadGEM3-PPE_16",   "HadAM3"),
    ("HadGEM3-PPE_17",   "HadAM3"),
    ("HadGEM3-PPE_18",   "HadAM3"),
    ("HadGEM3-PPE_19",   "HadAM3"),
    ("HadGEM3-PPE_20",   "HadAM3"),
    ("HadGEM3-PPE_21",   "HadAM3"),
    ("HadGEM3-PPE_22",   "HadAM3"),
    ("HadGEM3-PPE_23",   "HadAM3"),
    ("HadGEM3-PPE_24",   "HadAM3"),
    ("HadGEM3-PPE_25",   "HadAM3"),
    ("HadGEM3-PPE_26",   "HadAM3"),
    ("HadGEM3-PPE_27",   "HadAM3"),
    ("HadGEM3-PPE_28",   "HadAM3"),
    ("UKESM1-1-LL",      "HadAM3"),
    ("INM-CM4-8",        "INM"),
    ("IPSL-CM5A2-INCA",  "IPSL"),
    ("MRI-ESM2-0",       "UCLA"),
    ("GISS-E2-1-G_p3",   "UCLA"),
    ("GISS-E2-1-G_p5",   "UCLA"),
    ("GISS-E2-1-H_p3",   "UCLA"),
    ("GISS-E2-2-G",      "UCLA"),
    ("MCM-UA-1-0",       "UCLA"),
    ("GISS-E2-1-G-CC",   "UCLA"),
]

# convenience sets for quick membership checks
TEST_SET  = {name for name, _ in TEST_GROUPS}
VAL_SET   = {name for name, _ in VAL_GROUPS}
TRAIN_SET = {name for name, _ in TRAIN_GROUPS}