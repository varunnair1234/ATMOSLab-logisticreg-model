import io
import os
import numpy as np
import pandas as pd
import requests
 
# ---------------------------------------------------------------------------
# CONFIG -- edit before running
# ---------------------------------------------------------------------------
 
# Full ISD identifier = "USAF-WBAN". WBAN numbers below are CONFIRMED:
#   KMRY (Monterey Regional)   -> WBAN 23259
#   KWVI (Watsonville Muni)    -> WBAN 23277
#   KSJC (San Jose Intl)       -> WBAN 23293
# The USAF prefix is NOT confirmed yet -- look it up at
# https://www.ncei.noaa.gov/maps/hourly/  (click the station, it shows the
# exact "USAF-WBAN" filename) and replace the placeholders below.
STATIONS = {
    "KMRY": "72491523259",
    "KWVI": "74505823277",
    "KSJC": "72494523293",
}
 
YEARS = range(2015, 2024)
 
# Per Sec 4: "surface to ~1-2 km at ~10-25 m spacing"
HEIGHT_GRID_M = np.arange(0, 2000, 25)
 
# Ceiling below this height counts as a "fog/low cloud" event for splitting
# and labeling purposes. Tune per your definition in Sec 4/8.
FOG_CEILING_THRESHOLD_M = 500
 
TEST_FRACTION_BY_EVENT = 0.2
RANDOM_SEED = 42
 
ISD_BASE_URL = "https://noaa-global-hourly-pds.s3.amazonaws.com"
 
 
# ---------------------------------------------------------------------------
# ISD pull + parsing
# ---------------------------------------------------------------------------
 
def fetch_isd_station_year(usaf_wban: str, year: int) -> pd.DataFrame:
    """Pull one station-year of ISD global-hourly data (public S3, no auth)."""
    url = f"{ISD_BASE_URL}/{year}/{usaf_wban}.csv"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))
 
 
def parse_ceiling(cig_field) -> float:
    """
    ISD 'CIG' field is a packed string like '00450,1,N,9':
      ceiling height (m), quality code, determination code, CAVOK code.
    Height 99999 = not reported for this observation.
 
    NOTE (verify once you have real KMRY data): for a proper US ASOS
    station, 99999 usually means the report simply didn't include a
    ceiling group for that hour -- NOT necessarily "sky was clear."
    Confirmed-clear conditions are typically reported as a real height
    value (e.g. ~22000 ft / ~6700 m) rather than the missing code.
    Spot-check a known-clear day in your pulled data to confirm this holds
    for KMRY/KWVI/KSJC before trusting the fog labels at scale.
    """
    try:
        height_m = float(str(cig_field).split(",")[0])
        return np.nan if height_m >= 99999 else height_m
    except Exception:
        return np.nan
 
 
def parse_temp_field(field) -> float:
    """ISD TMP/DEW fields are packed like '+0125,1' in tenths of degC."""
    try:
        val = float(str(field).split(",")[0]) / 10.0
        return np.nan if val >= 999 else val
    except Exception:
        return np.nan
 
 
def build_surface_features(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["datetime"] = pd.to_datetime(raw["DATE"])
    out["ceiling_m"] = raw["CIG"].apply(parse_ceiling) if "CIG" in raw else np.nan
    out["temp_c"] = raw["TMP"].apply(parse_temp_field) if "TMP" in raw else np.nan
    out["dewpoint_c"] = raw["DEW"].apply(parse_temp_field) if "DEW" in raw else np.nan
    out["hour"] = out["datetime"].dt.hour
    out["day_of_year"] = out["datetime"].dt.dayofyear
    out["dewpoint_depression_c"] = out["temp_c"] - out["dewpoint_c"]
    return out
 
 
# ---------------------------------------------------------------------------
# Optional: ARM ceilometer pull (stub -- fill in once you have credentials)
# ---------------------------------------------------------------------------
 
def pull_arm_ceilometer(datastream: str, start: str, end: str):
    """
    Requires: pip install act-atmos, ARM_USERNAME/ARM_PASSWORD env vars.
    Returns an xarray Dataset of backscatter/cloud-base from ARM.
    Wire this in to fuse with ISD per Sec 3's "don't rely on backscatter
    alone" caveat -- e.g. join on nearest timestamp, add as extra columns.
    """
    import act  # local import so this file still runs without act-atmos installed
    username = os.environ["ARM_USERNAME"]
    token = os.environ["ARM_PASSWORD"]
    results = act.discovery.download_arm_data(username, token, datastream, start, end)
    return act.io.arm.read_arm_netcdf(results)
 
 
# ---------------------------------------------------------------------------
# Labeling + event-based split
# ---------------------------------------------------------------------------
 
def label_fog_bins(ceiling_m: float) -> np.ndarray:
    """
    Per-height-bin fog label from ceiling height (Sec 4/5).
 
    A NaN ceiling (no CIG group reported that hour) is treated as
    "no fog detected" (all-zero label) rather than dropping the row --
    this keeps the row as a usable negative example instead of silently
    discarding it. If you confirm 99999 sometimes means genuine missing
    data (not clear sky) for these stations, revisit this default.
 
    This is the simple ceiling-only version -- per Sec 3's caveat, swap
    in ARM backscatter-derived labels here once pull_arm_ceilometer() is
    wired in; ceiling alone is a coarse proxy, not the final label source.
    """
    labels = np.zeros(len(HEIGHT_GRID_M), dtype=int)
    if not np.isnan(ceiling_m) and ceiling_m < FOG_CEILING_THRESHOLD_M:
        labels[HEIGHT_GRID_M <= ceiling_m] = 1
    return labels
 
 
def event_based_split(df: pd.DataFrame, test_fraction: float, seed: int):
    """
    Splits by fog EVENT (contiguous run of foggy hours), not by row, so
    adjacent hours of the same event never land on both sides of the split --
    the temporal-leakage risk flagged in Sec 1. NaN ceiling rows are
    naturally treated as non-fog here (NaN comparisons are False).
    """
    df = df.sort_values("datetime").reset_index(drop=True)
    is_fog = df["ceiling_m"] < FOG_CEILING_THRESHOLD_M
    event_id = (is_fog != is_fog.shift()).cumsum()
 
    fog_event_ids = sorted(set(event_id[is_fog]))
    rng = np.random.default_rng(seed)
    rng.shuffle(fog_event_ids)
    n_test = max(1, int(len(fog_event_ids) * test_fraction))
    test_events = set(fog_event_ids[:n_test])
 
    test_mask = is_fog & event_id.isin(test_events)
    return df[~test_mask].copy(), df[test_mask].copy()
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main():
    if any("PLACEHOLDER" in v for v in STATIONS.values()):
        raise RuntimeError(
            "STATIONS still contains USAF_PLACEHOLDER values. Look up the "
            "real USAF-WBAN identifiers at https://www.ncei.noaa.gov/maps/hourly/ "
            "and fill them in before running."
        )
 
    all_frames = []
    for station_name, usaf_wban in STATIONS.items():
        for year in YEARS:
            try:
                raw = fetch_isd_station_year(usaf_wban, year)
            except Exception as e:
                print(f"  skip {station_name} {year}: {e}")
                continue
            feats = build_surface_features(raw)
            feats["station"] = station_name
            all_frames.append(feats)
            print(f"  pulled {station_name} {year}: {len(feats)} rows")
 
    if not all_frames:
        raise RuntimeError(
            "No ISD data pulled -- check station IDs in STATIONS and your "
            "network access to noaa-global-hourly-pds.s3.amazonaws.com"
        )
 
    data = pd.concat(all_frames, ignore_index=True)
    # Only require the actual predictive features to be present -- do NOT
    # drop rows just because ceiling is NaN (see label_fog_bins docstring).
    data = data.dropna(subset=["temp_c", "dewpoint_c"])
 
    label_matrix = np.stack([label_fog_bins(c) for c in data["ceiling_m"]])
    for i, h in enumerate(HEIGHT_GRID_M):
        data[f"fog_bin_{int(h)}m"] = label_matrix[:, i]
 
    train_df, test_df = event_based_split(data, TEST_FRACTION_BY_EVENT, RANDOM_SEED)
    train_df.to_csv("train.csv", index=False)
    test_df.to_csv("test.csv", index=False)
    print(f"\nwrote train.csv ({len(train_df)} rows), test.csv ({len(test_df)} rows)")
 
 
if __name__ == "__main__":
    main()