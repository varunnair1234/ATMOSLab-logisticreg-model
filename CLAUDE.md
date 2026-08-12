# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Builds a coastal-fog dataset from NOAA ISD (Integrated Surface Database) hourly
station observations, for training a per-height-bin fog/low-cloud classifier
(logistic regression per the repo name) around three Central California
stations: KMRY (Monterey), KWVI (Watsonville), KSJC (San Jose).

Currently the repo is just the data pipeline (`get_data.py`) plus its output
(`data/train.csv`, `data/test.csv`); no model-training code exists yet.

## Setup / commands

```bash
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt   # pandas, requests, numpy, torch
python get_data.py                # regenerates train.csv / test.csv (writes to CWD)
```

There is no test suite, linter, or build step currently in the repo.

`get_data.py` pulls directly from the public S3 bucket
`noaa-global-hourly-pds` (no auth) for `YEARS = range(2015, 2024)`. Before
running, verify the `STATIONS` dict in the CONFIG block at the top of the
file — the USAF prefix of each USAF-WBAN identifier is called out in-file as
unconfirmed and must be checked at https://www.ncei.noaa.gov/maps/hourly/.
The script raises immediately if any station ID still contains
`"PLACEHOLDER"`.

## Pipeline structure (`get_data.py`)

Single-file pipeline, run top-to-bottom via `main()`:

1. **Fetch** (`fetch_isd_station_year`) — one station-year of raw ISD CSV per
   `(station, year)` combination.
2. **Parse** (`build_surface_features`, `parse_ceiling`, `parse_temp_field`) —
   unpacks ISD's packed string fields (`CIG` for ceiling, `TMP`/`DEW` for
   temperature/dewpoint) into plain floats, deriving `dewpoint_depression_c`.
   ISD's `99999`/`999` sentinel values become `NaN`.
3. **Label** (`label_fog_bins`) — for each row, produces a binary vector over
   `HEIGHT_GRID_M` (0–2000m, 25m spacing): 1 where a bin's height is at or
   below the observed ceiling *and* the ceiling is below
   `FOG_CEILING_THRESHOLD_M` (500m). This is a coarse ceiling-only proxy; the
   code comments flag that it should eventually be replaced/fused with ARM
   ceilometer backscatter data via `pull_arm_ceilometer` (stub, requires
   `act-atmos` + `ARM_USERNAME`/`ARM_PASSWORD`) once available.
4. **Split** (`event_based_split`) — splits by contiguous **fog event**, not
   by row, so hours from the same fog event never straddle train/test. This
   is deliberate to avoid temporal leakage — preserve this invariant if you
   touch the splitting logic.
5. **Write** — concatenates all station-years, drops rows missing
   `temp_c`/`dewpoint_c` (but *not* rows with missing ceiling — those are
   valid negative/no-fog examples, see `label_fog_bins` docstring), appends
   one `fog_bin_{height}m` column per grid point, writes `train.csv` /
   `test.csv` to the current working directory.

## Known caveats baked into the code (read before changing labeling logic)

- Whether ISD's `99999` "missing ceiling" code actually means clear sky vs.
  truly missing data is unconfirmed for these three stations — see the
  docstring on `parse_ceiling`. Spot-check before trusting fog labels at
  scale.
- `label_fog_bins` treats `NaN` ceiling as "no fog" by design (keeps the row
  as a negative example) rather than dropping it — don't silently change this
  without updating the split/labeling assumptions downstream.
