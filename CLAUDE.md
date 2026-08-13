# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

This repo supports a research project predicting the **vertical distribution
of fog** (base, top, depth) before drone flights over the Santa Cruz /
Monterey Bay coast, for flight planning and go/no-go decisions. Full design
rationale lives in the project proposal (fog is governed by marine
boundary-layer structure, not surface conditions alone; training data comes
from long-record remote sensing, not from the team's own limited drone
flights, which serve only as in-situ validation).

Current phase: building a **logistic regression baseline** before any deep
model, per the proposal's "simple baselines first" plan.

## Data Pipeline — `get_data.py` (originally `build_training_data.py`)

Pulls NOAA Integrated Surface Database (ISD) hourly data directly from the
public S3 bucket `noaa-global-hourly-pds` (no auth needed) for three
stations, engineers features, builds per-height-bin fog labels, and writes
an event-based `train.csv` / `test.csv` split.

**Stations (confirmed via `isd-history.txt`, cross-checked against a real
downloaded sample file):**
| Station | USAF | WBAN | S3 filename |
|---|---|---|---|
| KMRY (Monterey Regional) | 724915 | 23259 | `72491523259` |
| KWVI (Watsonville Muni) | 745058 | 23277 | `74505823277` |
| KSJC (San Jose Intl) | 724945 | 23293 | `72494523293` |

**Known gotcha (already fixed, don't reintroduce):** ISD S3 filenames are
`USAF` and `WBAN` **concatenated directly, no separator** (e.g.
`72491523259.csv`), NOT `USAF-WBAN.csv`. The dash format 404s.

**Open question, not yet resolved:** the `CIG` (ceiling) field reports
`99999` for a lot of rows. Unclear whether this means "genuinely missing
observation" or something else for these specific ASOS stations. Currently
treated as "no fog detected" (label = all zeros) rather than dropped, so as
not to lose negative training examples. Spot-check a known-clear KMRY day
against this before trusting fog labels at scale.

**Design choices already made:**
- Height grid: surface to 2 km, 25 m bins (`HEIGHT_GRID_M`)
- Fog threshold: ceiling < 500 m (`FOG_CEILING_THRESHOLD_M`)
- Train/test split is by **fog event** (contiguous foggy hours), not by row
  — prevents adjacent hours of the same event leaking across the split
- ARM ceilometer backscatter fusion is stubbed (`pull_arm_ceilometer()`) but
  not wired in yet — current labels are ceiling-only, which the proposal
  itself flags as insufficient alone

## Pipeline / Roadmap

The project runs in five phases. Phase 1 is implemented (`get_data.py`);
phases 2–5 are planned, not yet built.

**Phase 1 — Data Pipeline** (`get_data.py`, done; see above for gotchas)
1. Identify data sources — NOAA ISD, public S3 bucket
2. Use Santa Cruz-area stations — KMRY, KWVI, KSJC
3. `get_data.py` — script for retrieving and parsing data into CSV files
4. Spot-check the `CIG=99999` assumption (still open, see above)

**Phase 2 — Feature Engineering** (not yet implemented)
1. Add wind speed/direction parsing — ISD `WND` field currently missing
   from `build_surface_features()`, even though the proposal's own cited
   radiation-fog rule (RH ≥ 94%, wind ≤ 3 m/s, inversion > 250 m) depends
   on it
2. Add RH (Magnus formula from temp + dewpoint) instead of relying on
   dewpoint depression alone
3. Add a dewpoint-depression × wind-speed interaction term — logistic
   regression is linear and can't represent the "AND" condition in the
   radiation-fog rule without it being handed in explicitly
4. Cyclical encoding (`sin`/`cos`) for `hour` and `day_of_year` instead of
   raw integers

**Phase 3 — Model Formulation** (not yet implemented)
1. Replace ~80 independent per-height-bin regressions with a base + top
   formulation (2 regressions), reconstructing per-bin labels from those
   two outputs — the label is a step function of height (1 below ceiling,
   0 above), so this also fixes sparse-positive-bin issues in upper bins,
   which rarely see fog

**Phase 4 — Data Fitting** (not yet implemented)
1. Fit with `class_weight='balanced'` — fog-positive bins are a small
   minority
2. Apply probability calibration (Platt or isotonic) after fitting so
   predicted probability is trustworthy for go/no-go decisions

**Phase 5 — Model Eval** (not yet implemented)
1. Score with POD, FAR, CSI, PR-AUC per height bin
2. MAE on predicted fog base/top height, plus reliability curves using
   Matplotlib
3. In-situ validation against the team's own drone flights
4. Keep data testing
5. 1D-CNN/GRU + MC-dropout uncertainty → go/no-go decision support

## Current Status / Next Step

- [ ] Confirm `get_data.py` runs cleanly end-to-end after the filename fix
      (last known state: fix applied, rerun not yet confirmed) and produces
      non-trivial `train.csv` / `test.csv`
- [ ] Spot-check the `CIG=99999` assumption against real KMRY data
- [ ] Begin Phase 2 (feature engineering: wind, RH, interaction term,
      cyclical encoding)

## Out of Scope for This Repo

A separate HuggingFace LLM integration ("actionable insights" layer) is
planned but belongs to a different part of the project, not this
logistic-regression pipeline. Don't conflate the two.
