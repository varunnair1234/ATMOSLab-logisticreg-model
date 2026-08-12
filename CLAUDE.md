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

## Modeling Plan (logistic regression baseline)

Decided direction, not yet implemented — do these together, not piecemeal:

1. **Base/top reformulation.** Don't fit ~80 independent per-bin logistic
   regressions. The label is a step function of height (1 below ceiling, 0
   above), so predict fog base + fog top directly (2 regressions) and
   reconstruct per-bin labels from those. Also fixes sparse-positive-bin
   issues in the upper bins, which rarely see fog.
2. **Add wind speed/direction** (ISD `WND` field) — currently missing from
   `build_surface_features()`, even though the proposal's own cited
   radiation-fog rule (RH ≥ 94%, wind ≤ 3 m/s, inversion > 250 m) depends on
   it.
3. **Add relative humidity** (Magnus formula from temp + dewpoint) instead
   of relying on dewpoint depression alone.
4. **Add a dewpoint-depression × wind-speed interaction term** — logistic
   regression is linear and can't represent the "AND" condition in the
   radiation-fog rule without it being handed in explicitly.
5. **Class weighting** (`class_weight='balanced'`) and **probability
   calibration** (Platt/isotonic) after fitting — fog-positive bins are a
   small minority; raw imbalanced logistic regression will under-report
   confidence on exactly the cases that matter for a go/no-go decision.
6. Cyclical encoding (`sin`/`cos`) for `hour` and `day_of_year` instead of
   raw integers.

## Current Status / Next Step

- [ ] Confirm `get_data.py` runs cleanly end-to-end after the filename fix
      (last known state: fix applied, rerun not yet confirmed) and produces
      non-trivial `train.csv` / `test.csv`
- [ ] Implement base/top reformulation + wind/RH features + class
      weighting + calibration together (Modeling Plan above)
- [ ] Spot-check the `CIG=99999` assumption against real KMRY data

## Out of Scope for This Repo

A separate HuggingFace LLM integration ("actionable insights" layer) is
planned but belongs to a different part of the project, not this
logistic-regression pipeline. Don't conflate the two.
