# Electrochemistry Analysis App

Streamlit application for square-wave voltammetry (SWV), cyclic voltammetry
(CV), Bayesian-optimization (BO) session review, and publication-ready figure
composition.

The repository contains source code, tests, default formatting presets, and
small reference configuration files. Raw measurements, saved analysis sessions,
derived caches, and generated figures are intentionally local artifacts and are
not versioned.

## Start the app

On Windows, run:

```powershell
.\Open_SWV_App.cmd
```

Or create an environment and launch Streamlit directly:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

On macOS, use `launch_app_mac.command` after installing the requirements.

## Main workflows

| Task | Mode | Main output |
|---|---|---|
| Batch SWV analysis | **SWV** | Corrected traces, peaks, response tables, titration fits, QC and exports |
| CV analysis | **CV** | CV traces, derived metrics and exports |
| Parameter-sweep landscape | **BO Session** with a survey session | Figure Types 1, 1A, 1B and 1C |
| Real BO validation | **BO Session** with an optimization session | Figure Types 2 and 2A |
| Titration comparison | **SWV > Paper Figures** | Figure Types 3, 3A, 3B, 3C and 3D |
| Concentration calibration | **SWV > Paper Figures** | Figure Types 4 and 4A |

Survey and optimization sessions expose different BO figure types. The titration
figure types appear only after SWV data have been analyzed and **Paper Figures**
is opened.

### SWV and titration analysis

1. Select **SWV**, choose the measurement folders, and verify the detected or
   manually selected BO analysis settings.
2. Run the analysis. Missing or failed scans remain missing/failed; the app does
   not silently impute measurements.
3. For titrations, enable **Treat vline intervals as titration steps**, review the
   interval labels/concentrations, and choose the accepted concentration steps.
4. Enable Langmuir fitting when required. The fit and predicted-versus-known
   panels update from the current interval selection, concentration range,
   methods, channels and analysis settings.
5. Save a named local session. Enable the derived-analysis cache when a fast
   reopen is more important than portability.

Display cropping, axis ticks and formatting do not change raw measurements.
Analysis controls such as accepted scan criteria, titration-step selection,
outlier filtering and peak processing can change derived values and must be
recorded with the figure.

### BO Figure Composer

- Types **1/1A/1B/1C** show parameter-sweep landscapes, slices and example SWVs.
- Types **2/2A** show real BO validation, including Q history, measured cubes,
  buffer/target trends, chronological stacks and parallel coordinates.
- Linked source controls keep a figure on one channel/group/direction. Disable
  linking only when intentionally comparing independent sources.
- A preset stores figure composition and formatting. A saved analysis session
  stores the experiment path, selected data and analysis state. Overwriting a
  formatting preset must not be used as a substitute for saving the session.

### Titration Paper Figures

| Type | Contents |
|---|---|
| 3 | SWV and titration-response comparison |
| 3A | Cross-channel signal-on/off comparison |
| 3B | Signal-on/off with a shared Langmuir treatment |
| 3C | Paired signal-on/off Langmuir fits |
| 3D | Manual SWVs, optimized SWVs and response versus measurement |
| 4 | Concentration by measurement and predicted versus known |
| 4A | Langmuir fit plus the two Type 4 panels |

The editable canvas is shared by the repeated rows. Each row reuses that layout
with its own selected channel and methods. Aspect-ratio changes affect both the
canvas and the rendered result. Formatting presets are stored per figure type so
loading or editing one type does not overwrite another type's canvas.

Click **Generate preview** after changing analysis, selection or formatting.
Preview rendering uses 100 DPI by default. Once a preview exists, separate
**PNG**, **PDF**, **SVG** and **ZIP** download buttons appear immediately. The
default export resolution is 600 DPI; vector exports retain vector-compatible
content where possible. Changing controls leaves the last completed preview
visible until the next preview is generated.

Type 4/4A predicted-versus-known panels invert the selected fitted calibration.
They assess the fitted concentration range; they are not independent held-out
validation unless the predictions originate from a separate experiment.

## Saved sessions and caches

New sessions are stored outside the checkout:

```text
Windows: %LOCALAPPDATA%\Electrochemistry Analysis Scripts\analysis_sessions
```

A session recipe is JSON and can contain absolute local data paths. An optional
`*.results.pkl.gz` cache can be tens or hundreds of megabytes. Cache files are
Python pickles: load only caches produced by this application or another trusted
collaborator. The original measurement folders remain the source of truth.

For transferring the workflow to another machine, see
[docs/handoff.md](docs/handoff.md).

## Repository and artifact policy

Keep in Git:

- application and core source code;
- automated tests;
- `figure_composer_presets.json` defaults;
- small, non-private reference configurations under `paper/reference/`;
- current documentation.

Keep outside Git:

- raw measurement folders and BO session directories;
- `analysis_sessions/` and `*.results.pkl.gz` caches;
- generated PNG/PDF/SVG/ZIP exports;
- `paper/generated/`, `paper/outputs/`, `paper/.local_archive/` and `outputs/`;
- local absolute paths, temporary verification scripts and test caches.

This separation lets collaborators run the same software without publishing
large data, machine-specific paths, or unsafe serialized caches. Share a small
trusted session/cache bundle out of band only when fast reopening is required.

## Project layout

```text
app.py                         Main Streamlit UI and SWV/paper workflow
bo_session_viewer.py           BO session loading, scoring and composer
bo_headless.py                 Headless BO helpers
core/                          Analysis, plotting, session and composer modules
tests/                         Automated regression tests
figure_composer_presets.json   Versioned default BO formatting presets
paper/reference/               Small reference configurations and channel table
paper/README.md                Paper-figure workflow and interpretation
docs/handoff.md                Setup and data/session handoff guide
```

## Verification

Run the complete suite before committing changes:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

For UI changes, also open the app and verify the affected mode, figure type,
preview, canvas and all requested export formats.

## Scientific safeguards

- Raw source files are never rewritten by figure formatting or Composer edits.
- Visual clipping is not data exclusion and must not be presented as such.
- Experimental abrupt-jump correction is off by default and changes processed
  traces, not the recorded BO objective.
- Saved BO Q, rescored Q and paired SNR are distinct quantities; report which is
  being shown.
- A Type 4/4A calibration inversion should not be described as independent
  validation without separate held-out data.
- Preserve figure metadata/JSON with final exports so selections, filters and
  formatting remain auditable.
