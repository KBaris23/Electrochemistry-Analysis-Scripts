# Handoff

This guide transfers the application and its analysis workflow without putting
raw measurements, private paths, generated figures, or large Python caches in
Git.

## 1. What the repository provides

After cloning, the collaborator receives:

- the Streamlit application and analysis code;
- automated tests;
- default BO Composer presets;
- small paper reference configurations;
- the current app and paper workflow documentation.

It does not provide the original measurement folders, BO session directories,
local saved-session recipes, cached analysis results, or manuscript exports.
Those artifacts remain on the data owner's machine or are transferred through an
approved data-sharing service.

## 2. Install and verify

```powershell
git clone <REPOSITORY_URL>
cd <CHECKOUT>
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\Open_SWV_App.cmd
```

Use `python3 -m venv .venv` and `pip` equivalents on macOS/Linux.

## 3. Put data in a local data root

Use a location outside the checkout, represented below by `<DATA_ROOT>`. Preserve
each experiment's internal filenames, queue files, BO snapshots and `bo_sessions`
subdirectories. The app accepts local absolute paths, so the collaborator's data
root does not need to match the original machine.

Recommended logical inputs:

| Saved-session name | Mode | Local source to select |
|---|---|---|
| `np_sweep` | BO Session | NP survey `bo_sessions/<survey-session>` |
| `np_validation` | BO Session | NP optimization `bo_sessions/<optimization-session>` |
| `np_titration` | SWV | NP titration measurement root |
| `kana_try_sweep` | BO Session | Planar/Kana survey session |
| `kana_try_validation` | BO Session | Planar/Kana optimization session |
| `kana_try_titration` | SWV | `<DATA_ROOT>/titration_only/kana_try` or its original experiment root |
| `vanco_validation` | BO Session | Vancomycin optimization session |
| `vanco_titration` | SWV | `<DATA_ROOT>/titration_only/vanco` or its original experiment root |
| `amp0_validation` | BO Session | Amp0 optimization session |
| `amp0_titration` | SWV | `<DATA_ROOT>/titration_only/amp0` or its original experiment root |

The names are conveniences, not required folder names. Planar/Kana and Amp0 are
planar-electrode datasets; NP is the nanoporous-electrode dataset. Confirm the
experiment metadata rather than inferring electrode type from a saved-session
name.

## 4. Recreate local sessions from original data

For every BO input:

1. Select **BO Session**.
2. Select the corresponding survey or optimization session folder.
3. Confirm groups/channels and the objective direction.
4. Save a named analysis session using the table above.

For every titration input:

1. Select **SWV** and choose only that experiment's measurement folders.
2. Verify the detected BO snapshot or explicitly select the matching trusted
   snapshot.
3. Run analysis, enable titration intervals, verify labels/concentrations and set
   the accepted concentration range.
4. Enable Langmuir fitting when needed and inspect QC/failures.
5. Save the named analysis session. Enable **Include derived analysis cache for
   fast reopen** if local storage permits.

Saved sessions normally live at:

```text
%LOCALAPPDATA%\Electrochemistry Analysis Scripts\analysis_sessions
```

The recipe retains the current source path and UI/analysis state. The optional
cache accelerates reopening but does not replace the raw data.

## 5. Optional fast-reopen bundle

If the collaborator is trusted and the data-sharing policy permits it, transfer
only the required pairs:

```text
<name>.analysis-session.json
<referenced-cache>.results.pkl.gz   # only when the JSON names one
```

Copy them into the collaborator's local analysis-session directory while the app
is closed. Open the app, load the session, and remap unavailable raw-data paths if
prompted. Do not rename the cache without also updating its recipe reference.

Important:

- caches are compressed Python pickle files and must be treated as trusted code;
- recipes may expose local absolute paths, so inspect/sanitize them before
  sharing;
- source signatures can invalidate a cache when measurement files change;
- GitHub is not the transport for these large/private artifacts.

Rebuilding from the original measurements is the safer and most reproducible
fallback.

## 6. Recreate figures

### Parameter sweep

Open the survey session in **BO Session > Figure Composer**, choose a Type 1
variant, select the intended channel/group, load the desired formatting preset,
and render. The measured slices and cube must come from the same active session.

### Real BO validation

Open the optimization session, choose Type 2 or 2A, and verify the linked
validation group, channel and maximize/minimize direction. Select **Original
recorded Q**, rescored Q or paired SNR explicitly; these are not interchangeable.

### Titration and concentration figures

Open the cached or freshly analyzed SWV session, then select **Paper Figures**:

- use Type 3/3A/3B/3C/3D for SWVs and response curves;
- use Type 4/4A for concentration calibration panels;
- use the saved per-type formatting preset, then verify row channels/methods;
- set a manual display range only when scientifically justified and record it in
  the figure metadata.

Click **Generate preview**. The preview defaults to 100 DPI. Download PNG, PDF,
SVG or ZIP directly from the separate buttons that appear after generation;
publication raster export defaults to 600 DPI.

## 7. Final handoff checklist

- [ ] Original data and BO session folders are accessible outside Git.
- [ ] Each saved session opens and points to the intended experiment.
- [ ] Cached and uncached reopening produce the same derived results.
- [ ] Channel, signal direction, analysis range and concentration range are
      verified for each figure.
- [ ] Final PNG/PDF/SVG/ZIP exports include their metadata/recipe sidecars.
- [ ] Type 4/4A results are described as within-calibration unless a separate
      held-out experiment was used.
- [ ] No raw data, caches, personal absolute paths or generated figures are staged
      in Git.
