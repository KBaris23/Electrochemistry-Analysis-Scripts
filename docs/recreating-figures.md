# Recreating scientific figures

Use the Streamlit application (`python -m streamlit run app.py`). Scientific
figures are rendered from analyzed measurements by the application's plotting
code. No AI image generation or raster rearrangement is required.

## What to keep

| Artifact | What it restores | What else it needs |
|---|---|---|
| Saved analysis session (`.analysis-session.json`) | Experiment paths, analysis choices, selected methods, and the working figure controls | Original input files; optionally its adjacent trusted results cache |
| Results cache (`.results.pkl.gz`) | Already analyzed numeric traces and metrics for a fast reopen | Its matching session; load only trusted locally created caches |
| Exported PNG with embedded metadata | The recipe for that specific figure | The original experiment and source files, or a matching loaded session |
| `.metadata.json` / `.composer.json` | The same recipe in readable JSON | The same source requirements as the PNG |

Metadata is a recipe, not a replacement for measurements. A saved session is a
working experiment, not a guarantee that its last layout matches every export.
Open the session and then load a figure's metadata when reproducing that exact
figure. With unchanged inputs and settings the scientific values are reproducible;
identical pixels additionally require the same software revision, dependency
versions, fonts, canvas dimensions, and export DPI.

Saved sessions are local to the computer. On Windows they live in
`%LOCALAPPDATA%\Electrochemistry Analysis Scripts\analysis_sessions`. Keep JSON
recipes together with the cache named in their `results_cache` field. Recipes,
caches, source data, and generated figures are excluded from Git because they can
contain private paths and experimental data. Committing code does not upload them.

## NP figures and sessions

Final review folder:
`%USERPROFILE%\Downloads\Figures materials for the paper\Results\journal_plots\np`.
Portable local session copies are also kept in `paper/generated/np_sessions`.

| Output stem | Saved session | Application template |
|---|---|---|
| `np_sweep_type1c_cube_center` | `np_sweep_type1c` | Figure Composer → Type 1C |
| `np_validation` | `np_validation` | Figure Composer → Type 2 |
| `np_titration_type3` | `np_titration` | Paper Figures → Type 3 |
| `np_titration_type3a_signal_on_minus_off` | `np_titration_type3a` | Paper Figures → Type 3A - Cross-channel signal-on/off |
| `np_titration_type4` | `np_titration_type4` | Paper Figures → Type 4 |

The existing `np_sweep` workspace remains available for the other sweep layouts.
Type 1C retains channel 2, iterations 87 and 80, the 4 and 7 mV slice planes,
and the selected pairwise-SNR scoring profile. Scoring profiles are now recorded
in Composer metadata and saved sessions so a fresh application page restores Q.
Type 3A uses optimized channel 5 for ON, optimized channel 3 for OFF, and each
channel's own manual reference. Both fit/response rows use source measurements
41–160; response axes display 1–120 and SWV overlays include the full analyzed
range. ON-minus-OFF connects matched accepted plateaus; it is not a separately
fitted binding curve.

## Recreate from a saved session

1. Connect the source volume and start the application. Expand **Saved analysis
   sessions**, select the session above, and click **Open**.
2. Confirm the source folder. A trusted matching cache can reopen titration
   results immediately. If no cache is present, click **Run Analysis**. Restore
   missing inputs before proceeding; missing files must not be replaced by images.
3. For sweep/validation, open **Figure Composer** and click **Render figure
   (fast preview)**. For titration, open **Paper Figures** and click **Generate
   paper figure**. The saved session already selects its figure type and methods.
4. Inspect the preview. Use **150 DPI** and download **PNG** and **Metadata**.
   In Paper Figures leave **Include PDF and SVG exports** off for this review.
   Composer's full ZIP includes vector formats; download the individual PNG and
   metadata when only the review files are wanted.

If moving the sessions to another computer, copy the recipes and their named
cache files into that computer's local session directory. Update unavailable
source paths in the application. Input files must match the experiment recorded
in the figure; merely having a folder with the same name is insufficient.

## Recreate from PNG or JSON metadata

For sweep/validation:

1. Open the matching BO session and **Figure Composer**.
2. Under **Load a preset or saved figure**, upload the PNG, metadata JSON, or
   Composer JSON. Choose **Open original experiment** when its recorded path
   exists, or **Apply to active experiment** after selecting the relocated source.
3. Render and export. The recipe restores panel selections, geometry, styles,
   linked iterations/slices, and the recorded scoring profile. Changing datasets
   intentionally changes the scientific content; channel adaptation is shown.

For titration:

1. Open **Paper Figures → Load figure recipe** and upload the PNG or metadata JSON.
2. Click **Load recipe into analyzer**. This restores the source list, analysis
   settings, concentration boundaries, methods, and figure controls, and clears
   stale analyzed results.
3. Verify source paths and click **Run Analysis**. Return to **Paper Figures**,
   confirm the recorded template, then click **Generate paper figure**.
4. Save a named session, optionally including its derived cache, for future reopen.

The NP Type 3A controls are part of the application and are stored in both export
metadata and `np_titration_type3a`. Types 3B/3C retain their same-channel shared or
paired-fit layouts. Historical paper drafts sometimes used “Type 3A” for a
different paired layout; use the recorded application template for those drafts.

## Checks and troubleshooting

- A missing Type 1 template usually means an optimization session was opened
  instead of a survey. Type 2 uses optimization sessions. Types 3/3A/3B/3C/4 live
  in SWV **Paper Figures**, with titration intervals and Langmuir fitting enabled.
- A stale figure download requires **Generate paper figure** or **Render figure**
  after controls change. The application deliberately invalidates stale renders.
- Raw files are required to recalculate SWVs. PNG metadata cannot reconstruct
  missing raw samples. Keep the BO session files and their referenced trace CSVs.
- Source changes invalidate a cache when a source signature was recorded. Rerun
  analysis after changing data or processing settings.
- The current NP caches were rebuilt through the application from raw inputs;
  older batch-derived caches are archived, not used for these figures. Input
  fingerprints, source hashes, and rendering-library versions are recorded in
  `np_reproducibility_audit.json` beside the delivered figures.
- Run `python -m pytest -q -rs` for the regression suite. Checks requiring
  optional scikit-learn, Playwright WebKit, or unavailable Streamlit test APIs
  report their skip reasons separately from failures.

Repository structure: `app.py` provides the analysis UI, `bo_session_viewer.py`
provides BO views and Composer templates, `core/` contains reusable analysis and
rendering helpers, `tests/` verifies behavior, `docs/` contains general workflows,
and `paper/` records experiment-specific decisions. `paper/generated/` is local
output, not application source. Batch audit utilities in `scripts/` are optional;
the application is the supported route for recreating these figures.
The obsolete `prepare_np_paper_results.py` one-off renderer has been retired;
use the native templates above rather than a separate NP rendering script.
