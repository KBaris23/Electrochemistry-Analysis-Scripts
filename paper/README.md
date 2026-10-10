# Paper figures: real-data runbook

Updated 10 October 2026. Order: **sweep, BO validation, titration, concentration diagnostics**. These are preliminary figures, not publication approval. Raw data were not modified. The previous guide is retained in the generated run's `audit/paper_guide_before_run.md`.

## 1. Open the generated figures

Start with [titrations — preliminary PNGs only](generated/preliminary_journal_plots/titrations/). All refreshed titration comparisons and concentration diagnostics are together there, named by dataset/channel/type. No PDFs, SVGs or individual SWV exports are in this review folder. BO/sweep outputs remain in their dataset folders.

**What changed:** Type 3/3B SWVs use stored **smoothed + corrected current**, clipped to each trace's **final bracketing minima** inside the analysis voltage crop. The former plots included out-of-bracket tails. This is display-only: peak heights, dose statistics and fits are not recomputed or changed by the clipping.

To make these again without starting over:

1. Restart the app, then **Saved analysis sessions > Saved session > Open**: `preliminary_kana_ch3_titration`, `preliminary_amp0_analysis` or `preliminary_vanco_analysis`.
2. Choose **View > Paper Figures** and the channel/type listed below. Under **SWV processing and filtering**, keep **Corrected peak region (between minima)**. Keep extreme filtering OFF for the main figures.
3. Use **Generate paper figure > Download PNG**. Save into `paper/generated/preliminary_journal_plots/titrations/` with the dataset/channel/type in the filename. Detailed per-figure settings are in sections 6–7.

| Figure | Open PNG | Assessment |
|---|---|---|
| Kana sweep, ch3, Type 1A | [Sweep](generated/preliminary_journal_plots/kana_try/sweep/ch3/bo_111904_a0d8ad_multipanel.png) | Main landscape candidate |
| Kana BO, ch3 maximize, Type 2 | [Validation](generated/preliminary_journal_plots/kana_try/bo/ch3_maximize/bo_113013_8dd581_multipanel.png) | Matches the recovered titration channel |
| Kana ch3 ON/manual, Type 3 | [Titration](generated/preliminary_journal_plots/titrations/kana_try_ch3_type3.png) | Matches your earlier strong channel/segment |
| Kana ch5 ON/OFF, Type 3B | [Two-direction comparison](generated/preliminary_journal_plots/titrations/kana_try_ch5_type3B.png) | Cleaner two-direction example |
| Kana ch3 ON/OFF, Type 3B | [Alternative](generated/preliminary_journal_plots/titrations/kana_try_ch3_type3B.png) | OFF saturation less convincing than ch5 |
| Kana ch5 ON, Type 4 | [Calibration diagnostic](generated/preliminary_journal_plots/titrations/kana_try_ch5_type4_calibration_diagnostic.png) | Not independent prediction validation |
| Amp0 BO, ch6 minimize | [Validation](generated/preliminary_journal_plots/amp0/bo/ch6_minimize/bo_145633_9f5760_multipanel.png) | Alternative to the earlier ch10 choice |
| Amp0 ch6 OFF/manual, Type 3 | [Titration](generated/preliminary_journal_plots/titrations/amp0_ch6_type3.png) | Detectability example; weak calibration |
| Amp0 ch10 OFF/manual, Type 3 | [Original candidate](generated/preliminary_journal_plots/titrations/amp0_ch10_type3.png) | Retained, not a strong Langmuir result |
| Amp0 ch6, Type 4 | [Diagnostic only](generated/preliminary_journal_plots/titrations/amp0_ch6_type4_calibration_diagnostic.png) | Not recommended as a paper prediction result |
| Vanco BO, ch6 minimize | [Validation](generated/preliminary_journal_plots/vanco/bo/ch6_minimize/bo_142117_84662a_multipanel.png) | Physical ch6 is BO **Group 4** |
| Vanco ch6 OFF/manual, Type 3 | [Titration](generated/preliminary_journal_plots/titrations/vanco_ch6_type3.png) | Modest response, usable fit |
| Vanco ch2 OFF/manual, Type 3 | [Alternative](generated/preliminary_journal_plots/titrations/vanco_ch2_type3.png) | Higher full-series R² |
| Vanco ch6, Type 4 | [Calibration diagnostic](generated/preliminary_journal_plots/titrations/vanco_ch6_type4_calibration_diagnostic.png) | Same-calibration inversion only |
| Kana station 2 BO, ch10 maximize | [Replicate/SI](generated/preliminary_journal_plots/kana_st2/bo/ch10_maximize/bo_171120_dc5dbc_multipanel.png) | Separate run, not pooled with setup 5 |

Amp0 ch10 BO is also retained under `amp0/bo/ch10_minimize/`.

Refreshed titration settings and widget actions are in `audit/titration_png_refresh/`. Older titration exports, tables, markers and settings are preserved under `audit/previous_titration_exports/<dataset>/`; they are not the current PNGs. BO figure folders retain their `ui_settings.json`.

Filtering, simply:

- **Main PNGs:** extreme filtering OFF; no interpolation, missing-dose patching or moving-average replacement.
- **Separate sensitivity check:** `titrations/kana_try_filtered_sensitivity_ch3_type3.png`, extreme filtering ON. The response plot and fits now honor the same exclusions. Original measurements remain intact.
- **Smoothing:** applies to each voltage/current SWV, using the recorded analysis settings (15-point, order 2), not across concentration steps. Cropping never invents a missing waveform or baseline bracket.

The original run's `audit/output_manifest.json` records its original paths/hashes; archived titration paths now map to `audit/previous_titration_exports/<dataset>/`. The refreshed PNG inventory is `audit/titration_png_refresh/manifest.json`. Six older intermediate exports also remain in `audit/stale_ui_exports/`; use the figure links above, not those archives.

### How these were made

The actual `app.py` was operated through **Streamlit AppTest widgets**: folder controls, Run Analysis, preset loading, Generate/Render, and the app's download functions. No independent numerical pipeline or custom figure generator replaced the application. There was no browser-driving tool available: this was an automated application-widget run, not manual browser clicking. Initial actions are in `audit/widget_actions.jsonl`; the PNG refresh uses `regenerate_titration_pngs.py` to reopen the saved app sessions, select the settings below, click Generate, and collect the app's PNG download bytes. Refresh actions/settings are in `audit/titration_png_refresh/`. Reproduce the same operations below in your browser; still inspect final exports at publication size.

## 2. Launch and save

```powershell
cd "C:\Users\Asus\OneDrive\Desktop\Jun-Chau Lab\Chien Lab Scripts\Analysis Scripts\swv_app"
.\Open_SWV_App.cmd
```

Restart the app process once after these code updates so the new QC functions load.

Save with sidebar **Saved analysis sessions > Save as > Save**. Recipes live in app-relative `analysis_sessions/`. **Include derived analysis cache for fast reopen** is optional: it adds tens of MB, not a copy of the raw experiment. Named preliminary kana/amp0/vanco sessions were saved. Choose one under **Saved session > Open**; regenerate figures after reopening. Without a cache, reload inputs and Run Analysis.

Raw paths must remain accessible. Moving a recipe alone does not make its data portable. Use the app's Open control; do not assume a Windows double-click association or save prompt when closing the browser. Keep a named save, not only autosaved recovery.

## 3. Figure A: planar-kana sweep

1. Select **Analysis mode > BO Session**. Paste this in **Experiment or session folder**:

   ```text
   C:\TEMP\BO\500um_planar_BO_try_again_20260918_112055\parameter_sweep_20260921_110003\bo_sessions\bo_111904_a0d8ad
   ```

2. Select `bo_111904_a0d8ad`, not the empty `bo_111849_1099a5`. Keep **Original recorded Q**. Set **Channel groups: Group 3**.
3. Open **Figure Composer > Load a preset or saved figure**. Select **Type 1A - Sweep comparison (cube left)**, wait for the selection rerun, then **Load preset**.
4. Keep shared channel **3**, automatic ON/OFF examples and **two step slices: 4 and 7 mV**. This export uses ON iteration **174** (500 Hz, 40 mV amplitude, 2 mV step) and OFF **184** (108 Hz, 60 mV amplitude, 6 mV step). These are sweep examples, not the later titration methods.
5. Keep preset layout/camera/fonts. Traces are **Raw / unsmoothed (corrected)**, with snapshot correction and the selected peak-bracket crop. No Q clipping or exclusions were applied.
6. Click **Render figure (fast preview)**, inspect, then the final-export button. Download PNG/PDF/SVG and the recipe/package into `kana_try/sweep/ch3/`.

For exact formatting, upload the saved `.figure.json` or `.figure.zip` after loading this session. **Edit panel** changes content, camera, markers, text and layout; interactive camera changes must be cached/applied. Type 1B mirrors left/right. Type 1 is the larger four-slice layout. Sweep presets require a survey session, not an optimization session.

## 4. Figure B: BO validation for each run

**Kana setup 5**

```text
C:\TEMP\BO\500um_planar_BO_try_again_20260918_112055\planar_BO_kana_20260918_112056\bo_sessions\bo_113013_8dd581
```

**Amp0**

```text
C:\TEMP\BO\100um_amp0_high_conc_20260915_130213\amp0_100um_20260915_130234\bo_sessions\bo_145633_9f5760
```

**Vanco**

```text
C:\TEMP\BO\vanco_first_try_20260826_132944\vanco_try_1_20260826_132945\bo_sessions\bo_142117_84662a
```

**Kana station 2**

```text
C:\TEMP\BO\500um_planar_kana_20260917_170330\500um_planar_kana_20260917_170349\bo_sessions\bo_171120_dc5dbc
```

| Run | Sidebar group | Composer direction | Linked channel |
|---|---:|---|---|
| Kana setup 5 | 3 | maximize | `3_max` |
| Amp0 preferred preliminary | 6 | minimize | `6_min` |
| Amp0 earlier candidate | 10 | minimize | `10_min` |
| Vanco | 4 | minimize | `6_min` |
| Kana station 2 | 10 | maximize | `10_max` |

For each row:

1. Paste its folder in **BO Session**, select its group, keep **Original recorded Q**.
2. In **Figure Composer**, select **Type 2 - BO validation**, then **Load preset**.
3. Set **Optimizer shown in BO panels** to the table's direction. Check linked group/channel across Q, paired metrics, cube, stack and parallel coordinates; do not leave Q on all groups.
4. Keep the preset: all 50 iterations, dashed **5-point trailing mean**, best-so-far line and best-observed marker; buffer/target **Peak prominence**; snapshot-corrected unsmoothed chronological traces, no normalization; maximum displayed traces 120, line width 1.5, fading toward the back. Display sampling does not remove analysis observations.
5. Render preview, final export, and save in `dataset/bo/channel_direction/`. Saved figure JSON records the exact camera and formatting.

Type **2A** is the focused alternative, omitting stack/parallel coordinates. Optimization inputs offer 2/2A; survey inputs offer 1/1A/1B. Types 3/3B/4 belong to SWV mode.

Peak prominence is noise-normalized, not current in microamps. Recorded `Q_run` can differ from the cube's paired-Q measure because of objective aggregation/penalties. Larger raw peaks do not necessarily mean better paired SNR. Best-so-far necessarily improves by construction; it does not establish superiority to random search without a matched benchmark.

## 5. Figure C: prepare titration in SWV mode

| Dataset | Folders input | Scans analyzed | Per-method display interval |
|---|---|---:|---|
| Kana setup 5 | `C:\TEMP\BO\titration_only\kana_try` | 6,000 | 21-200 |
| Amp0 | `C:\TEMP\BO\titration_only\amp0` | 4,800 | 21-160 |
| Vanco | `C:\TEMP\BO\titration_only\vanco` | 4,800 | 21-200 |

1. Select **Analysis mode > SWV** and replace **Folders** with one input above, not all three. These inputs omit warm-up/BO scans.
2. Leave automatic BO-settings loading enabled. Verify the snapshot path belongs to the corresponding BO session in section 4. For extracted inputs the app resolves the original run; normal experiment folders resolve their associated session. If ambiguous, explicitly select that session's `bo_config_snapshot.json` and load it. Use the settings-check control before overriding anything.
3. Choose **Grouping > SWV settings**, **SWV peak height source > Corrected + smoothed**, all physical channels. Click **Run Analysis** once and wait.
4. Enable **Treat vline intervals as titration steps** and **Fit Langmuir**. Check the queue-derived vlines. If absent, paste the matching app-exported [kana markers](generated/preliminary_journal_plots/audit/previous_titration_exports/kana_try/vlines.txt), [amp0 markers](generated/preliminary_journal_plots/audit/previous_titration_exports/amp0/vlines.txt) or [vanco markers](generated/preliminary_journal_plots/audit/previous_titration_exports/vanco/vlines.txt). These are preserved under `audit/previous_titration_exports/<dataset>/vlines.txt`; the old `C:\TEMP\BO\analysis_plan\vlines_*.txt` files are no longer present.
5. Set **Titration baseline mode > Immediately preceding buffer**, units **uM**, **Plateau edge trim fraction = 0.15**.
6. Under **Concentrations included in titration statistics**, deselect only `buffer_1` and the first duplicate low-dose block (`20 uM_1`, `1000 uM_1` or `0.2 uM_1`). Keep the second low-dose block and all later buffers/doses. This treats the first pair as equilibration per the supplied design; confirm the rationale in the notebook.
7. Leave **Remove extreme titration outliers OFF**. LOD/ULOQ display annotations were OFF; numerical estimates still export. Click **Apply Display Controls**.
8. Existing CSVs/quality checks are preserved under `audit/previous_titration_exports/<dataset>/unfiltered/` and `audit/`. You need not regenerate these to review the PNGs. For new analysis settings, use **Export** for updated CSVs and **Quality audit** for waveform flags/exclusion candidates; keep them outside the PNG review folder.

### Exact analysis settings

Kana setup 5 and amp0 share these settings; vanco differs in minima requirements:

| Control | Value |
|---|---|
| Crop | -0.55 to 0.00 V |
| Smoothing | window 15, polynomial order 2 |
| Minima search window | 0.30 V |
| Minimum start voltage | -0.60 V |
| Minimum peak height | 0.001 uA |
| Double correction | ON |
| Prominent minima / require local minima both sides | ON / ON for kana and amp0; OFF / OFF for vanco |
| Accepted peak voltage | -0.45 to -0.10 V |
| Left minimum acceptance | -0.54 to -0.10 V |
| Right minimum acceptance | -0.45 to -0.01 V |
| Apply BO acceptance windows | ON |
| Wavelet denoising / wavelet correction | OFF / OFF |

Station 2 BO uses its own snapshot: crop -0.45 to +0.10 V, minimum start -0.50 V and different acceptance windows. Do not overwrite it with this table.

Vlines use **physical-channel acquisition scans**: `1,buffer; 31,lowest dose; 61,buffer; 91,lowest dose again; 121,buffer; 151,next dose; ...`. End markers: 601 kana/vanco, 481 amp0. Grouping maps these to method-local 1,11,21,31,...; **do not divide pasted vlines yourself**. Doses double: kana 20-5,120 uM, amp0 1,000-64,000 uM, vanco 0.2-51.2 uM.

Display start 21 removes the first 20 **method** scans (initial buffer/target pair), not 20 raw files. A display crop does not alter fit inclusion. Selected doses feed the existing hybrid fit: Langmuir through the largest absolute response, with later points retained and potentially a polynomial guide. Export now distinguishes R² on that fitted branch from R² across all selected accepted targets. Do not report one as the other.

## 6. Figure C: generate Type 3 or 3B

Open **SWV > Paper Figures**. Common settings: **14 in width, 10 pt font, panel raster DPI 300, Stacked (offset) SWVs, stride 10, offset 0.35, Overlaid response curves, no extra columns, panel letters OFF**. Set **SWV processing and filtering > SWV voltage window > Corrected peak region (between minima)**. All waveform panels use smoothed + corrected current; bounds come from the final correction pass. PNG exports at 600 DPI; constituent panels retain their selected raster DPI. Use **Download PNG** only for this preliminary round.

### Kana ch3: recovered earlier channel, Type 3

Select **Type 3 - SWV and titration response**, one row, physical **ch3**, optimized **500 Hz / 50 mV / 1 mV (maximize)**, manual **200 Hz / 36 mV / 2 mV**, display **21-200**. Click **Generate paper figure > Download PNG**. Save as `titrations/kana_try_ch3_type3.png`.

Your older SVG under `C:\Users\Asus\Downloads\Figures materials for the paper\08_Q_scoring_scheme` matches this method/segment. Its 179 values match current app-export values essentially exactly (squared correlation approximately 1). It omits method scan **189**, physical scan **565**, which is currently accepted but flagged by the existing extreme filter. Primary figures retain it. A separate filtered Type 3 is in `titrations/kana_try_filtered_sensitivity_ch3_type3.png`; no replacement point was synthesized. The original exclusion rationale cannot be proved from the SVG alone.

The adjacent `resources/conceptual_titration.py` creates synthetic data and was **not** used as experimental evidence.

### Kana ON/OFF: Type 3B

Select **Type 3B - Signal-on/off shared Langmuir**, physical **ch5**, ON **500/40/2**, OFF **138/90/3**, manual **200/36/2** (Hz/mV/mV), display **21-200**. Generate/download PNG as `titrations/kana_try_ch5_type3B.png`.

Ch3 alternative: ON **500/50/1**, OFF **125/70/8**, same manual. Type 3B has two rows and one shared final column containing optimized ON, OFF and matched-dose ON-minus-OFF. The difference is descriptive, **not a third independently fitted binding curve**. Manual SWVs/time courses remain in the first three columns, not the shared Langmuir panel.

### Amp0: Type 3 detectability comparison

Use **ch6**, optimized **347 Hz / 70 mV / 10 mV (minimize)** versus manual **200/36/2**, display **21-160**. Generate/download PNG as `titrations/amp0_ch6_type3.png`.

Earlier ch10 candidate: optimized **362/80/9 (minimize)** versus manual, same interval; retained as `titrations/amp0_ch10_type3.png`. Neither establishes robust full-range calibration. Many manual scans and some high-dose optimized blocks fail detection; these were not filled. Do not force an ON/OFF layout when both directions are not supported by the data.

### Vanco: Type 3

Use **ch6**, optimized **401/70/5 (minimize)** versus manual **200/36/2**, display **21-200**. Save PNG as `titrations/vanco_ch6_type3.png`.

Alternative **ch2**: optimized **322/60/4 (minimize)** versus manual, same interval; `titrations/vanco_ch2_type3.png`. Chip `350_r_np_45_15` is recorded; confirm what `np` means before labeling it planar/nanoporous.

## 7. Figure D: Type 4 diagnostics

Keep the same channel/method/step selection, choose **Type 4 - Concentration validation**, Generate, download. Examples use kana **ch5 ON**, amp0 **ch6 OFF**, vanco **ch6 OFF**, each against same-channel manual with the intervals above. Save PNG as `titrations/<dataset>_ch<channel>_type4_calibration_diagnostic.png`.

These invert a calibration fitted to these data: **not held-out validation**. Inversion is unstable near saturation; unavailable predictions remain missing. Amp0 Type 4 is an audit output, not a recommended paper result. Do not hide failed predictions or insert averaged concentrations into accuracy statistics.

## 8. Findings and limitations

Current app results below: unfiltered, preceding-buffer correction, initial equilibration pair excluded. R² evaluates the Langmuir model across **all selected accepted target doses**, not only its fitted branch. SNR is the largest dose-level value in the exported table, not BO Q or performance at every dose.

| Method | Full-series R² | App Kd, uM | App LOD, uM | Max titration SNR |
|---|---:|---:|---:|---:|
| Kana ch3 ON | 0.9846 | 265.0 | 9.98 | 78.8 |
| Kana ch3 manual | 0.5640 | 136.6 | 9.67 | 33.2 |
| Kana ch5 ON | 0.9858 | 422.7 | 25.52 | 48.0 |
| Kana ch5 OFF | 0.9742 | 655.7 | 78.85 | 23.0 |
| Kana ch5 manual | 0.9573 | 166.8 | 8.61 | 53.7 |
| Amp0 ch6 OFF | 0.6808 | Not reliable | Not recommended | 10.6 |
| Amp0 ch10 OFF | -367.7 | Not reliable | Not recommended | 3.3 |
| Vanco ch2 OFF | 0.9719 | 0.384 | 0.187 | 6.25 |
| Vanco ch2 manual | 0.8815 | 0.719 | 0.459 | 5.51 |
| Vanco ch6 OFF | 0.9319 | 0.871 | 0.350 | 8.06 |
| Vanco ch6 manual | 0.7857 | 0.440 | 0.406 | 4.03 |

Kd/LOD are model outputs, not independently established constants/limits. Some manual fits stop before later selected doses. Amp0 ch10's two-dose fitted branch has R² 0.7213 but extrapolates badly to later selected doses; its very negative full-series R² is not a typo. Ch6's fitted-branch R² is 0.9148, not its full-series value. The earlier blanket amp0 R² claim and vanco's alleged 1 uM Kd floor were not reproduced.

Defensible preliminary storyline: strong kana concentration dependence and selected-method advantages; amp0 detectability rescue and modest vanco improvements as qualified secondary results. There is **no universal LOD/SNR advantage**: kana ch5 manual has lower fitted LOD and higher maximum SNR than optimized ON. Preserve paired comparisons and all-channel tables. These figures alone do not prove universal aptamer performance, specificity, reversibility or BO superiority to random search.

### Waveform audit

Entire inputs, all methods, before display crops/outlier filtering:

| Dataset | Total | Analysis failures | Jump-screen candidates |
|---|---:|---:|---:|
| Kana setup 5 | 6,000 | 1,029 | 857 |
| Amp0 | 4,800 | 1,363 | 455 |
| Vanco | 4,800 | 214 | 1 |

Jump candidates are screening hints, not confirmed artifacts or exclusions: internal adjacent-current change above 10 robust derivative SD and 15% of waveform range. Sharp peaks, crop edges and gaps can trigger them. Use **Quality audit > Inspect analysed waveform > Preview waveform QC** to compare stored raw/corrected/smoothed arrays and download. Screening covers the analysis crop, not voltages outside it.

- Kana ch3: ON/manual each 200/200 accepted, OFF 199/200; its one jump candidate is near the crop edge, physical scan 421. Ch5 ON 194/200, OFF/manual each 200/200. Ch9 failed 553/600, **not 100%** as previously summarized. Other strong ON candidates remain ch1,4,7,8,10; full-channel exports preserve the selection context.
- Amp0 ch6: OFF 131/160 accepted, manual 20/160. Ch10: OFF 119/160, manual 15/160. These counts do not establish binding specificity. Raw changes near -0.43 to -0.44 V are visible in saved QC previews and less obvious after smoothing/correction. Some high-dose method blocks lack accepted plateaus.
- Vanco ch6: all 600 accepted; ch2 OFF/manual each 200/200. The sole jump candidate is ch10 physical scan 155. Its raw CSV jumps from -0.470662 to -0.343809 V (0.126853 V versus typical 0.001186 V increments). A connecting line spans missing voltage samples; no intermediate measurements were created. Cause unknown. This is not on chosen ch2/ch6 channels.
- `C:\TEMP\BO\vanco_500um_planar_20260908_113415` currently contains metadata/logs but no local raw scans. Its earlier claimed poor response was not re-established, nor was it substituted for the available run.

### Exclusions and sensitivity analysis

Primary figures: initial buffer/duplicate-dose pair omitted from titration statistics; original acceptance rules retained. **No new jump-based exclusions, Q clipping or moving-average gap filling.** BO's five-point mean is a trend line, not replacement observations. SWV stride 10 is display sampling.

For the sensitivity branch: **Remove extreme titration outliers ON > Apply Display Controls**, regenerate and export separately. Candidate rows and retained statistics are preserved under `audit/previous_titration_exports/<dataset>/audit/` and `filtered_sensitivity/`. Do not silently switch primary analysis to that branch. Exclusions need not be printed inside the image, but rules/counts must remain available in Methods/SI and the audit.

## 9. Templates, checks and remaining work

- Built-ins: `bo_session_viewer.py` functions `_paper_parameter_sweep_*` and `_paper_bo_validation_*`; SWV Types 3/3B/4 are in `app.py` Paper Figures. Visible, tracked `figure_composer_presets.json` stores **custom saved presets**, not every built-in.
- Composer **Save preset** saves a template; `.figure.json` saves a composition; analysis sessions restore inputs/settings/optional derived results. They are different objects.
- To update a custom template: apply pending panel/layout edits, then **Reusable preset > Saved preset to overwrite > Overwrite selected preset**. No name retyping or rendering needed. Save a named copy of a built-in once before using overwrite; built-in definitions stay unchanged.
- Layout boxes include labels/legends; grid defaults to 1%, overlap is optional, export crops unused outer workspace. Workspace zoom does not alter export scale. Use the generated JSON for exact formatting.
- This run fixed session restoration of button events/JSON identity keys, final-export invalidation, extracted-folder direction lookup, stale Paper Figures downloads, mislabeled response axes and inconsistent method colors. Added app-visible raw/corrected QC and fitted-branch/full-series R² exports. Correction/fitting algorithms were not replaced.
- Refresh fixes: smoothed/corrected waveform clipping to final correction minima; one clearly labeled filter status; consistent extreme filtering in response plots and fits; selected-preset overwrite without name retyping. Full-analysis-crop display remains available explicitly.
- Automated tests: **420 passed, 4 skipped** (one existing Matplotlib open-figure warning in tests). Real-input widget generation and saved-session reopening were exercised. Browser/print-size inspection and scientific validation remain separate checks.
- Generated outputs/local caches remain Git-ignored; source changes and this guide are version-controlled. Committing the code does not upload raw data or these local PNGs.

Still needed: notebook immobilization/cleaning details, chip morphology confirmation and equilibration rationale; independent concentration validation, specificity controls, reversibility assessment and the planned BO-versus-random simulations for stronger claims.
