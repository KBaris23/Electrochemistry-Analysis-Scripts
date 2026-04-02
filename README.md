# SWV Analysis UI

Interactive Streamlit app for batch SWV electrochemistry analysis.

## Setup

```bash
cd swv_app
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

If `streamlit` is installed globally on Windows instead of in the active virtualenv, you can also run:

```bash
py -m streamlit run app.py
```

The app opens automatically at http://localhost:8501

## Project layout

```
swv_app/
├── app.py              ← Streamlit UI (sidebar params, tabs, export)
├── requirements.txt
└── core/
    ├── io.py           ← File discovery, CSV loading, NaN filtering
    ├── processing.py   ← Smoothing, peak detection, baseline correction
    ├── analysis.py     ← Single-file analysis, partial failure traces, run_batch()
    └── plotting.py     ← All figure-returning plot functions
```

## UI tabs

| Tab | What it shows |
|-----|---------------|
| 🌈 Overlays | Colormapped raw / smoothed / corrected traces per channel |
| 📊 Metrics | Peak current, skew, wavelet energy vs scan — all channels combined |
| ⚠️ Failures | Failed trace plots + single-trace inspector |
| 🗂 Data Table | Filterable results table |
| 💾 Export | Download results.csv and a ZIP of all figures |

## Peak finding and baseline correction

For each SWV trace, the app follows this sequence:

1. Crop the raw trace to the selected voltage range.
2. Smooth the cropped current with a Savitzky-Golay filter.
3. Find the dominant peak on the smoothed trace.
4. Search for one bracketing minimum to the left of the peak and one to the right.
5. Draw a straight-line local baseline through those two minima.
6. Subtract that baseline from the smoothed trace.
7. Smooth the corrected trace again and re-detect the peak.
8. Report the corrected peak height and corrected peak voltage.

### 1. Cropping

Starting from raw voltage and current arrays:

```text
v_raw, i_raw
```

the app keeps only the points inside the crop window:

```text
v_min <= v_k <= v_max
```

which gives the cropped arrays:

```text
v = {v_k},  i = {i_k}
```

All peak finding and baseline correction are done on this cropped trace.

### 2. Smoothing

The current is smoothed with a Savitzky-Golay filter:

```text
i_smooth = SG(i)
```

Conceptually, the filter fits a low-order polynomial within a moving window. If the local polynomial is

```text
p(v) = a0 + a1*v + a2*v^2 + ... + am*v^m
```

then the smoothed value at the center of the window is:

```text
i_smooth(v_c) = p(v_c)
```

This reduces noise while preserving the peak shape better than a simple moving average.

### 3. Dominant peak detection

The app searches the smoothed trace for candidate peaks and keeps the dominant one. In practice this is the valid peak with the largest smoothed current:

```text
k_peak = argmax(i_smooth[k]) over valid detected peaks
```

If no peaks pass the prominence filters, the algorithm falls back to the global maximum:

```text
k_peak = argmax(i_smooth[k])
```

The peak voltage from this first pass is:

```text
v_peak = v[k_peak]
```

This first-pass peak is used to define where the baseline anchors should be searched.

### 4. Left and right minima search

Let the user-selected minima search window be `W = minima_search_window_V`. The algorithm defines:

```text
L = {k : v_peak - W <= v_k < v_peak}
R = {k : v_peak < v_k <= v_peak + W}
```

These are the allowed left-side and right-side search regions around the peak. The bracketing minima are then chosen as:

```text
k_L = argmin(i_smooth[k]) for k in L
k_R = argmin(i_smooth[k]) for k in R
```

The two anchor points are therefore:

```text
(v0, y0) = (v[k_L], i_smooth[k_L])
(v1, y1) = (v[k_R], i_smooth[k_R])
```

If either side has no points inside the requested voltage window, the code falls back to using all points on that side of the peak.

### 5. Local baseline from the two minima

The local baseline is the straight line through the two anchor minima. Its slope is:

```text
m = (y1 - y0) / (v1 - v0)
```

and the intercept form is:

```text
b = y0 - m*v0
```

so the baseline at any voltage `v` is:

```text
B(v) = m*v + b
```

or equivalently:

```text
B(v) = y0 + ((y1 - y0) / (v1 - v0)) * (v - v0)
```

This line represents the local background under the peak, approximated as linear between the two bracketing minima.

### 6. Baseline correction

The corrected current is calculated point-by-point by subtracting that baseline from the smoothed trace:

```text
I_corr(v) = I_smooth(v) - B(v)
```

or in index form:

```text
I_corr[k] = i_smooth[k] - B(v_k)
```

At the two anchor minima, the corrected signal is approximately zero:

```text
I_corr(v0) = 0
I_corr(v1) = 0
```

So this correction removes both:

- vertical offset
- local linear tilt

That is why the code refers to the step as a rotate/offset correction.

### 7. Final corrected peak measurement

After baseline subtraction, the corrected trace is smoothed again:

```text
I_corr_smooth = SG(I_corr)
```

The dominant peak is then re-detected on the corrected trace:

```text
k_peak,corr = argmax(I_corr_smooth[k]) over valid detected peaks
```

The final reported values are:

```text
Peak voltage  = v[k_peak,corr]
Peak current  = I_corr[k_peak,corr]
```

So the app uses the first-pass peak only to place the baseline anchors, but the final reported peak position and peak height come from the baseline-corrected trace.

### 8. Interpretation

If the measured signal is thought of as

```text
I(v) = s(v) + p(v)
```

where:

- `s(v)` is a slowly varying background or sloped baseline
- `p(v)` is the actual SWV peak

then the line through the left and right minima is used as a local estimate of `s(v)`:

```text
B(v) ~= s(v)
```

and the corrected trace becomes:

```text
I_corr(v) = I(v) - B(v) ~= p(v)
```

This works well when the local baseline is approximately linear near the peak. If the true baseline is strongly curved, some residual baseline shape may remain after correction.

### 9. Effect of `minima_search_window_V`

The parameter `minima_search_window_V` changes the allowed regions `L` and `R`:

- Smaller values force the minima to be closer to the peak, making the correction more local but also more sensitive to noise or shoulders.
- Larger values allow the minima to be farther from the peak, which can be more stable but may span a region where the true baseline is less linear.

## Using core modules directly (no UI)

```python
from core import run_batch, plot_metric_vs_scan

results = run_batch(
    folders=["/path/to/data"],
    crop_range=(-0.61, -0.30),
    smooth_window=9,
    min_start_voltage=-0.7,
)

fig = plot_metric_vs_scan(results, metric="peak_current")
fig.savefig("peak_current.png")
```
