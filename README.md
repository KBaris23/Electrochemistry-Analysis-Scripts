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
