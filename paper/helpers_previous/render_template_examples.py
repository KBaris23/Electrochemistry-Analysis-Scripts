"""Render one example of every Composer template from the real data, headlessly.

Drives the actual Streamlit app (SWV mode -> Add to Composer -> BO Session
mode -> load template -> Render figure) and writes the PNGs to ./examples.
Run with the swv_app virtualenv:  python render_template_examples.py
"""
import os
import re
import sys
import traceback
from pathlib import Path

APP = r"C:\Users\Asus\OneDrive\Desktop\Jun-Chau Lab\Chien Lab Scripts\Analysis Scripts\swv_app"
BO = Path(r"C:\TEMP\BO")
OUT = Path(__file__).resolve().parent / "examples"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, APP)
os.chdir(APP)
os.environ.setdefault("MPLBACKEND", "Agg")

from streamlit.testing.v1 import AppTest  # noqa: E402

KANA = BO / "500um_planar_BO_try_again_20260918_112055"
SESSIONS = {
    "bo_main": KANA / "planar_BO_kana_20260918_112056" / "bo_sessions" / "bo_113013_8dd581",
    "bo_compact": BO / "100um_amp0_high_conc_20260915_130213" / "amp0_100um_20260915_130234" / "bo_sessions" / "bo_145633_9f5760",
    "landscape": KANA / "parameter_sweep_20260921_110003" / "bo_sessions" / "bo_111904_a0d8ad",
}
CHANNEL = 5
LOG = []


def log(message):
    LOG.append(str(message))
    print(message, flush=True)


def errors(at):
    return [e.value[:300] for e in at.exception]


def click(at, button):
    button.click().run()
    if at.exception:
        log(f"  ! exception after click: {errors(at)}")


def add_buttons(at):
    return {b.key: b for b in at.button if b.label == "Add to Composer"}


def capture(at, prefix, label):
    matches = [key for key in add_buttons(at) if key.startswith(prefix)]
    if not matches:
        log(f"  ! no Add to Composer button for {label} (prefix {prefix!r})")
        return False
    click(at, add_buttons(at)[matches[0]])
    log(f"  captured {label}: {matches[0][:90]}")
    return True


def set_view(at, view):
    at.radio(key="analysis_view").set_value(view).run()


def swv_setup(at):
    at.run()
    at.session_state["folders"] = [str(BO / "titration_only" / "kana_try")]
    at.run()
    at.selectbox(key="swv_grouping_mode").select("SWV settings").run()
    at.text_input(key="swv_bo_config_path").set_value(
        str(KANA / "planar_BO_kana_20260918_112056")
    ).run()
    next(b for b in at.button if b.key == "swv_bo_config_load").click().run()
    next(b for b in at.button if "Run Analysis" in b.label).click().run()
    at.checkbox(key="swv_enable_titration_analysis").check().run()
    at.checkbox(key="swv_fit_titration_langmuir").check().run()
    at.checkbox(key="swv_show_titration_lod").check().run()
    at.selectbox(key="swv_titration_baseline_mode").select("Immediately preceding buffer").run()
    included = at.multiselect(key="swv_titration_included_step_labels")
    included.set_value([o for o in included.options if o != "20 uM_1"]).run()
    next(b for b in at.button if "Apply Display Controls" in b.label).click().run()
    log(f"SWV ready, exceptions: {errors(at)}")


def show_channels(at, channels):
    box = next(t for t in at.text_input if t.label.startswith("Channels to plot"))
    box.set_value(channels).run()
    log(f"  showing channels {channels}; exceptions {errors(at)}")


def capture_main_titration(at):
    show_channels(at, str(CHANNEL))
    set_view(at, "Metrics")
    combine = at.multiselect(key="metric_combined_channels")
    wanted = [o for o in combine.options if re.match(rf"Channel {CHANNEL}\b", str(o))]
    log(f"  channels-to-combine options (first 6): {[str(o) for o in combine.options][:6]}")
    log(f"  selected for channel {CHANNEL}: {[str(o) for o in wanted]}")
    if wanted:
        combine.set_value(wanted).run()
    trace_keys = [k for k in add_buttons(at) if not k.startswith("titration_")]
    log(f"  non-titration Add buttons in Metrics view: {[k[:70] for k in trace_keys][:8]}")
    # Every click rebuilds the page, so trigger all buttons of a view in ONE rerun.
    capture_batch(at, [
        (trace_keys[0] if trace_keys else "metric_combined", "A trace"),
        (f"titration_langmuir_channel_{CHANNEL}_", "B Langmuir"),
        # Only one channel is displayed, so the group suffix of the key varies.
        ("titration_snr_plot_peak_current_selected_Per channel", "C SNR"),
        ("titration_accuracy_plot_peak_current_selected_Per channel", "D predicted vs known"),
    ])
    set_view(at, "Overlays")
    capture_batch(at, [
        (f"swv_overlay_{CHANNEL} group {group} |", label)
        for group, label in ((1, "E overlay ON"), (2, "F overlay OFF"), (3, "G overlay manual"))
    ])


def capture_batch(at, specs):
    triggered = []
    buttons = add_buttons(at)
    for prefix, label in specs:
        matches = [key for key in buttons if key.startswith(prefix)]
        if not matches:
            log(f"  ! no Add to Composer button for {label} (prefix {prefix!r})")
            continue
        buttons[matches[0]].click()
        triggered.append(label)
    at.run()
    registry = at.session_state["bo_composer_captured_plots"] if "bo_composer_captured_plots" in at.session_state else {}
    order = [(v.get("seq"), v.get("label")) for v in registry.values()]
    log(f"  triggered {triggered}; registry (seq,label): {sorted(order, key=lambda x: x[0] or 0)[-len(triggered) or None:]}")
    if at.exception:
        log(f"  ! exception after batch: {errors(at)}")


def capture_si_langmuir(at, channels):
    show_channels(at, ",".join(str(c) for c in channels))
    set_view(at, "Metrics")
    capture_batch(at, [(f"titration_langmuir_channel_{c}_", f"Langmuir ch{c}") for c in channels])


def mode_radio(at):
    return next(r for r in at.radio if r.label == "Analysis mode")


CURRENT_SESSION = [""]
CURRENT_GROUP = ["all"]


def to_bo(at, session_folder, group=None):
    CURRENT_GROUP[0] = "all" if group is None else str(group)
    # The app's render key uses the session id (bo_<date>_<time>_<hash>); the
    # folder is bo_<time>_<hash>, so match on the hash they share.
    CURRENT_SESSION[0] = session_folder.name.split("_")[-1]
    mode_radio(at).set_value("BO Session").run()
    at.text_input(key="bo_session_folder").set_value(str(session_folder)).run()
    if group is not None:
        at.selectbox(key="bo_channel_group_scope").select(group).run()
    log(f"BO session {session_folder.name} (group {group}): exceptions {errors(at)}")


def to_swv(at):
    mode_radio(at).set_value("SWV").run()


def render_template(at, preset, out_name):
    select = at.selectbox(key="bo_composer_preset_select")
    if preset not in select.options:
        log(f"  ! preset {preset!r} not offered; options: {list(select.options)}")
        return False
    select.select(preset).run()
    load = next(b for b in at.button if b.key == "bo_composer_preset_load")
    if load.disabled:
        log(f"  ! preset {preset!r} cannot load for this session: "
            f"{[e.value[:300] for e in at.error]}")
        return False
    load.click().run()
    state = at.session_state
    log(f"  after loading {preset!r}: layout={state['bo_composer_layout'] if 'bo_composer_layout' in state else None}, "
        f"count={state['bo_composer_count'] if 'bo_composer_count' in state else None}")
    next(b for b in at.button if b.key == "bo_composer_render_button").click().run()
    # Several renders can coexist (automatic renders when captures are added,
    # one per group scope): take the newest, i.e. the last one stored.
    payloads = [
        v for k, v in at.session_state.filtered_state.items()
        if str(k).startswith("bo_composer_render_") and isinstance(v, dict) and v.get("png")
        and CURRENT_SESSION[0] in str(k)
        and str(k).endswith(f"::{CURRENT_GROUP[0]}")
    ]
    payload = payloads[-1] if payloads else None
    log(f"  {preset}: errors={[e.value[:300] for e in at.error]} exceptions={errors(at)}")
    if payload is None:
        log(f"  ! nothing rendered for {preset}")
        return False
    (OUT / out_name).write_bytes(payload["png"])
    for fmt in ("pdf", "svg"):
        if payload.get(fmt):
            (OUT / out_name.replace(".png", f".{fmt}")).write_bytes(payload[fmt])
    log(f"  wrote {OUT / out_name}")
    return True


def guarded(name, function, *args):
    try:
        return function(*args)
    except Exception:
        log(f"!! {name} failed:\n{traceback.format_exc()[-1500:]}")
        return False


def main():
    """Usage: render_template_examples.py [titration | si | bo | all]

    titration: only the 7-panel titration figure (needs the SWV analysis, ~10 min)
    si:        titration main + the SI Langmuir grid
    bo:        the three BO-session templates (no SWV analysis needed)
    all:       everything
    """
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    at = AppTest.from_file(APP + r"\app.py", default_timeout=3000)
    at.run()
    if stage in ("titration", "si", "all"):
        guarded("swv setup", swv_setup, at)
        guarded("capture titration main", capture_main_titration, at)
        guarded("to_bo", to_bo, at, SESSIONS["bo_main"], CHANNEL)
        guarded("titration main", render_template, at, "Titration main (7 panels)", "titration_main.png")
    if stage in ("si", "all"):
        guarded("to_swv", to_swv, at)
        guarded("capture SI", capture_si_langmuir, at, [1, 3, 4, 5, 7, 8, 10])
        guarded("to_bo", to_bo, at, SESSIONS["bo_main"])
        guarded("titration si", render_template, at, "Titration SI grid (9 panels)", "titration_si_grid.png")
    if stage in ("bo", "all"):
        guarded("to_bo", to_bo, at, SESSIONS["bo_main"], CHANNEL)
        guarded("bo main", render_template, at, "BO optimization (main)", "bo_optimization_main.png")
        guarded("to_bo compact", to_bo, at, SESSIONS["bo_compact"], 10)
        guarded("bo compact", render_template, at, "BO compact (SI 2x2)", "bo_compact_si.png")
    if stage in ("bo", "landscape", "all"):
        guarded("to_bo landscape", to_bo, at, SESSIONS["landscape"], "all")
        guarded("landscape", render_template, at, "Landscape (3D + traces + 2D slices)", "landscape.png")
    (OUT / f"render_log_{stage}.txt").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
