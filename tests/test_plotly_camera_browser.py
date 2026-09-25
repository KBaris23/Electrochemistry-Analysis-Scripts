"""Browser regressions; requires Playwright and Chromium (or Google Chrome)."""
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

playwright = pytest.importorskip("playwright.sync_api")
COMPONENT = Path(__file__).resolve().parents[1] / ".streamlit_components/plotly_camera_capture/index.html"


@pytest.fixture(params=["chromium", "webkit"])
def page(request):
    with playwright.sync_playwright() as p:
        options = {"headless": True, "args": ["--max-active-webgl-contexts=4"]}
        try:
            browser = (p.webkit.launch(headless=True) if request.param == "webkit"
                       else p.chromium.launch(**options))
        except playwright.Error:
            if request.param == "webkit":
                pytest.skip("Install Playwright WebKit")
            try:
                browser = p.chromium.launch(channel="chrome", **options)
            except playwright.Error:
                pytest.skip("Install Playwright Chromium or Google Chrome")
        page = browser.new_page(viewport={"width": 1000, "height": 700})
        yield page
        browser.close()


def render_plot(frame, storage_key="test-camera"):
    frame.evaluate("""storageKey => render({
        figure: {data: [{type: 'scatter3d', mode: 'markers',
            x: [1,2,3], y: [3,1,2], z: [1,3,2],
            marker: {size: 15, color: 'red'}}], layout: {}},
        height: 500, show_download: true, storage_key: storageKey,
        download_width: 800, download_height: 500,
        download_file_stem: 'tensor'
    })""", storage_key)


def wait_for_scene(frame):
    frame.wait_for_function("""graph._fullLayout && graph._fullLayout.scene &&
        graph._fullLayout.scene._scene && !sceneContextLost()""")
    frame.evaluate("renderPromise")


def assert_download_has_points(page, frame, tmp_path):
    with page.expect_download() as download:
        frame.locator("#download").click()
    target = tmp_path / "tensor.png"
    download.value.save_as(target)
    pixels = np.asarray(Image.open(BytesIO(target.read_bytes())).convert("RGB"))
    assert pixels.shape == (1000, 1600, 3)
    # Exclude SVG legends and colorbars: they can survive a blank 3D layer.
    pixels = pixels[200:850, 200:1100]
    red = (pixels[:, :, 0] > 180) & (pixels[:, :, 1] < 80) & (pixels[:, :, 2] < 80)
    assert red.sum() > 1000, "Export lost its WebGL data points"


def test_many_channel_plots_release_contexts_and_export_after_scrolling(page, tmp_path):
    host = tmp_path / "channels.html"
    host.write_text("".join(
        f'<iframe src="{COMPONENT.as_uri()}" style="display:block;width:950px;height:560px"></iframe>'
        for _ in range(12)
    ))
    page.goto(host.as_uri())
    page.wait_for_function("document.querySelectorAll('iframe').length === 12")
    frames = page.frames[1:]
    for index, frame in enumerate(frames):
        frame.wait_for_function("typeof render === 'function'")
        render_plot(frame, f"channel-{index}")
    first = frames[0]
    wait_for_scene(first)
    camera = {"eye": {"x": 2, "y": -1, "z": 0.7}}
    first.evaluate("camera => Plotly.relayout(graph, {'scene.camera': camera})", camera)
    for index in (5, 11, 0):
        page.locator("iframe").nth(index).scroll_into_view_if_needed()
        wait_for_scene(frames[index])
    # Previously visited plots must relinquish their context slots.
    frames[11].wait_for_function("!graph._fullLayout")
    assert first.evaluate("renderedCamera().eye") == pytest.approx(camera["eye"])
    size_before = first.evaluate("[graph._fullLayout.width, graph._fullLayout.height]")
    assert_download_has_points(page, first, tmp_path)
    assert first.evaluate("renderedCamera().eye") == pytest.approx(camera["eye"])
    assert first.evaluate("[graph._fullLayout.width, graph._fullLayout.height]") == size_before


def test_lost_scene_is_rebuilt_and_downloadable(page, tmp_path):
    page.goto(COMPONENT.as_uri())
    render_plot(page)
    wait_for_scene(page)
    page.evaluate("""() => {
        window.oldScene = graph._fullLayout.scene._scene;
        oldScene.glplot.gl.getExtension('WEBGL_lose_context').loseContext();
    }""")
    page.wait_for_function("""graph._fullLayout && graph._fullLayout.scene &&
        graph._fullLayout.scene._scene !== window.oldScene && !sceneContextLost()""")
    assert_download_has_points(page, page, tmp_path)


def test_download_survives_lost_hidden_export_context(page, tmp_path):
    page.goto(COMPONENT.as_uri())
    page.evaluate("""() => {
        window.contexts = new Set();
        const original = HTMLCanvasElement.prototype.getContext;
        HTMLCanvasElement.prototype.getContext = function(...args) {
            const context = original.apply(this, args);
            if (context && typeof context.getExtension === 'function') contexts.add(context);
            return context;
        };
    }""")
    render_plot(page)
    wait_for_scene(page)
    # Plotly retains this static renderer across downloads even after its
    # context is evicted by other channel plots.
    page.evaluate("Plotly.toImage(graph, {format: 'png'})")
    page.evaluate("""() => {
        const live = graph._fullLayout.scene._scene.glplot.gl;
        window.evictedContexts = [...contexts].filter(gl => gl !== live);
        if (!evictedContexts.length) throw new Error('Missing hidden export context');
        evictedContexts.forEach(gl => gl.getExtension('WEBGL_lose_context').loseContext());
    }""")
    page.wait_for_function("evictedContexts.every(gl => gl.isContextLost())")
    assert_download_has_points(page, page, tmp_path)


@pytest.mark.parametrize("trace_type", ["scatter", "scatter3d"])
def test_export_keeps_labels_colorbar_and_legend(page, tmp_path, trace_type):
    page.goto(COMPONENT.as_uri())
    render_plot(page)
    wait_for_scene(page)
    page.evaluate("""async traceType => {
        const figure = JSON.parse(JSON.stringify(latestArgs.figure));
        figure.data[0].type = traceType;
        figure.data[0].name = 'Measured channel';
        figure.data[0].showlegend = true;
        figure.data[0].marker = {size: 15, color: [1,2,3],
            colorscale: [[0,'red'],[1,'red']], showscale: true,
            colorbar: {title: 'Paired Q'}};
        figure.layout.title = 'Tensor export regression';
        figure.layout.annotations = [{text: 'Selected optimum', x: 0.1, y: 0.9,
            xref: 'paper', yref: 'paper', showarrow: false}];
        render({...latestArgs, figure, camera_enabled: traceType === 'scatter3d'});
        await renderPromise;
        const original = XMLSerializer.prototype.serializeToString;
        XMLSerializer.prototype.serializeToString = function(node) {
            const result = original.call(this, node);
            window.exportedSvg = result;
            return result;
        };
    }""", trace_type)
    assert_download_has_points(page, page, tmp_path)
    svg = page.evaluate("window.exportedSvg")
    for label in ("Measured channel", "Paired Q", "Tensor export regression", "Selected optimum"):
        assert label in svg


def test_empty_scene_is_rejected_instead_of_downloading_labels_only(page):
    page.goto(COMPONENT.as_uri())
    render_plot(page)
    wait_for_scene(page)
    downloads = []
    page.on("download", lambda item: downloads.append(item))
    page.evaluate("""() => {
        const scene = graph._fullLayout.scene._scene;
        const empty = document.createElement('canvas');
        empty.width = 500;
        empty.height = 400;
        scene.toImage = () => empty.toDataURL('image/png');
        const redraw = scene.glplot.redraw;
        scene.glplot.redraw = function() {
            redraw.call(this);
            const gl = this.gl;
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.clearColor(0, 0, 0, 0);
            gl.clear(gl.COLOR_BUFFER_BIT);
        };
    }""")
    page.locator('#download').click()
    page.wait_for_function("downloadButton.textContent === 'Download failed — retry'")
    assert 'empty 3D snapshot' in page.locator('#download').get_attribute('title')
    assert not downloads


def test_scene_is_composited_without_nested_svg_images(page, tmp_path):
    page.goto(COMPONENT.as_uri())
    render_plot(page)
    wait_for_scene(page)
    # Model a browser rasterizer omitting nested images inside an SVG.
    page.evaluate("""() => {
        const original = XMLSerializer.prototype.serializeToString;
        XMLSerializer.prototype.serializeToString = function(node) {
            const clone = node.cloneNode(true);
            clone.querySelectorAll('image').forEach(image => image.remove());
            return original.call(this, clone);
        };
    }""")
    assert_download_has_points(page, page, tmp_path)
