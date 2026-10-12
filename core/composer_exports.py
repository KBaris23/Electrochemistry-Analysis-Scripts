"""On-demand final figure exports, independent of preview/UI state."""
import copy
from io import BytesIO
import json
import threading
import zipfile

import matplotlib.pyplot as plt


RENDER_LOCK = threading.RLock()


class FigureExports:
    """Render once per immutable request; encode only requested formats.

    Callbacks run on Streamlit's download worker, so render must not read live
    widget state. Closed pyplot figures remain serializable by savefig.
    """

    def __init__(self, render, encode, metadata, dpi, stem):
        self.render = render
        self.encode = encode
        self.metadata = copy.deepcopy(metadata)
        self.dpi = dpi
        self.stem = stem
        self._figure = None
        self._files = {}

    def get(self, fmt):
        if fmt not in ('png', 'pdf', 'svg', 'zip'):
            raise ValueError(f'Unsupported export format: {fmt}')
        with RENDER_LOCK:
            if fmt in self._files:
                return self._files[fmt]
            if self._figure is None:
                figure = self.render()
                self.metadata['render_audit'] = {
                    'panels': list(getattr(figure, '_composer_panel_audit', ())),
                }
                self.metadata['export'] = {'dpi': self.dpi, 'transparent': True}
                self._figure = figure
                plt.close(figure)
                # Render inputs may be large; retain only the final figure.
                self.render = None
            encoded = json.dumps(self.metadata, ensure_ascii=False, default=str,
                                 separators=(',', ':')).encode('utf-8')
            if fmt == 'zip':
                output = BytesIO()
                with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
                    for extension in ('png', 'pdf', 'svg'):
                        archive.writestr(f'{self.stem}.{extension}', self.get(extension))
                    archive.writestr('figure-metadata.json', encoded)
                self._files[fmt] = output.getvalue()
            else:
                self._files[fmt] = self.encode(self._figure, fmt, self.dpi,
                                              metadata_json=encoded.decode('utf-8'))
            return self._files[fmt]
