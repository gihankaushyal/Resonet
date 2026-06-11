"""Shared pytest fixtures for resonet tests."""
import os
import math
import tempfile
import numpy as np
import pytest

METADATA = {
    'detector_name': 'EIGER 4M',
    'distance_m': 0.300,
    'pixel_size_m': 7.5e-5,
    'photon_energy_eV': 8750.0,
    'wavelength_m': 1.417e-10,
}

GEOM_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "geoms", "Eigar.geom"
)

# A minimal synthetic 2-panel .geom for unit tests (no file I/O dependency)
MINIMAL_GEOM = """\
clen = 0.300
photon_energy = 8750
adu_per_eV = 0.001
res = 10000.0
data = /entry_1/data_1/data
dim0 = %
dim1 = ss
dim2 = fs

p0a0/fs = +1.0x +0.0y
p0a0/ss = +0.0x +1.0y
p0a0/corner_x = -100.0
p0a0/corner_y = -50.0
p0a0/min_fs = 0
p0a0/max_fs = 127
p0a0/min_ss = 0
p0a0/max_ss = 63

p0a1/fs = +1.0x +0.0y
p0a1/ss = +0.0x +1.0y
p0a1/corner_x = 10.0
p0a1/corner_y = -50.0
p0a1/min_fs = 128
p0a1/max_fs = 255
p0a1/min_ss = 0
p0a1/max_ss = 63
"""


@pytest.fixture
def minimal_geom_file(tmp_path):
    """Write MINIMAL_GEOM to a temp file and return the path."""
    p = tmp_path / "minimal.geom"
    p.write_text(MINIMAL_GEOM)
    return str(p)


@pytest.fixture
def two_panel_detector():
    """Return a minimal 2-panel dxtbx Detector for shift_* tests."""
    from dxtbx.model.detector import DetectorFactory
    pixel_size_mm = 1000.0 / 10000.0  # 0.1 mm
    panel_dicts = []
    for i, (cx, cy, min_fs, max_fs, min_ss, max_ss) in enumerate([
        (-10.0, -5.0, 0, 127, 0, 63),
        (1.0, -5.0, 128, 255, 0, 63),
    ]):
        panel_dicts.append({
            'name': f'p{i}',
            'type': '',
            'fast_axis': (1.0, 0.0, 0.0),
            'slow_axis': (0.0, 1.0, 0.0),
            'origin': (cx * pixel_size_mm, -cy * pixel_size_mm, -300.0),
            'pixel_size': (pixel_size_mm, pixel_size_mm),
            'image_size': (128, 64),
            'trusted_range': (0.0, 65536.0),
            'thickness': 0.0,
            'material': 'Si',
            'mu': 0.0,
            'gain': 1.0,
            'pedestal': 0.0,
            'identifier': '',
            'mask': [],
            'raw_image_offset': (0, 0),
            'px_mm_strategy': {'type': 'SimplePxMmStrategy'},
        })
    return DetectorFactory.from_dict({'panels': panel_dicts})
