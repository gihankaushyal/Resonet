import math
import os
import pytest

GEOM_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "geoms", "Eigar.geom"
)


def test_panel_count():
    from resonet.sims.geom_parser import parse_geom
    detector, panel_map, globals_ = parse_geom(GEOM_PATH)
    assert len(panel_map) == 64


def test_pixel_size():
    from resonet.sims.geom_parser import parse_geom
    detector, panel_map, globals_ = parse_geom(GEOM_PATH)
    panel = detector[0]
    px, py = panel.get_pixel_size()
    # Eigar.geom has res = 10000.075 px/m -> pixel_size = 1000/10000.075 ≈ 0.1 mm (100 µm)
    expected_px = 1000.0 / 10000.075
    assert abs(px - expected_px) < 1e-6
    assert abs(py - expected_px) < 1e-6


def test_panel_map_fields():
    from resonet.sims.geom_parser import parse_geom
    _, panel_map, _ = parse_geom(GEOM_PATH)
    required = {'name', 'min_fs', 'max_fs', 'min_ss', 'max_ss',
                'panel_idx', 'n_fast', 'n_slow'}
    for pm in panel_map:
        assert required.issubset(pm.keys())


def test_unassembled_shape():
    from resonet.sims.geom_parser import parse_geom
    _, panel_map, _ = parse_geom(GEOM_PATH)
    n_ss = max(pm['max_ss'] for pm in panel_map) + 1
    n_fs = max(pm['max_fs'] for pm in panel_map) + 1
    assert n_ss == 5632
    assert n_fs == 384


def test_panel_orthogonality():
    from resonet.sims.geom_parser import parse_geom
    detector, panel_map, _ = parse_geom(GEOM_PATH)
    for i, pm in enumerate(panel_map):
        panel = detector[i]
        fast = panel.get_fast_axis()
        slow = panel.get_slow_axis()
        dot = sum(f * s for f, s in zip(fast, slow))
        assert abs(dot) < 0.01, f"Panel {pm['name']} not orthogonal: dot={dot:.4f}"


def test_global_params():
    from resonet.sims.geom_parser import parse_geom
    _, _, globals_ = parse_geom(GEOM_PATH)
    assert 'clen' in globals_
    assert 'res' in globals_
    assert 'photon_energy' in globals_
    assert abs(globals_['clen'] - 0.300) < 1e-6
    assert abs(globals_['res'] - 10000.075) < 0.01
