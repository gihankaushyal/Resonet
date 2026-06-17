import math
import os
import tempfile
import pytest

GEOM_PATH = os.path.join(
    os.path.dirname(__file__), "..", "sims", "geoms", "Eiger4m.geom"
)


# ---------------------------------------------------------------------------
# Tests on the real Eigar.geom file
# ---------------------------------------------------------------------------

def test_panel_count():
    from resonet.sims.geom_parser import parse_geom
    detector, panel_map, globals_ = parse_geom(GEOM_PATH)
    assert len(panel_map) == 64


def test_pixel_size():
    from resonet.sims.geom_parser import parse_geom
    detector, panel_map, globals_ = parse_geom(GEOM_PATH)
    panel = detector[0]
    px, py = panel.get_pixel_size()
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


def test_panel_map_index_matches_detector():
    """panel_map[i]['panel_idx'] == i and detector[i] is accessible."""
    from resonet.sims.geom_parser import parse_geom
    detector, panel_map, _ = parse_geom(GEOM_PATH)
    for i, pm in enumerate(panel_map):
        assert pm['panel_idx'] == i
        _ = detector[i]  # no IndexError


def test_fast_slow_axes_unit_vectors():
    """Fast and slow axes are unit vectors (norm ≈ 1) for all panels."""
    from resonet.sims.geom_parser import parse_geom
    detector, _, _ = parse_geom(GEOM_PATH)
    for panel in detector:
        fast = panel.get_fast_axis()
        slow = panel.get_slow_axis()
        assert abs(math.sqrt(sum(x**2 for x in fast)) - 1.0) < 1e-4
        assert abs(math.sqrt(sum(x**2 for x in slow)) - 1.0) < 1e-4


def test_image_size_matches_panel_map():
    """detector[i].get_image_size() == (n_fast, n_slow) from panel_map[i]."""
    from resonet.sims.geom_parser import parse_geom
    detector, panel_map, _ = parse_geom(GEOM_PATH)
    for i, pm in enumerate(panel_map):
        img_sz = detector[i].get_image_size()  # (fast, slow)
        assert img_sz[0] == pm['n_fast'], f"Panel {i}: fast {img_sz[0]} != {pm['n_fast']}"
        assert img_sz[1] == pm['n_slow'], f"Panel {i}: slow {img_sz[1]} != {pm['n_slow']}"


# ---------------------------------------------------------------------------
# Tests on _parse_axis (unit tests for the helper)
# ---------------------------------------------------------------------------

def test_parse_axis_standard():
    from resonet.sims.geom_parser import _parse_axis
    x, y, z = _parse_axis('-0.999991x -0.004221y')
    assert abs(x - (-0.999991)) < 1e-6
    assert abs(y - (-0.004221)) < 1e-6
    assert z == 0.0


def test_parse_axis_positive():
    from resonet.sims.geom_parser import _parse_axis
    x, y, z = _parse_axis('+0.5x +0.866y')
    assert abs(x - 0.5) < 1e-6
    assert abs(y - 0.866) < 1e-6


def test_parse_axis_missing_component():
    """Axis with only x component → y and z default to 0."""
    from resonet.sims.geom_parser import _parse_axis
    x, y, z = _parse_axis('1.0x')
    assert x == pytest.approx(1.0)
    assert y == 0.0
    assert z == 0.0


def test_parse_axis_scientific_notation():
    """Scientific notation in axis values is parsed correctly."""
    from resonet.sims.geom_parser import _parse_axis
    x, y, z = _parse_axis('1.5e-3x -2.0e2y')
    assert abs(x - 1.5e-3) < 1e-9
    assert abs(y - (-200.0)) < 1e-6


def test_parse_axis_all_three_components():
    """All three xyz components parsed."""
    from resonet.sims.geom_parser import _parse_axis
    x, y, z = _parse_axis('0.1x 0.2y 0.3z')
    assert abs(x - 0.1) < 1e-6
    assert abs(y - 0.2) < 1e-6
    assert abs(z - 0.3) < 1e-6


# ---------------------------------------------------------------------------
# Tests on _panel_sort_key
# ---------------------------------------------------------------------------

def test_panel_sort_numeric_order():
    """p9a3 sorts before p10a0 (numeric, not lexicographic)."""
    from resonet.sims.geom_parser import _panel_sort_key
    panels = [{'name': 'p10a0'}, {'name': 'p1a0'}, {'name': 'p9a3'}]
    sorted_names = [p['name'] for p in sorted(panels, key=_panel_sort_key)]
    assert sorted_names == ['p1a0', 'p9a3', 'p10a0']


# ---------------------------------------------------------------------------
# Tests on parse_geom error handling (using synthetic .geom strings)
# ---------------------------------------------------------------------------

def _write_geom(tmp_path, content):
    p = tmp_path / 'test.geom'
    p.write_text(content)
    return str(p)


VALID_TWO_PANEL = """\
clen = 0.300
photon_energy = 8750
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


def test_valid_two_panel_geom(tmp_path):
    from resonet.sims.geom_parser import parse_geom
    path = _write_geom(tmp_path, VALID_TWO_PANEL)
    detector, panel_map, globals_ = parse_geom(path)
    assert len(panel_map) == 2
    assert globals_['clen'] == pytest.approx(0.300)


def test_missing_clen_raises(tmp_path):
    from resonet.sims.geom_parser import parse_geom
    content = VALID_TWO_PANEL.replace('clen = 0.300\n', '')
    path = _write_geom(tmp_path, content)
    with pytest.raises(ValueError, match='clen'):
        parse_geom(path)


def test_missing_res_raises(tmp_path):
    from resonet.sims.geom_parser import parse_geom
    content = VALID_TWO_PANEL.replace('res = 10000.0\n', '')
    path = _write_geom(tmp_path, content)
    with pytest.raises(ValueError, match='res'):
        parse_geom(path)


def test_dynamic_clen_reference_raises(tmp_path):
    """Dynamic LCLS-style clen reference is skipped → missing clen → ValueError."""
    from resonet.sims.geom_parser import parse_geom
    content = VALID_TWO_PANEL.replace('clen = 0.300', 'clen = /LCLS/detector/distance')
    path = _write_geom(tmp_path, content)
    with pytest.raises(ValueError, match='clen'):
        parse_geom(path)


def test_nonexistent_file_raises():
    from resonet.sims.geom_parser import parse_geom
    with pytest.raises(FileNotFoundError):
        parse_geom('/nonexistent/path/to/file.geom')


def test_no_valid_panels_raises(tmp_path):
    """geom with panels missing required fields raises ValueError with panel details."""
    from resonet.sims.geom_parser import parse_geom
    content = """\
clen = 0.300
res = 10000.0
photon_energy = 8750
p0a0/corner_x = 0.0
p0a0/corner_y = 0.0
"""
    path = _write_geom(tmp_path, content)
    with pytest.raises(ValueError, match='[Nn]o valid panel'):
        parse_geom(path)


def test_corner_y_negated_in_origin(tmp_path):
    """corner_y=20 → origin_mm[1] == -20 * pixel_size_mm (CrystFEL +y up, dxtbx +y down)."""
    from resonet.sims.geom_parser import parse_geom
    content = """\
clen = 0.300
res = 10000.0
photon_energy = 8750
p0a0/fs = +1.0x +0.0y
p0a0/ss = +0.0x +1.0y
p0a0/corner_x = 0.0
p0a0/corner_y = 20.0
p0a0/min_fs = 0
p0a0/max_fs = 63
p0a0/min_ss = 0
p0a0/max_ss = 63
"""
    path = _write_geom(tmp_path, content)
    detector, panel_map, _ = parse_geom(path)
    origin = detector[0].get_origin()
    pixel_size_mm = 1000.0 / 10000.0
    expected_y = -20.0 * pixel_size_mm
    assert abs(origin[1] - expected_y) < 1e-6, (
        f"origin_mm[1]={origin[1]:.6f} expected {expected_y:.6f} "
        "(corner_y must be negated for dxtbx convention)"
    )


def test_normalize_zero_vector_raises():
    """_normalize raises ValueError for a zero-length input vector."""
    from resonet.sims.geom_parser import _normalize
    with pytest.raises(ValueError, match="[Zz]ero"):
        _normalize((0.0, 0.0, 0.0))


def test_orthogonalize_parallel_axes_raises():
    """_orthogonalize raises ValueError when fast and slow are parallel."""
    from resonet.sims.geom_parser import _orthogonalize
    with pytest.raises(ValueError, match="parallel"):
        _orthogonalize((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))


def test_parse_axis_no_xyz_token_raises():
    """_parse_axis raises ValueError when the string has no x/y/z token."""
    from resonet.sims.geom_parser import _parse_axis
    with pytest.raises(ValueError):
        _parse_axis("0.5")


def test_comments_ignored(tmp_path):
    """Lines starting with ; are treated as comments and ignored."""
    from resonet.sims.geom_parser import parse_geom
    content = """\
; This is a comment
clen = 0.300
; Another comment
res = 10000.0
photon_energy = 8750
; bad_panel/corner_x = 999.0   <-- this should be ignored
p0a0/fs = +1.0x +0.0y
p0a0/ss = +0.0x +1.0y
p0a0/corner_x = 5.0
p0a0/corner_y = 5.0
p0a0/min_fs = 0
p0a0/max_fs = 63
p0a0/min_ss = 0
p0a0/max_ss = 63
"""
    path = _write_geom(tmp_path, content)
    _, panel_map, _ = parse_geom(path)
    assert len(panel_map) == 1
