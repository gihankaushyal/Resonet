"""Smoke tests for AGIPD 3D frame assembly and noise application.

These tests exercise the exact code path in main.py using synthetic data
rather than full simtbx simulation (no GPU required).
"""
import os
import tempfile
import numpy as np
import pytest
import h5py


AGIPD_GEOM = os.path.join(
    os.path.dirname(__file__), "..", "sims", "geoms", "AGIPD.geom"
)


def _build_fake_flat_img(panel_map):
    """Build a synthetic flat pixel array with sequential values per panel."""
    total = sum(pm['n_fast'] * pm['n_slow'] for pm in panel_map)
    return np.arange(total, dtype=np.float32)


def test_group_by_module_agipd_geom():
    """_group_by_module returns 16 modules with 8 panels each for AGIPD.geom."""
    from resonet.sims.geom_parser import parse_geom
    from resonet.sims.main import _group_by_module
    _, panel_map, _ = parse_geom(AGIPD_GEOM)
    groups = _group_by_module(panel_map)
    assert groups is not None, "_group_by_module returned None for AGIPD.geom"
    assert len(groups) == 16, f"Expected 16 modules, got {len(groups)}"
    assert all(len(v) == 8 for v in groups.values()), "Each module must have 8 ASICs"


def test_3d_frame_shape_from_agipd_geom():
    """3D assembly of a synthetic flat array produces (16, 512, 128) frame."""
    from resonet.sims.geom_parser import parse_geom
    from resonet.sims.main import _group_by_module
    _, panel_map, _ = parse_geom(AGIPD_GEOM)
    groups = _group_by_module(panel_map)
    assert groups is not None

    n_modules = len(groups)
    ss_per_mod = max(pm['max_ss'] for pm in groups[0]) + 1
    fs_per_mod = max(pm['max_fs'] for pm in groups[0]) + 1
    frame_shape = (n_modules, ss_per_mod, fs_per_mod)

    assert frame_shape == (16, 512, 128), f"Expected (16,512,128), got {frame_shape}"


def test_3d_frame_no_overwrite():
    """Each ASIC writes to a unique (module, ss, fs) location — no panel overwrites another."""
    from resonet.sims.geom_parser import parse_geom
    from resonet.sims.main import _group_by_module
    _, panel_map, _ = parse_geom(AGIPD_GEOM)
    groups = _group_by_module(panel_map)

    frame_shape = (16, 512, 128)
    written = np.zeros(frame_shape, dtype=np.int32)

    _panel_pix_offset = {}
    _offset = 0
    for pm in panel_map:
        _panel_pix_offset[pm['name']] = _offset
        _offset += pm['n_fast'] * pm['n_slow']

    flat_img = _build_fake_flat_img(panel_map)

    for mod_idx, panels in sorted(groups.items()):
        for pm in panels:
            pix_off = _panel_pix_offset[pm['name']]
            n_px = pm['n_fast'] * pm['n_slow']
            panel_data = flat_img[pix_off:pix_off + n_px].reshape(pm['n_slow'], pm['n_fast'])
            written[
                mod_idx,
                pm['min_ss']:pm['max_ss'] + 1,
                pm['min_fs']:pm['max_fs'] + 1,
            ] += 1

    assert np.all(written == 1), (
        f"{np.sum(written != 1)} pixels were written != 1 times "
        f"(overwrite detected if > 1, missed if == 0)"
    )


def test_cxi_writer_accepts_3d_frame_shape():
    """CXIWriter with frame_shape=(16,512,128) creates (N,16,512,128) dataset."""
    from resonet.sims.cxi_writer import CXIWriter
    frame_shape = (16, 512, 128)
    meta = {
        'detector_name': 'AGIPD 1M',
        'distance_m': 0.15165,
        'pixel_size_m': 0.0002,
        'photon_energy_eV': 9385.0,
        'wavelength_m': 1239.84193e-9 / 9385.0,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test.cxi')
        writer = CXIWriter(path, frame_shape, meta)
        img = np.zeros(frame_shape, dtype=np.uint16)
        writer.add_frame(img, labels={'hit': 0.0})
        writer.add_frame(img, labels={'hit': 1.0})
        writer.close()

        with h5py.File(path, 'r') as f:
            data = f['entry_1/data_1/data']
            assert data.shape == (2, 16, 512, 128), (
                f"Expected (2,16,512,128), got {data.shape}"
            )
            assert data.dtype == np.uint16


def test_agipd_noise_on_3d_array():
    """apply_agipd_noise works on (16,512,128) input without shape change."""
    from resonet.sims.make_sims import apply_agipd_noise
    img = np.random.default_rng(0).random((16, 512, 128)).astype(np.float32) * 100
    out = apply_agipd_noise(img, rng=np.random.default_rng(1))
    assert out.shape == (16, 512, 128)
    assert out.dtype == np.float32
    assert np.all(out >= 0)
    assert np.max(out) > 0
