"""Tests for _merge_cxi in merge_h5s.py."""
import os
import tempfile
import numpy as np
import h5py
import pytest


METADATA = {
    'detector_name': 'TEST',
    'distance_m': 0.1,
    'pixel_size_m': 1e-4,
    'photon_energy_eV': 9000.0,
    'wavelength_m': 1.38e-10,
}
SHAPE = (8, 16)  # small shape for fast tests


def _make_cxi(path, n_frames, labels=True):
    """Helper: write n_frames to a CXI file."""
    from resonet.sims.cxi_writer import CXIWriter
    w = CXIWriter(path, SHAPE, METADATA)
    for i in range(n_frames):
        lbl = {'hit': float(i % 2), 'distance': 0.1 + i * 0.01} if labels else None
        w.add_frame(np.full(SHAPE, i, dtype=np.uint16), labels=lbl)
    w.close()


def test_merge_total_frame_count(tmp_path):
    """Merging 2 files × 3 frames each → merged dataset has 6 frames."""
    from resonet.sims.merge_h5s import _merge_cxi
    f0 = str(tmp_path / 'rank0.cxi')
    f1 = str(tmp_path / 'rank1.cxi')
    _make_cxi(f0, 3)
    _make_cxi(f1, 3)
    out = str(tmp_path / 'merged.cxi')
    _merge_cxi([f0, f1], out, '')
    with h5py.File(out, 'r') as f:
        assert f['/entry_1/data_1/data'].shape == (6,) + SHAPE


def test_merge_empty_list_no_output(tmp_path, capsys):
    """Empty file list prints a message and does NOT create the output file."""
    from resonet.sims.merge_h5s import _merge_cxi
    out = str(tmp_path / 'merged.cxi')
    _merge_cxi([], out, '')
    assert not os.path.exists(out)
    captured = capsys.readouterr()
    assert 'No CXI files found' in captured.out


def test_merge_preserves_metadata(tmp_path):
    """instrument_1 metadata from first file is present in merged output."""
    from resonet.sims.merge_h5s import _merge_cxi
    f0 = str(tmp_path / 'rank0.cxi')
    f1 = str(tmp_path / 'rank1.cxi')
    _make_cxi(f0, 2)
    _make_cxi(f1, 2)
    out = str(tmp_path / 'merged.cxi')
    _merge_cxi([f0, f1], out, '')
    with h5py.File(out, 'r') as f:
        assert '/entry_1/instrument_1/detector_1/description' in f
        assert f['/entry_1/instrument_1/source_1/wavelength'][()] == pytest.approx(1.38e-10)


def test_merge_label_keys_present(tmp_path):
    """All label keys from source files appear in merged output."""
    from resonet.sims.merge_h5s import _merge_cxi
    f0 = str(tmp_path / 'rank0.cxi')
    f1 = str(tmp_path / 'rank1.cxi')
    _make_cxi(f0, 2)
    _make_cxi(f1, 2)
    out = str(tmp_path / 'merged.cxi')
    _merge_cxi([f0, f1], out, '')
    with h5py.File(out, 'r') as f:
        assert '/entry_1/labels/hit' in f
        assert '/entry_1/labels/distance' in f
        assert f['/entry_1/labels/hit'].shape == (4,)


def test_merge_virtual_slicing_correct(tmp_path):
    """Virtual dataset correctly maps file0 → frames [0:3], file1 → frames [3:6]."""
    from resonet.sims.merge_h5s import _merge_cxi
    f0 = str(tmp_path / 'rank0.cxi')
    f1 = str(tmp_path / 'rank1.cxi')
    # Each frame is filled with its index for easy identification
    _make_cxi(f0, 3, labels=False)  # frames: 0,1,2
    _make_cxi(f1, 3, labels=False)  # frames: 0,1,2 (same values, different file)
    out = str(tmp_path / 'merged.cxi')
    _merge_cxi([f0, f1], out, '')
    with h5py.File(out, 'r') as f:
        data = f['/entry_1/data_1/data']
        # Frame 0 (from f0) should be all zeros
        assert int(data[0, 0, 0]) == 0
        # Frame 2 (from f0) should be all 2s
        assert int(data[2, 0, 0]) == 2
        # Frame 3 (from f1, first frame) should be all 0s
        assert int(data[3, 0, 0]) == 0


def test_merge_single_file(tmp_path):
    """Merging a single file produces output equal to that file."""
    from resonet.sims.merge_h5s import _merge_cxi
    f0 = str(tmp_path / 'rank0.cxi')
    _make_cxi(f0, 4)
    out = str(tmp_path / 'merged.cxi')
    _merge_cxi([f0], out, '')
    with h5py.File(out, 'r') as f:
        assert f['/entry_1/data_1/data'].shape == (4,) + SHAPE
        assert '/entry_1/labels/hit' in f


def test_merge_no_labels_file(tmp_path):
    """Merging files without labels produces output without labels group."""
    from resonet.sims.merge_h5s import _merge_cxi
    f0 = str(tmp_path / 'rank0.cxi')
    f1 = str(tmp_path / 'rank1.cxi')
    _make_cxi(f0, 2, labels=False)
    _make_cxi(f1, 2, labels=False)
    out = str(tmp_path / 'merged.cxi')
    _merge_cxi([f0, f1], out, '')
    with h5py.File(out, 'r') as f:
        assert f['/entry_1/data_1/data'].shape == (4,) + SHAPE
        assert 'labels' not in f.get('entry_1', {})
