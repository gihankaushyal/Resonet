import os
import math
import tempfile
import numpy as np
import h5py
import pytest
from unittest.mock import patch


METADATA = {
    'detector_name': 'EIGER 4M',
    'distance_m': 0.300,
    'pixel_size_m': 7.5e-5,
    'photon_energy_eV': 8750.0,
    'wavelength_m': 1.417e-10,
}

FRAME_SHAPE = (64, 32)  # small shape speeds up all tests


def make_writer(path, shape=FRAME_SHAPE, meta=None):
    from resonet.sims.cxi_writer import CXIWriter
    return CXIWriter(path, shape, meta or METADATA)


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

def test_cxi_data_shape():
    from resonet.sims.cxi_writer import CXIWriter
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'test.cxi')
        w = CXIWriter(path, (5632, 384), METADATA)
        for _ in range(3):
            w.add_frame(np.zeros((5632, 384), dtype=np.uint16))
        w.close()
        with h5py.File(path, 'r') as f:
            assert f['/entry_1/data_1/data'].shape == (3, 5632, 384)
            assert f['/entry_1/data_1/data'].dtype == np.uint16


def test_cxi_metadata():
    from resonet.sims.cxi_writer import CXIWriter
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'test.cxi')
        w = CXIWriter(path, (5632, 384), METADATA)
        w.add_frame(np.zeros((5632, 384), dtype=np.uint16))
        w.close()
        with h5py.File(path, 'r') as f:
            det = f['/entry_1/instrument_1/detector_1']
            assert det['description'][()].decode() == 'EIGER 4M'
            assert abs(float(det['distance'][()]) - 0.300) < 1e-6
            assert abs(float(det['x_pixel_size'][()]) - 7.5e-5) < 1e-9
            src = f['/entry_1/instrument_1/source_1']
            assert abs(float(src['energy'][()]) - 8750.0) < 0.01
            assert abs(float(src['wavelength'][()]) - 1.417e-10) < 1e-14


def test_cxi_labels():
    from resonet.sims.cxi_writer import CXIWriter
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'test.cxi')
        w = CXIWriter(path, (5632, 384), METADATA)
        w.add_frame(np.zeros((5632, 384), dtype=np.uint16),
                    labels={'hit': 1.0, 'resolution': 2.5})
        w.add_frame(np.zeros((5632, 384), dtype=np.uint16),
                    labels={'hit': 0.0, 'resolution': 0.0})
        w.close()
        with h5py.File(path, 'r') as f:
            assert f['/entry_1/labels/hit'].shape == (2,)
            assert f['/entry_1/labels/resolution'].shape == (2,)
            assert float(f['/entry_1/labels/hit'][0]) == 1.0


def test_cxi_compression():
    from resonet.sims.cxi_writer import CXIWriter
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'test.cxi')
        w = CXIWriter(path, (5632, 384), METADATA)
        w.add_frame(np.zeros((5632, 384), dtype=np.uint16))
        w.close()
        with h5py.File(path, 'r') as f:
            ds = f['/entry_1/data_1/data']
            assert ds.compression == 'gzip'


def test_add_frame_wrong_shape_raises():
    from resonet.sims.cxi_writer import CXIWriter
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'test.cxi')
        w = CXIWriter(path, (5632, 384), METADATA)
        with pytest.raises(ValueError):
            w.add_frame(np.zeros((512, 512), dtype=np.uint16))


# ---------------------------------------------------------------------------
# New: metadata validation
# ---------------------------------------------------------------------------

def test_metadata_missing_key_raises():
    from resonet.sims.cxi_writer import CXIWriter
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'test.cxi')
        bad = {k: v for k, v in METADATA.items() if k != 'wavelength_m'}
        with pytest.raises(ValueError, match='wavelength_m'):
            CXIWriter(path, FRAME_SHAPE, bad)


def test_metadata_all_keys_missing_raises():
    from resonet.sims.cxi_writer import CXIWriter
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'test.cxi')
        with pytest.raises(ValueError):
            CXIWriter(path, FRAME_SHAPE, {})


# ---------------------------------------------------------------------------
# New: close() idempotency and add_frame after close
# ---------------------------------------------------------------------------

def test_close_idempotent(tmp_path):
    from resonet.sims.cxi_writer import CXIWriter
    path = str(tmp_path / 'test.cxi')
    w = make_writer(path)
    w.add_frame(np.zeros(FRAME_SHAPE, dtype=np.uint16))
    w.close()
    w.close()  # must not raise


def test_add_frame_after_close_raises(tmp_path):
    from resonet.sims.cxi_writer import CXIWriter
    path = str(tmp_path / 'test.cxi')
    w = make_writer(path)
    w.close()
    with pytest.raises(Exception):
        w.add_frame(np.zeros(FRAME_SHAPE, dtype=np.uint16))


# ---------------------------------------------------------------------------
# New: label edge cases
# ---------------------------------------------------------------------------

def test_label_new_key_appears_after_first_frame(tmp_path):
    """A label key introduced on frame 1 creates a dataset with len==2; frame 0 is zero."""
    from resonet.sims.cxi_writer import CXIWriter
    path = str(tmp_path / 'test.cxi')
    w = make_writer(path)
    w.add_frame(np.zeros(FRAME_SHAPE, dtype=np.uint16), labels={'hit': 1.0})
    w.add_frame(np.zeros(FRAME_SHAPE, dtype=np.uint16), labels={'hit': 0.0, 'new_key': 7.0})
    w.close()
    with h5py.File(path, 'r') as f:
        assert f['/entry_1/labels/hit'].shape == (2,)
        assert f['/entry_1/labels/new_key'].shape == (2,)
        assert float(f['/entry_1/labels/new_key'][1]) == pytest.approx(7.0)


def test_label_nan_value_preserved(tmp_path):
    """NaN float label is stored as NaN in float32."""
    from resonet.sims.cxi_writer import CXIWriter
    path = str(tmp_path / 'test.cxi')
    w = make_writer(path)
    w.add_frame(np.zeros(FRAME_SHAPE, dtype=np.uint16), labels={'val': float('nan')})
    w.close()
    with h5py.File(path, 'r') as f:
        assert math.isnan(float(f['/entry_1/labels/val'][0]))


def test_no_labels_no_labels_group(tmp_path):
    """Writing frames with no labels doesn't create /entry_1/labels group."""
    from resonet.sims.cxi_writer import CXIWriter
    path = str(tmp_path / 'test.cxi')
    w = make_writer(path)
    w.add_frame(np.zeros(FRAME_SHAPE, dtype=np.uint16))
    w.close()
    with h5py.File(path, 'r') as f:
        assert 'labels' not in f.get('entry_1', {})


# ---------------------------------------------------------------------------
# New: HDF5 structure
# ---------------------------------------------------------------------------

def test_hdf5_group_structure(tmp_path):
    """Verify required CXI group paths exist after writing."""
    from resonet.sims.cxi_writer import CXIWriter
    path = str(tmp_path / 'test.cxi')
    w = make_writer(path)
    w.add_frame(np.zeros(FRAME_SHAPE, dtype=np.uint16))
    w.close()
    with h5py.File(path, 'r') as f:
        assert '/entry_1/data_1/data' in f
        assert '/entry_1/instrument_1/detector_1/description' in f
        assert '/entry_1/instrument_1/detector_1/distance' in f
        assert '/entry_1/instrument_1/source_1/energy' in f
        assert '/entry_1/instrument_1/source_1/wavelength' in f


def test_data_chunking(tmp_path):
    """Data dataset is chunked as (1, *frame_shape)."""
    from resonet.sims.cxi_writer import CXIWriter
    path = str(tmp_path / 'test.cxi')
    w = make_writer(path)
    w.add_frame(np.zeros(FRAME_SHAPE, dtype=np.uint16))
    w.close()
    with h5py.File(path, 'r') as f:
        chunks = f['/entry_1/data_1/data'].chunks
        assert chunks == (1,) + FRAME_SHAPE


# ---------------------------------------------------------------------------
# New: init try/finally on create_dataset failure
# ---------------------------------------------------------------------------

def test_init_closes_file_on_create_dataset_error(tmp_path):
    """If __init__ create_dataset raises, the HDF5 file handle is closed."""
    from resonet.sims.cxi_writer import CXIWriter
    path = str(tmp_path / 'test.cxi')
    with patch('h5py.File.create_dataset', side_effect=RuntimeError("injected")):
        with pytest.raises(RuntimeError, match='injected'):
            CXIWriter(path, FRAME_SHAPE, METADATA)
    # File was opened then closed — re-opening should not raise
    with h5py.File(path, 'r') as _:
        pass


# ---------------------------------------------------------------------------
# New: float input converted to uint16
# ---------------------------------------------------------------------------

def test_float32_input_converted_to_uint16(tmp_path):
    """add_frame converts float32 input to uint16 correctly."""
    from resonet.sims.cxi_writer import CXIWriter
    path = str(tmp_path / 'test.cxi')
    img = np.full(FRAME_SHAPE, 1234.9, dtype=np.float32)
    w = make_writer(path)
    w.add_frame(img)
    w.close()
    with h5py.File(path, 'r') as f:
        stored = f['/entry_1/data_1/data'][0]
        assert stored.dtype == np.uint16
        assert int(stored[0, 0]) == 1234  # truncation to uint16
