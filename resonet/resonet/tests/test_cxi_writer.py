import os
import tempfile
import numpy as np
import h5py
import pytest


METADATA = {
    'detector_name': 'EIGER 4M',
    'distance_m': 0.300,
    'pixel_size_m': 7.5e-5,
    'photon_energy_eV': 8750.0,
    'wavelength_m': 1.417e-10,
}


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
        with pytest.raises(AssertionError):
            w.add_frame(np.zeros((512, 512), dtype=np.uint16))
