"""Write unassembled detector images to CXI HDF5 format."""
import numpy as np
import h5py


class CXIWriter:
    """Accumulates per-shot unassembled images and writes to a CXI HDF5 file."""

    def __init__(self, filepath: str, frame_shape: tuple, metadata: dict):
        self._filepath = filepath
        self._frame_shape = frame_shape
        self._metadata = metadata
        self._frames: list = []
        self._labels: list = []

    def add_frame(self, image: np.ndarray, labels: dict = None):
        """Append one unassembled frame. image must have shape == frame_shape."""
        assert image.shape == self._frame_shape, (
            f"Expected shape {self._frame_shape}, got {image.shape}"
        )
        self._frames.append(image.astype(np.uint16))
        self._labels.append(labels or {})

    def close(self):
        """Write all accumulated frames to disk. No-op if no frames added."""
        if not self._frames:
            return
        data = np.stack(self._frames, axis=0)
        meta = self._metadata

        with h5py.File(self._filepath, 'w') as f:
            det = f.require_group('entry_1/instrument_1/detector_1')
            det.create_dataset('description',
                               data=np.bytes_(meta['detector_name']))
            det.create_dataset('distance', data=float(meta['distance_m']))
            det.create_dataset('x_pixel_size',
                               data=float(meta['pixel_size_m']))
            det.create_dataset('y_pixel_size',
                               data=float(meta['pixel_size_m']))

            src = f.require_group('entry_1/instrument_1/source_1')
            src.create_dataset('energy',
                               data=float(meta['photon_energy_eV']))
            src.create_dataset('wavelength',
                               data=float(meta['wavelength_m']))

            f.create_dataset(
                'entry_1/data_1/data',
                data=data,
                compression='gzip',
                compression_opts=4,
                shuffle=True,
            )

            if any(self._labels):
                all_keys = set()
                for d in self._labels:
                    all_keys.update(d.keys())
                lbl = f.require_group('entry_1/labels')
                for key in sorted(all_keys):
                    vals = [d.get(key, float('nan')) for d in self._labels]
                    lbl.create_dataset(key, data=np.array(vals,
                                                           dtype=np.float32))
