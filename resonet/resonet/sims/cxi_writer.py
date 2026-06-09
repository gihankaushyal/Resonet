"""Write unassembled detector images to CXI HDF5 format."""
import numpy as np
import h5py


class CXIWriter:
    """Streams per-shot unassembled images to a CXI HDF5 file without buffering."""

    def __init__(self, filepath: str, frame_shape: tuple, metadata: dict):
        """
        Args:
            filepath:     output .cxi path
            frame_shape:  (n_ss, n_fs) unassembled image shape
            metadata:     dict with keys:
                          detector_name (str), distance_m (float),
                          pixel_size_m (float), photon_energy_eV (float),
                          wavelength_m (float)
        """
        self._frame_shape = frame_shape
        self._n_written = 0
        self._label_keys: set = set()

        self._file = h5py.File(filepath, 'w')
        meta = metadata

        det = self._file.require_group('entry_1/instrument_1/detector_1')
        det.create_dataset('description', data=np.bytes_(meta['detector_name']))
        det.create_dataset('distance', data=float(meta['distance_m']))
        det.create_dataset('x_pixel_size', data=float(meta['pixel_size_m']))
        det.create_dataset('y_pixel_size', data=float(meta['pixel_size_m']))

        src = self._file.require_group('entry_1/instrument_1/source_1')
        src.create_dataset('energy', data=float(meta['photon_energy_eV']))
        src.create_dataset('wavelength', data=float(meta['wavelength_m']))

        self._data_ds = self._file.create_dataset(
            'entry_1/data_1/data',
            shape=(0,) + frame_shape,
            maxshape=(None,) + frame_shape,
            dtype=np.uint16,
            compression='gzip',
            compression_opts=4,
            shuffle=True,
            chunks=(1,) + frame_shape,
        )

    def add_frame(self, image: np.ndarray, labels: dict = None):
        """Append one unassembled frame. image must have shape == frame_shape."""
        assert image.shape == self._frame_shape, (
            f"Expected shape {self._frame_shape}, got {image.shape}"
        )
        n = self._n_written
        self._data_ds.resize(n + 1, axis=0)
        self._data_ds[n] = image.astype(np.uint16)
        self._n_written += 1

        if labels:
            lbl_grp = self._file.require_group('entry_1/labels')
            for key, val in labels.items():
                if key not in self._label_keys:
                    lbl_grp.create_dataset(
                        key,
                        shape=(0,),
                        maxshape=(None,),
                        dtype=np.float32,
                    )
                    self._label_keys.add(key)
                ds = lbl_grp[key]
                ds.resize(n + 1, axis=0)
                ds[n] = float(val)

    def close(self):
        """Flush and close the HDF5 file."""
        if not self._file.id.valid:
            return
        self._file.close()
