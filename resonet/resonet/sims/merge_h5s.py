"""Merge per-rank CXI or HDF5 simulation output files via HDF5 virtual datasets."""
import h5py
import numpy as np


def _merge_cxi(fnames, outname, prefix):
    if not fnames:
        print("No CXI files found to merge.")
        return

    with h5py.File(fnames[0], 'r') as dummie:
        frame_shape = dummie['/entry_1/data_1/data'].shape[1:]
        data_dtype = dummie['/entry_1/data_1/data'].dtype
        label_keys = []
        if 'entry_1/labels' in dummie:
            label_keys = list(dummie['entry_1/labels'].keys())
        has_instrument = 'entry_1/instrument_1' in dummie

    imgs_per_fname = []
    for f in fnames:
        with h5py.File(f, 'r') as h:
            imgs_per_fname.append(h['/entry_1/data_1/data'].shape[0])
    total_imgs = sum(imgs_per_fname)

    data_layout = h5py.VirtualLayout(
        shape=(total_imgs,) + frame_shape, dtype=data_dtype
    )
    label_layouts = {
        key: h5py.VirtualLayout(shape=(total_imgs,), dtype=np.float32)
        for key in label_keys
    }

    start = 0
    for i_f, f in enumerate(fnames):
        print(f"virtualizing file {i_f+1} / {len(fnames)}")
        nimg = imgs_per_fname[i_f]
        vsource = h5py.VirtualSource(f, '/entry_1/data_1/data',
                                      shape=(nimg,) + frame_shape)
        data_layout[start:start + nimg] = vsource
        for key in label_keys:
            vs = h5py.VirtualSource(f, f'entry_1/labels/{key}',
                                     shape=(nimg,))
            label_layouts[key][start:start + nimg] = vs
        start += nimg

    print(f"Saving to {outname}, total shots={total_imgs}")
    with h5py.File(outname, 'w') as H:
        if has_instrument:
            with h5py.File(fnames[0], 'r') as src:
                src.copy('entry_1/instrument_1',
                         H.require_group('entry_1'),
                         name='instrument_1')
        H.create_virtual_dataset('/entry_1/data_1/data', data_layout)
        for key, layout in label_layouts.items():
            H.create_virtual_dataset(f'entry_1/labels/{key}', layout)
    print("Done!")


def _merge_h5(fnames, outname, prefix, more_keys):
    with h5py.File(fnames[0], "r") as dummie_h:
        shapes = {}
        for key in ["images_mean", "images", "labels", "full_maximg", "geom"] + more_keys:
            try:
                shapes[key] = dummie_h[key].shape[1:]
            except KeyError:
                pass

        imgs_per_fname = []
        for f in fnames:
            with h5py.File(f, 'r') as fh:
                imgs_per_fname.append(fh['labels'].shape[0])
        total_imgs = sum(imgs_per_fname)

        Layouts = {}
        for key, shape in shapes.items():
            Layouts[key] = h5py.VirtualLayout(shape=(total_imgs,) + shape, dtype=dummie_h[key].dtype)

        start = 0
        for i_f, f in enumerate(fnames):
            print("virtualizing file %d / %d" % (i_f+1, len(fnames)))
            nimg = imgs_per_fname[i_f]
            for key in Layouts:
                vsource = h5py.VirtualSource(f, key, shape=(nimg,) + shapes[key])
                Layouts[key][start:start+nimg] = vsource

            start += nimg

        print("Saving it all to %s!" % outname)
        print("Total number of shots=%d" % total_imgs)
        with h5py.File(outname, "w") as H:
            for key in Layouts:
                vd = H.create_virtual_dataset(key, Layouts[key])
                for attr in ["names", "pdbmap"]:
                    if attr in dummie_h[key].attrs:
                        vd.attrs[attr] = dummie_h[key].attrs[attr]

    print("Done!")
