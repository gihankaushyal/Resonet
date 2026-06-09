import h5py
import numpy as np
import glob
import os
from argparse import ArgumentParser


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
    dummie_h = h5py.File(fnames[0], "r")

    shapes = {}
    for key in ["images_mean", "images", "labels", "full_maximg", "geom"] + more_keys:
        try:
            shapes[key] = dummie_h[key].shape[1:]
        except KeyError:
            pass

    imgs_per_fname = [h5py.File(f, 'r')['labels'].shape[0] for f in fnames]
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


def main():
    parser = ArgumentParser()
    parser.add_argument("dirnames", nargs="+", type=str, help="output folders from runme.py or runme_joblib.py")
    parser.add_argument("outname", type=str, help="name of the  master file")
    parser.add_argument("--moreKeys", nargs="+", type=str, default=[], help="names of additional datasets to virtualize. These should be present in all rank* files!")
    parser.add_argument("--prefix", type=str, default="compressed",
            help="merge h5 files that start with this (default: compressed)")
    parser.add_argument("--cxi", action="store_true",
            help="merge .cxi files instead of .h5 files")
    args = parser.parse_args()

    """
    Use this method to merge the rank*.h5 files that are output by
    runme_cpu.py (when using MPI mode , each rank writes a file)
    Or merge .cxi files with --cxi flag.
    """

    ext = 'cxi' if args.cxi else 'h5'
    fnames = []
    for dirname in args.dirnames:
        fnames += glob.glob(os.path.join(dirname, "%s*.%s" % (args.prefix, ext)))
    fnames = [os.path.abspath(f) for f in sorted(fnames)]

    print("Combining %d files" % len(fnames))

    if args.cxi:
        _merge_cxi(fnames, args.outname, args.prefix)
    else:
        _merge_h5(fnames, args.outname, args.prefix, args.moreKeys)


if __name__=="__main__":
    main()
