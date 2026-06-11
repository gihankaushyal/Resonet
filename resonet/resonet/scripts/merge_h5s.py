import glob
import os
from argparse import ArgumentParser

from resonet.sims.merge_h5s import _merge_cxi, _merge_h5


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
