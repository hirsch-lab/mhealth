#!/usr/bin/env python
import sys
import time
import argparse
import importlib
import pandas as pd
from pathlib import Path
from collections import defaultdict

ret = importlib.util.find_spec("mhealth")
if not ret:
    dir_path = Path(__file__).parent.resolve()
    src_path = (dir_path.parent / "src").resolve()
    sys.path.insert(0, str(src_path))

from mhealth.utils.context_info import dump_context
from mhealth.utils.commons import create_progress_bar
from mhealth.utils.file_helper import read_hdf, write_csv


def convert_single(path, out_dir, key_sep, forced):
    store = pd.HDFStore(path, mode="r")
    keys = store.keys()
    files_created = []
    try:
        for key in keys:
            key_out = key.replace("/", key_sep)
            df = store[key]
            path_ret = write_csv(df=df, path=out_dir/(path.stem+".csv"),
                                 key=key_out, key_sep="",
                                 exist_ok=forced,
                                 return_path=True)
            files_created.append(path_ret)
    finally:
        store.close()
    return files_created


def convert_multiple(in_dir, out_dir, key_sep, forced, glob):
    paths = list(sorted(in_dir.glob(glob)))
    if len(paths)==0:
        print("WARNING: No HDF stores found in input directory: '%s'" % in_dir)
        return {}
    files_created = {}
    prefix = "Converting {variables.file_name:<3}... "
    progress = create_progress_bar(label=None,
                                   size=len(paths),
                                   prefix=prefix,
                                   variables={"file_name": "N/A"})
    for i, path in enumerate(paths):
        progress.update(i, file_name=path.name)
        ret = convert_single(path=path, out_dir=out_dir,
                             key_sep=key_sep, forced=forced)
        files_created[path] = ret
    progress.finish()
    return files_created


def run(args):
    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)
    key_sep = args.key_sep
    forced = args.forced
    glob = args.glob

    dump_context(out_dir)
    start = time.time()
    if in_path.is_file():
        files = convert_single(path=in_path, out_dir=out_dir,
                               key_sep=key_sep, forced=forced)
        files = {in_path: files}
    else:
        files = convert_multiple(in_dir=in_path, out_dir=out_dir,
                                 key_sep=key_sep, forced=forced,
                                 glob=glob)
    stop = time.time()
    n_in = len(files)
    n_out = sum(len(v) for v in files.values())
    print("Converted %d HDF stores" % n_in)
    print("Created   %d CSV stores" % n_out)
    print("Duration: %.1fs" % (stop-start))


def parse_args():
    description = "Convert HDF stores to .csv files."
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(add_help=False,
                                     formatter_class=formatter,
                                     description=description)
    parser.add_argument("-h", "--help", action="help",
                        help="Show this help text")
    parser.add_argument("-i", "--in-path", default="./",
                        help="Input file or directory. Default: './'")
    parser.add_argument("-o", "--out-dir", default="./csv",
                        help="Output directory")
    parser.add_argument("-f", "--forced", action="store_true",
                        help="Force writing if folder is not empty.")
    parser.add_argument("-g", "--glob", type=str, default="*.h5",
                        help="Glob expression to select files.")
    parser.add_argument("--key-sep", type=str, default="-",
                        help="Key separator. Default: '-'")
    parser.set_defaults(func=run)
    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
