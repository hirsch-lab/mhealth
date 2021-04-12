"""
This script was used to process the manual measurements of De Morton exercises.

Input:  Folder with .xlsx files with the following pattern: <pid>-<day>.xlsx
        The files contain time measurements for the De Morton exercises.
Output: A single table per patient with proper format
        The files are written to:   - out_dir/exercises/ as .csv
                                    - out_dir/store/     as .h5
"""

import re
import itertools
import pandas as pd
from pathlib import Path
from collections import defaultdict

import context
from mhealth.utils.commons import create_progress_bar, print_title
from mhealth.patient.imove_label_loader import load_labels


def write_csv(df, out_path, **kwargs):
    out_dir = out_path.parent
    if not out_dir.is_dir():
        out_dir.mkdir(parents=True, exist_ok=True)
    sep = kwargs.pop("sep", ",")
    with_index = kwargs.pop("index", False)
    df.to_csv(out_path, sep=sep, index=with_index, **kwargs)


def write_hdf(df, out_path, key=None, **kwargs):
    """
    Convention: out_path = "path/to/file.h5/sub/path"
                is equivalent to
                out_path = "path/to/file.h5"
                key = "sub/path" if key is None else key

    Detail: note the file-size overhead!
    https://stackoverflow.com/questions/21635224
    Solution: Use ptrepack, a command line utility (part of PyTables/tables):
            ptrepack --chunkshape=auto --complevel=9 --complib=blosc \
                     infile.h5 outfile.repack.h5
    """
    out_path = str(out_path).split(".h5")
    assert len(out_path)==2
    key = out_path[1] if key is None else key
    out_path = Path(out_path[0]+".h5")
    key = None if not key else key
    mode = kwargs.pop("mode", "a")
    out_dir = out_path.parent
    if not out_dir.is_dir():
        out_dir.mkdir(parents=True, exist_ok=True)
    df.to_hdf(out_path, key=key, mode=mode, **kwargs)


def run(data_dir, out_dir):
    print_title("Processing De Morton data:")
    print("    data_dir:", data_dir)
    print("    out_dir:", out_dir)
    print()
    code = re.compile("^([0-9]{3})-([0-9]).xlsx$")
    data = defaultdict(list)
    files = sorted(data_dir.glob("*.xlsx"))
    progress = create_progress_bar(label="Reading...", size=len(files))
    progress.start()
    for i, path in enumerate(files):
        ret = code.match(path.name)
        if not ret:
            msg = "Info: Skipping file with invalid name: %s"
            print(msg % path.name)
            continue
        pid = ret.group(1)
        day = ret.group(2)
        df = load_labels(path)
        df.insert(0, "Day", day)
        df.insert(0, "Patient", pid)
        data[pid].append(df)
        progress.update(i)
    progress.finish()

    progress = create_progress_bar(label="Writing...", size=len(data))
    progress.start()
    for i, pid in enumerate(data):
        df = pd.concat(data[pid], axis=0)
        path_csv = out_dir / "exercises" / f"{pid}.csv"
        write_csv(df=df, out_path=path_csv)
        path_hdf = out_dir / "store" / f"{pid}.h5/exercises"
        write_hdf(df=df, out_path=path_hdf)
        progress.update(i)
    progress.finish()

    data_flat = itertools.chain(*list(data.values()))
    df = pd.concat(data_flat, axis=0)
    df.to_csv(out_dir / "exercises" / "_all.csv", index=False)
    print("Done!")


if __name__ == "__main__":
    data_root = Path("/Users/norman/workspace/education/phd/data/wearables")
    data_dir = data_root / "studies/usb-imove/original/exercises"
    out_dir = Path("../results/preprocessed_new")
    run(data_dir=data_dir, out_dir=out_dir)
