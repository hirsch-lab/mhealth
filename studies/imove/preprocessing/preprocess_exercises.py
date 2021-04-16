"""
This script was used to process the manual measurements of De Morton exercises.

Input:  Folder with .xlsx files with the following pattern: <pid>-<day>.xlsx
        The files contain time measurements for the De Morton exercises.
Output: A single table per patient with proper format
        The files are written to:   - out_dir/exercises/ as .csv
                                    - out_dir/store/     as .h5
"""

import re
import argparse
import itertools
import pandas as pd
from pathlib import Path
from collections import defaultdict

import context
from mhealth.utils.commons import create_progress_bar, print_title
from mhealth.patient.imove_label_loader import load_labels
from mhealth.utils.file_helper import write_csv, write_hdf

def run(args):
    data_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
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
        df["StartDate"] = df["StartDate"].dt.tz_convert("UTC")
        df["EndDate"] = df["EndDate"].dt.tz_convert("UTC")
        data[pid].append(df)
        progress.update(i)
    progress.finish()

    progress = create_progress_bar(label="Writing...", size=len(data))
    progress.start()
    for i, pid in enumerate(data):
        df = pd.concat(data[pid], axis=0)
        path_csv = out_dir / "exercises" / f"{pid}.csv"
        write_csv(df=df, path=path_csv, index=False)
        path_hdf = out_dir / "store" / f"{pid}.h5/exercises"
        write_hdf(df=df, path=path_hdf)
        progress.update(i)
    progress.finish()

    data_flat = itertools.chain(*list(data.values()))
    df = pd.concat(data_flat, axis=0)
    df.to_csv(out_dir / "exercises" / "_all.csv", index=False)
    print("Done!")


def parse_args():
    description = ("Collect and format timing measurements for "
                   "the De Morton exercises.")
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(add_help=False,
                                     formatter_class=formatter,
                                     description=description)
    parser.add_argument("-h", "--help", action="help",
                        help="Show this help text")
    parser.add_argument("-i", "--in-dir", required=True,
                        help="Input directory")
    parser.add_argument("-o", "--out-dir", default="../output/preprocessed",
                        help="Output directory")
    parser.set_defaults(func=run)
    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
