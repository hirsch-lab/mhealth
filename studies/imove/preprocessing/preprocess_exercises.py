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
from mhealth.utils.file_helper import write_csv, write_hdf

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


if __name__ == "__main__":
    data_root = Path("/Users/norman/workspace/education/phd/data/wearables")
    data_dir = data_root / "studies/usb-imove/original/exercises"
    out_dir = Path("../results/preprocessed_new")
    run(data_dir=data_dir, out_dir=out_dir)
