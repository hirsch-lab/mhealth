"""
This script was used to process the Everion data received from
University Hospital Geneva
"""
import re
import shutil
import pandas as pd
from pathlib import Path

import context
from utils.file_helper import FileHelper
from data_analysis.quality_filter import filter_bad_quality_mixed_vital_raw
from patient.imove_label_loader import load_labels_for_patient, merge_labels

# Preprocessing:
#   - read data
#   - extract columns of interest
#   - keep timestamps in UTC
#   x remove all empty lines
#   - apply quality filter
#   - add de Morton mobility index/label

def read_data(path, col_lookup):
    if path.stat().st_size <= 0:
        print("Warning: encountered an empty file (%s)" % path.name)
        return pd.DataFrame()
    df = pd.read_csv(path, sep=";", header=[0,1])
    df.drop("raw_value", level=1, axis=1, inplace=True)
    df.columns = df.columns.droplevel(level=1)
    df.rename(col_lookup, axis=1, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Just to be explicit (it"s already UTC I think)
    df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
    return df


def read_lookup(path):
    lookup = pd.read_csv(path, header=None, skipinitialspace=True, dtype=str)
    lookup.columns = ["index", "name"]
    lookup = lookup.set_index("index").squeeze()
    return lookup


def extract_pat_id(filename):
    pattern = "iMove_([0-9]{3})_.*__(left|right).*"
    ret = re.match(pattern, filename)
    assert ret is not None, ("Expected file pattern: " % pattern)
    pat_id = ret.group(1)
    side = ret.group(2)
    return pat_id, side


def get_out_filepath(out_dir, pat_id, side):
    lookup = {"left": "L", "right": "R"}
    filename = "%s%s_storage-vital.csv" % (pat_id, lookup[side])
    return out_dir / filename


def preprocess(data_dir, labels_dir, out_dir, col_lookup_file, quality):

    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    files_left = list(data_dir.glob("*vital__left.csv"))
    files_right = list(data_dir.glob("*vital__right.csv"))
    files = sorted(files_left + files_right)
    col_lookup = read_lookup(path=col_lookup_file)

    size = shutil.get_terminal_size()
    for i, filepath in enumerate(files):
        print("Processing %s..." % filepath.name)
        pat_id, side = extract_pat_id(filepath.name)
        df = read_data(path=filepath, col_lookup=col_lookup)
        if df.empty:
            print("Warning: Encountered empty file (%s)" % filepath.name)
            continue
        out_path = get_out_filepath(out_dir=out_dir, pat_id=pat_id, side=side)
        if quality is not None:
            # In-place.
            filter_bad_quality_mixed_vital_raw(df=df, min_quality=quality)
        df_labels = load_labels_for_patient(labels_dir=labels_dir, pat_id=pat_id)
        merge_labels(df=df, df_labels=df_labels)
        if True:
            FileHelper.get_out_dir(out_dir=out_dir)
            df.to_csv(out_path)


def print_title(title, width=80):
    print()
    print("#"*width)
    print(title)
    print("#"*width)
    print()


def run():
    data_dir = "/Users/norman/workspace/education/phd/data/wearables/studies/usb-imove/raw_data"
    labels_dir = "/Users/norman/workspace/education/phd/data/wearables/studies/usb-imove/processed_data/cleaned_labels"
    out_dir = "./results/preprocessed_data"
    col_file = "./everion_columns.csv"

    print_title("Clean data without quality filtering")
    out_sub_dir = Path(out_dir) / "cleaned2_labeled"
    preprocess(data_dir=data_dir, labels_dir=labels_dir, out_dir=out_sub_dir,
               col_lookup_file=col_file, quality=None)

    print_title("Clean data with quality filter 50")
    out_sub_dir = Path(out_dir) / "cleaned2_labeled_quality_filtered_50"
    preprocess(data_dir=data_dir, labels_dir=labels_dir, out_dir=out_sub_dir,
               col_lookup_file=col_file, quality=50)

    print_title("Clean data with quality filter 80")
    out_sub_dir = Path(out_dir) / "cleaned2_labeled_quality_filtered_80"
    preprocess(data_dir=data_dir, labels_dir=labels_dir, out_dir=out_sub_dir,
               col_lookup_file=col_file, quality=80)

if __name__ == "__main__":
    run()
