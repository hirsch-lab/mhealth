"""
This script was used to process the Everion data received from
University Hospital Basel.
"""

import shutil
import argparse
import warnings
import pandas as pd
from pathlib import Path
from codetiming import Timer

import context
from mhealth.utils import IOManager
from mhealth.patient import merge_labels
from mhealth.utils.commons import (print_title,
                                   print_subtitle,
                                   catch_warnings,
                                   create_progress_bar)
from mhealth.utils.context_info import dump_context
from mhealth.utils.file_helper import write_csv, write_hdf
from mhealth.data_analysis import filter_bad_quality_mixed_vital_raw

# Preprocessing:
#   - read data
#   - extract columns of interest
#   - keep timestamps in UTC
#   x remove all empty lines
#   - apply quality filter
#   - add de Morton mobility index/label
#   - store in .csv or .h5 format
#   - fix problem of shifted columns

DEFAULT_LOGGER=None
#DEFAULT_LOGGER=print

def _timer_format(label):
    return "{:6.2f}s: "+label


def fix_shifted_columns(df_raw, how):
    """
    For a few rows, somehow the readout is broken such that the values
    of 8 columns are missing. Unfortunately, the missing data is not
    filled with nans, but replaced with the next valid readout,
    causing the table to be shifted for those broken rows. The problem
    occurs relatively rarely (<0.01%, observed only for _raw data), but
    it breaks the dtype-consistency of the columns, which is why the
    problem must be handled.

    Illustrative example:
        val1    val2    val3    val4    val5    val6    val7
        val1    val2    val3    val4    val5    val6    val7
        val1    val2    val3    val4    val5    val6    val7
        val1    val4    val7
        val1    val4    val7
        val1    val2    val3    val4    val5    val6    val7
        val1    val2    val3    val4    val5    val6    val7
        val1    val4    val7

    Possible modes:
        - remove broken lines
        - remove lines with the same timestamp
        - shift the columns

    See protocol.md for further information.
    """
    assert(how in ("remove", "remove_time", "shift"))
    lookup = {
        ("30", "raw_value") : ("30", "raw_value"),          # same
        ("31", "raw_value") : ("31", "raw_value"),          # same
        ("32", "raw_value") : ("32", "raw_value"),          # same
        ("33", "raw_value") : ("33", "raw_value"),          # same
        ("34", "raw_value") : ("34", "raw_value"),          # same
        ("35", "raw_value") : ("35", "raw_value"),          # same
        ("36", "raw_value") : ("36", "raw_value"),          # same

        ("37", "raw_value") : ("30", "value"),              # shifted +4
        ("38", "raw_value") : ("31", "value"),              # shifted +4
        ("39", "raw_value") : ("32", "value"),              # shifted +4
        ("40", "raw_value") : ("33", "value"),              # shifted +4
        ("30", "value")     : ("34", "value"),              # shifted +4
        ("31", "value")     : ("35", "value"),              # shifted +4
        ("32", "value")     : ("36", "value"),              # shifted +4
        ("33", "value")     : ("timestamp", "timestamp")    # shifted +8

        # Missing cols:
        # ("37", "raw_value")
        # ("38", "raw_value")
        # ("39", "raw_value")
        # ("40", "raw_value")
        # ("37", "value")
        # ("38", "value")
        # ("39", "value")
        # ("40", "value")
    }

    shape_in = df_raw.shape
    mask = df_raw["timestamp"].squeeze().isna()
    df_to_shift = df_raw.loc[mask, lookup.keys()].copy()
    if how == "remove":
        df_raw = df_raw[~mask].copy()
    elif how == "remove_time":
        # (Use lookup in backward direction)
        timestamp_col0 = ("timestamp", "timestamp")
        timestamp_col = list(lookup.values()).index(timestamp_col0)
        timestamp_col = list(lookup.keys())[timestamp_col]
        times = df_raw.loc[mask, timestamp_col].unique()
        # Mask 1
        df_raw = df_raw[~mask].copy()
        # Mask 2
        mask2 = df_raw[timestamp_col0].isin(times)
        df_raw = df_raw[~mask2]
    elif how == "shift":
        df_raw.loc[mask] = None
        df_raw.loc[mask, lookup.values()] = \
            df_to_shift.loc[:, lookup.keys()].values
    shape_out = df_raw.shape
    if shape_in[0] != shape_out[0]:
        diff = shape_in[0] - shape_out[0]
        print("Warning: removed %d inconsistent rows." % diff)
    elif mask.sum() > 0:
        print("Warning: fixed %d inconsistent rows." % mask.sum())
    assert shape_in[1]==shape_out[1]
    if how is not None:
        assert df_raw[("timestamp", "timestamp")].isna().sum() == 0
    return df_raw


def read_lookup(path):
    lookup = pd.read_csv(path, header=None, skipinitialspace=True, dtype=str)
    lookup.columns = ["index", "name", "dtype"]
    lookup = lookup.set_index("index")
    return lookup


@Timer(text=_timer_format("read .csv"), logger=DEFAULT_LOGGER)
def read_data(path, col_lookup, mode):
    assert mode in ("raw", "vital")

    if path.stat().st_size <= 0:
        print("Warning: encountered an empty file (%s)" % path.name)
        return pd.DataFrame()
    # (1) read csv
    with warnings.catch_warnings(record=True) as ws:
        # Setting low_memory=False to avoid a rare exception.
        # https://github.com/pandas-dev/pandas/issues/40587
        df = pd.read_csv(path, sep=";", header=[0,1], low_memory=False)
        catch_warnings(ws, warning_to_catch=pd.errors.DtypeWarning,
                       message_to_catch="have mixed types",
                       enabled=(mode=="raw"))
    # (2) fix inconsistently formatted rows
    if mode=="raw":
        df = fix_shifted_columns(df, how="remove")
    # (3) drop raw data, use only values represented in physical units
    df.drop("raw_value", level=1, axis=1, inplace=True)
    df.columns = df.columns.droplevel(level=1)
    # (4) rename columns
    df.rename(col_lookup["name"], axis=1, inplace=True)
    col_lookup = col_lookup.reset_index().set_index("name")
    # (5) ensure dtypes (this also converts timestamp)
    #     df["timestamp"] = pd.to_datetime(df["timestamp"])
    #     df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
    df = df.astype(col_lookup["dtype"][col_lookup.index.isin(df.columns)])
    assert df["timestamp"].dtype == "datetime64[ns, UTC]"
    col_diff = set(df.columns)-set(col_lookup.index)
    assert len(col_diff)==0, ("Unknown columns occurred: %s" % col_diff)
    # Move to front
    col = df.pop("timestamp")
    df.insert(0, col.name, col)
    return df


@Timer(text=_timer_format("write .csv"), logger=DEFAULT_LOGGER)
def write_csv_timed(df, path, **kwargs):
    return write_csv(df=df, path=path, index=False, **kwargs)


@Timer(text=_timer_format("write .hdf"), logger=DEFAULT_LOGGER)
def write_hdf_timed(df, path, key=None, **kwargs):
    return write_hdf(df=df, path=path, key=key, **kwargs)


@Timer(text=_timer_format("filter quality"), logger=DEFAULT_LOGGER)
def filter_by_quality(df, quality):
    if quality is not None:
        # In-place.
        filter_bad_quality_mixed_vital_raw(df=df, min_quality=quality)
    return df


@Timer(text=_timer_format("load labels"), logger=DEFAULT_LOGGER)
def load_label_data(df, labels_dir, pat_id, side, **kwargs):
    exercise_file = labels_dir / f"{pat_id}.csv"
    if not exercise_file.is_file():
        msg = ("No exercise files are present. Skipping the calculation of "
               "De Morton labels. Run preprocess_exercises.py first to not "
               "see this warning.")
        warnings.warn(msg)
        return df
    df_exercises = pd.read_csv(exercise_file)
    df = merge_labels(df=df, df_labels=df_exercises)
    return df


def preprocess_files(mode, files, col_lookup_file, labels_dir,
                     use_cols, quality, iom):
    col_lookup = read_lookup(path=col_lookup_file)
    size = shutil.get_terminal_size()
    progress = create_progress_bar(label=None,
                                   size=len(files),
                                   prefix="{variables.file:<36}",
                                   variables={"file": "Processing...",
                                              "mode": mode,
                                              "pat_id": "",
                                              "side": ""})
    progress.start()
    for i, filepath in enumerate(files):
        iom.set_current(filepath, mode=mode)
        pat_id = iom.info.get("pat_id")
        side = iom.info.get("side")
        progress.update(i, file=filepath.stem, pat_id=pat_id, side=side)
        if iom.skip_existing():
            continue
        df = read_data(path=filepath, col_lookup=col_lookup, mode=mode)
        if df.empty:
            print("Warning: Encountered empty file (%s)" % filepath.name)
            continue
        df = filter_by_quality(df=df, quality=quality)
        df = load_label_data(df=df, labels_dir=labels_dir,
                             pat_id=pat_id, side=side)
        if use_cols:
            # if "DeMorton" not in df.columns and "DeMorton" in use_cols:
            #     use_cols = list(use_cols)
            #     use_cols.remove("DeMorton")
            #     use_cols.remove("DeMortonDay")
            #     use_cols.remove("DeMortonLabel")
            df = df[use_cols].copy()
        iom.write_data(data=df)
    progress.finish()


def preprocess(mode, data_dir, glob_expr,
               col_lookup_file, labels_dir,
               use_cols, quality, iom):
    """
    mode:   "raw": sensor data at (50Hz)
            "vital": vital data signals (at 1Hz)
    """
    mode = "raw" if mode=="raw_sensor" else mode
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(glob_expr))
    preprocess_files(mode=mode, files=files,
                     col_lookup_file=col_lookup_file,
                     labels_dir=labels_dir,
                     use_cols=use_cols,
                     quality=quality,
                     iom=iom)


def run(args):
    data_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    with_csv = args.csv
    dump_context(out_dir=out_dir)

    print_title("Processing Everion data:")
    print("    data_dir:", data_dir)
    print("    out_dir:", out_dir)
    print()

    col_file = Path("./everion_columns.csv")
    labels_dir = out_dir / "exercises"

    # The filenames look as follows:
    # vital signs:
    #   - iMove_001_storage-vital__left.csv
    #   - iMove_001_storage-vital__right.csv
    # raw sensor data:
    #   - iMove_001_storage-vital_raw__left.csv
    #   - iMove_001_storage-vital_raw__right.csv

    # Extract information from the filenames.
    info_patterns = {
        "pat_id": "iMove_([0-9]{3})_.*__(?:left|right).*",
        "side": "iMove_[0-9]{3}_.*__(left|right).*",
        "side_short": "iMove_[0-9]{3}_.*__(left|right).*",
    }
    info_transformers = {
        # Transformation: {left, right} -> {L, R}
        "side_short": lambda ret: ret.group(1)[0].upper()
    }
    # Write methods.
    target_writers = {
        ".h5": write_hdf_timed
    }
    if with_csv:
        target_writers[".csv"] = write_csv_timed

    # Construct filenames out of parts.
    target_names = {
        ".csv": "sensor/{pat_id}{side_short}-{mode}.csv",
        ".h5":  "store/{pat_id}.h5/{mode}/{side}"
    }
    # Works only for target .csv. (Checks inside a .h5 file not possible yet)
    skip_existing = False

    if True:
        print_subtitle("Vital signals (no quality filtering)")
        use_cols = [ "timestamp", "HR", "HRQ", "SpO2", "SpO2Q", "BloodPressure",
                     "BloodPerfusion", "Activity", "Classification",
                     "QualityClassification", "RespRate", "HRV", "LocalTemp",
                     "ObjTemp",
                     "DeMortonLabel", "DeMortonDay", "DeMorton",
                    ]
        iom = IOManager(out_dir=out_dir,
                        info_patterns=info_patterns,
                        info_transformers=info_transformers,
                        target_writers=target_writers,
                        target_names=target_names,
                        skip_existing=skip_existing,
                        dry_run=False)
        preprocess(mode="vital",
                   data_dir=data_dir,
                   glob_expr="*vital__*.csv",
                   col_lookup_file=col_file,
                   labels_dir=labels_dir,
                   use_cols=use_cols,
                   quality=None,
                   iom=iom)

    if True:
        print_subtitle("Raw sensor data (no quality filtering)")
        use_cols = [ "timestamp", "AX", "AY", "AZ",
                     "DeMortonLabel", "DeMortonDay", "DeMorton" ]
        iom = IOManager(out_dir=out_dir,
                        info_patterns=info_patterns,
                        info_transformers=info_transformers,
                        target_writers=target_writers,
                        target_names=target_names,
                        skip_existing=skip_existing,
                        dry_run=False)
        # Raw sensor data cannot be quality-filtered (quality=None).
        preprocess(mode="raw",
                   data_dir=data_dir,
                   glob_expr="*vital_raw__*.csv",
                   col_lookup_file=col_file,
                   labels_dir=labels_dir,
                   use_cols=use_cols,
                   quality=None,
                   iom=iom)


def parse_args():
    description = ("Collect and format data from Everion devices.")
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
    parser.add_argument("--csv", action="store_true",
                        help="Enable .csv output")
    parser.set_defaults(func=run)
    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
