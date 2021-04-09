"""
This script was used to process the Everion data received from
University Hospital Basel
"""
import shutil
import warnings
import pandas as pd
from pathlib import Path
from codetiming import Timer

import context
from mhealth.utils import FileHelper, IOManager
from mhealth.data_analysis import filter_bad_quality_mixed_vital_raw
from mhealth.patient import load_labels_for_patient, merge_labels

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


def catch_warnings(ws, warning_to_catch=None,
                   message_to_catch=None, enabled=True):
    """
    Print warnings as usual, except the ones in warning_to_catch.
    """
    caught_warnings = 0
    if warning_to_catch is None:
        warning_to_catch = []
    if message_to_catch is None:
        message_to_catch = []
    elif isinstance(message_to_catch, str):
        message_to_catch = [message_to_catch]
    for w in ws:
        if (not issubclass(w.category, warning_to_catch) or
            not any(m in str(w.message) for m in message_to_catch) or
            not enabled):
            warnings.warn_explicit(message=w.message,
                                   category=w.category,
                                   filename=w.filename,
                                   lineno=w.lineno)
        else:
            caught_warnings += 1
    return bool(caught_warnings)


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
    return df


@Timer(text=_timer_format("write .csv"), logger=DEFAULT_LOGGER)
def write_csv(df, out_path, **kwargs):
    sep = kwargs.pop("sep", ",")
    with_index = kwargs.pop("index", False)
    df.to_csv(out_path, sep=sep, index=with_index, **kwargs)


@Timer(text=_timer_format("write .hdf"), logger=DEFAULT_LOGGER)
def write_hdf(df, out_path, key=None, **kwargs):
    """
    Convention: out_path = "path/to/file.h5/sub/path"
                is equivalent to
                out_path = "path/to/file.h5"
                key = "sub/path" if key is None else key
    """
    out_path = str(out_path).split(".h5")
    assert len(out_path)==2
    key = out_path[1] if key is None else key
    out_path = out_path[0]+".h5"
    key = None if not key else key
    mode = kwargs.pop("mode", "a")
    df.to_hdf(out_path, key=key, mode=mode, **kwargs)


@Timer(text=_timer_format("filter quality"), logger=DEFAULT_LOGGER)
def filter_by_quality(df, quality):
    if quality is not None:
        # In-place.
        filter_bad_quality_mixed_vital_raw(df=df, min_quality=quality)
    return df


@Timer(text=_timer_format("filter quality"), logger=DEFAULT_LOGGER)
def filter_quality(df, quality):
    if quality is not None:
        # In-place.
        filter_bad_quality_mixed_vital_raw(df=df, min_quality=quality)
    return df


@Timer(text=_timer_format("load labels"), logger=DEFAULT_LOGGER)
def load_label_data(df, labels_dir, pat_id, side, **kwargs):
    df_labels = load_labels_for_patient(labels_dir=labels_dir, pat_id=pat_id)
    df = merge_labels(df=df, df_labels=df_labels)
    df = df.astype({"DeMorton": bool, "DeMortonLabel": str})
    return df


def preprocess_files(mode, files, col_lookup_file, labels_dir,
                     use_cols, quality, iom):
    col_lookup = read_lookup(path=col_lookup_file)
    size = shutil.get_terminal_size()
    for i, filepath in enumerate(files):
        iom.set_current(filepath, mode=mode)
        if iom.skip_existing():
            continue
        print("Processing %s..." % filepath.name)
        df = read_data(path=filepath, col_lookup=col_lookup, mode=mode)
        if df.empty:
            print("Warning: Encountered empty file (%s)" % filepath.name)
            continue
        df = filter_by_quality(df=df, quality=quality)
        pat_id = iom.info.get("pat_id")
        side = iom.info.get("side")
        df = load_label_data(df=df, labels_dir=labels_dir,
                             pat_id=pat_id, side=side)
        if use_cols:
            df = df[use_cols].copy()
        iom.write_data(data=df)


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


def print_title(title, width=80):
    print()
    print("#"*width)
    print(title)
    print("#"*width)
    print()


def run():
    data_dir = "/Users/norman/workspace/education/phd/data/wearables/studies/usb-imove/raw_data"
    labels_dir = "/Users/norman/workspace/education/phd/data/wearables/studies/usb-imove/processed_data/cleaned_labels"
    out_dir = "./results/preprocessed_data_new"
    col_file = "./everion_columns.csv"

    info_patterns = {
        "pat_id": "iMove_([0-9]{3})_.*__(?:left|right).*",
        "side": "iMove_[0-9]{3}_.*__(left|right).*",
        "side_short": "iMove_[0-9]{3}_.*__(left|right).*",
    }
    info_transformers = {
        "side_short": lambda ret: ret.group(1)[0].upper()
    }
    target_writers = {
        ".csv": write_csv,
        ".h5": write_hdf
    }
    target_names = {
        ".csv": "{pat_id}{side_short}-{mode}.csv",
        ".h5":  "{pat_id}.h5/{mode}/{side}"
    }
    targets = [
        ".h5",
        #".csv"
    ]

    ###########################################################################
    print_title("Vital signals (no quality filtering)")
    use_cols = [ "HR", "HRQ", "SpO2", "SpO2Q", "BloodPressure",
                 "BloodPerfusion", "Activity", "Classification",
                 "QualityClassification", "RespRate", "HRV", "LocalTemp",
                 "ObjTemp", "timestamp", "DeMortonLabel", "DeMorton", ]
    iom = IOManager(out_dir=out_dir,
                    targets=targets,
                    info_patterns=info_patterns,
                    info_transformers=info_transformers,
                    target_writers=target_writers,
                    target_names=target_names,
                    skip_existing=True,
                    dry_run=False)
    preprocess(mode="vital",
               data_dir=data_dir,
               glob_expr="*vital__*.csv",
               col_lookup_file=col_file,
               labels_dir=labels_dir,
               use_cols=use_cols,
               quality=None,
               iom=iom)

    ###########################################################################
    print_title("Raw sensor data (no quality filtering)")
    use_cols = [ "AX", "AY", "AZ", "timestamp",
                 "DeMortonLabel", "DeMorton" ]
    iom = IOManager(out_dir=out_dir,
                    targets=targets,
                    info_patterns=info_patterns,
                    info_transformers=info_transformers,
                    target_writers=target_writers,
                    target_names=target_names,
                    skip_existing=True,
                    dry_run=False)
    preprocess(mode="raw",
               data_dir=data_dir,
               glob_expr="*vital_raw__*.csv",
               col_lookup_file=col_file,
               labels_dir=labels_dir,
               use_cols=use_cols,
               quality=None,
               iom=iom)

    # print_title("Clean data with quality filter 50")
    # out_sub_dir = Path(out_dir) / "cleaned2_labeled_quality_filtered_50"
    # preprocess(data_dir=data_dir, labels_dir=labels_dir, out_dir=out_sub_dir,
    #            col_lookup_file=col_file, quality=50)

    # print_title("Clean data with quality filter 80")
    # out_sub_dir = Path(out_dir) / "cleaned2_labeled_quality_filtered_80"
    # preprocess(data_dir=data_dir, labels_dir=labels_dir, out_dir=out_sub_dir,
    #            col_lookup_file=col_file, quality=80)

if __name__ == "__main__":
    run()
