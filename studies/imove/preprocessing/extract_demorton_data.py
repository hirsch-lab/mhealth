import shutil
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
from collections import defaultdict

import context
from mhealth.utils.commons import print_title
from mhealth.utils.context_info import dump_context
from mhealth.utils.commons import create_progress_bar
from mhealth.utils.file_helper import ensure_dir, write_csv, write_hdf


KEYS = [
    "vital/left",
    "vital/right",
    "raw/left",
    "raw/right"
]

# QUALITY_COLS = [ "QualityClassification", "HRQ" ]
QUALITY_COLS = [ "HRQ" ]

SAMPLING_RATE_VITAL = 1
SAMPLING_RATE_SENSOR = 50

def extract_data(df, delta_minutes):
    # df_dm: where DeMorton was active.
    df_dm = df.loc[df["DeMorton"]]
    delta = timedelta(minutes=delta_minutes)
    g = df_dm.groupby("DeMortonDay")
    starts = g["timestamp"].min() - delta
    stops = g["timestamp"].max() + delta
    # Extract data
    mask = pd.Series(index=df.index, dtype=bool)
    mask[:] = False
    for t0, t1 in zip(starts, stops):
        mask |= ((df["timestamp"]>=t0) & (df["timestamp"]<=t1))
    df = df[mask].copy()
    return df


def filter_time_gaps(df, max_gap_hours, dataset_id):
    def _first_loc_index_where(mask):
        return next((idx for idx, x in zip(mask.index, mask) if x), None)
    def _first_iloc_index_where(mask):
        return next((idx for idx, x in zip(range(len(mask)), mask) if x), None)

    diff = df["timestamp"].diff()
    gaps = diff.dt.total_seconds() > (max_gap_hours*3600)
    # idx is None if no gaps are present
    # .loc[] is inclusive for both upper and lower bound.
    # .iloc[] is not inclusive for upper bound (as usual)
    # Consequences:
    #   .loc[idx:] and .loc[:idx] overlap at idx!
    #   .iloc[idx:] and .iloc[:idx] don't overlap
    # Conclusion: better use iloc[]
    idx = _first_iloc_index_where(gaps)
    if idx is not None:
        delta = diff.iloc[idx].total_seconds()/3600
        gap = pd.Series(False, index=df.index)
        gap[:idx] = True
        msg = "Info: %-16s Found a gap of %3d hours, clipped %5d rows (%.3f%%)"
        print(msg % (dataset_id+":", delta, sum(~gap), (~gap).mean()*100))
        # tmp = df.copy()
        # tmp.insert(1, "gap", gap)
        df = df.loc[gap].copy()
    return df


def quality_filter_vital(df, quality):
    mask = (df[QUALITY_COLS] > quality).all(axis=1)
    mask &= (df["HR"] > 0)
    df = df[mask]
    return df.copy()


def quality_filter(data, side, quality):
    """
    Filter by quality of the vital data. Also exclude the corresponding
    timestamps for the raw sensor data if it is available.
    """

    # See the notes in preprocessing.md!
    vital_key = f"vital/{side}"
    raw_key = f"raw/{side}"
    if vital_key not in data:
        # This apparently holds always
        assert raw_key not in data
        # Nothing to do
        return
    # Filter vital.
    vital = data[vital_key]
    vital = quality_filter_vital(df=vital, quality=quality)
    data[vital_key] = vital
    if raw_key not in data:
        # Nothing to do
        return
    # Filter raw.
    vital_tmp = vital.set_index("timestamp")
    raw = data[raw_key].set_index("timestamp")
    index = raw.index.intersection(vital_tmp.index)
    raw = raw.loc[index]
    data[raw_key] = raw.reset_index().copy()


def measure_info(key, case, group, df, info):
    def _get_info_from_key(key):
        ret = key.split("/")
        assert len(ret)==2, "Expecting keys of shape: <mode>/<side>"
        mode, side = ret
        assert mode in ("vital", "raw")
        assert side in ("left", "right")
        return mode, side

    def _measure_stats(data, info, key, group, name):
        info[key][(group, name, "mean")] = data.mean()
        info[key][(group, name, "std")]  = data.std()
        info[key][(group, name, "min")]  = data.min()
        info[key][(group, name, "25%")]  = data.quantile(0.25)
        info[key][(group, name, "50%")]  = data.median()
        info[key][(group, name, "75%")]  = data.quantile(0.75)
        info[key][(group, name, "max")]  = data.max()

    def _measure_vital(df, info, group, key):
        COLS = ["HR", "HRQ", "SpO2", "SpO2Q", "Activity",
                "QualityClassification", "ObjTemp",]
        for col in COLS:
            _measure_stats(data=df[col], info=info, key=key,
                           group=group, name=col)

    def _measure_both(df, info, group, key, sr):
        n_samples = len(df)
        ts_min = df["timestamp"].min()
        ts_max = df["timestamp"].max()
        ts_diff = (ts_max-ts_min).total_seconds()
        ts_delta = df["timestamp"].diff(periods=1).dt.total_seconds()/3600

        info[key][(group, "Counts", "nSamples")] = n_samples
        info[key][(group, "Time", "Start")] = ts_min
        info[key][(group, "Time", "End")] = ts_max
        info[key][(group, "Time", "ValidHours")] = n_samples/3600./sr
        info[key][(group, "Time", "TotalHours")] = ts_diff/3600
        info[key][(group, "TimeGaps", "MaxGap")] = ts_delta.max()
        info[key][(group, "TimeGaps", "nGaps>1m")]  = (ts_delta>(1/60)).sum()
        info[key][(group, "TimeGaps", "nGaps>2m")]  = (ts_delta>(1/30)).sum()
        info[key][(group, "TimeGaps", "nGaps>5m")]  = (ts_delta>(1/12)).sum()
        info[key][(group, "TimeGaps", "nGaps>10m")] = (ts_delta>(1/6)).sum()
        info[key][(group, "TimeGaps", "nGaps>30m")] = (ts_delta>(1/2)).sum()
        info[key][(group, "TimeGaps", "nGaps>1h")]  = (ts_delta>1).sum()
        info[key][(group, "TimeGaps", "nGaps>3h")]  = (ts_delta>3).sum()
        info[key][(group, "TimeGaps", "nGaps>6h")]  = (ts_delta>6).sum()
        info[key][(group, "TimeGaps", "nGaps>12h")] = (ts_delta>12).sum()
        info[key][(group, "TimeGaps", "nGaps>24h")] = (ts_delta>24).sum()
        info[key][(group, "TimeGaps", "nGaps>36h")] = (ts_delta>36).sum()

    def _measure_empty(info, key):
        info[key] = None

    mode, side = _get_info_from_key(key)
    if df is None:
        _measure_empty(info=info, key=(case, mode, side))
        return
    else:
        sr = SAMPLING_RATE_SENSOR if mode=="raw" else SAMPLING_RATE_VITAL
        _measure_both(df=df, info=info, group=group,
                      key=(case, mode, side), sr=sr)
        if mode=="vital":
            _measure_vital(df=df, info=info, group=group,
                           key=(case, mode, side))


def measure_infos(case, group, data, info):
    for key in KEYS:
        df = data.get(key, None)
        measure_info(key=key, case=case, group=group, df=df, info=info)


def extract_data_store(filepath, delta_minutes, quality, max_gap, info):
    # Structure: {pat_id}.h5/{mode}/{side}"
    store = pd.HDFStore(filepath, mode="r")
    case = filepath.stem
    data = {}
    for key in KEYS:
        dataset_id = f"{filepath.stem}/{key}"
        if key not in store:
            continue
        df = store[key]
        df = filter_time_gaps(df=df, max_gap_hours=max_gap,
                              dataset_id=dataset_id)

        # Measure the effect of quality filtering on the entire data, that
        # is, before extraction of De Morton data. Only do this for vital
        # data. This was added to reproduce legacy code (SanityChecker).
        measure_info(key=key, case=case, group="original", df=df, info=info)
        if "vital" in key:
            dfq = quality_filter_vital(df=df, quality=quality)
            measure_info(key=key, case=case, group="original_filtered",
                         df=dfq, info=info)


        if delta_minutes is not None:
            df = extract_data(df, delta_minutes=delta_minutes)
        data[key] = df
    data["exercises"] = store.get("exercises")
    store.close()

    # Quality filtering, this filters both vital and acc:
    # The actual filter takes place on the data frame for vital data,
    # the accelerometer data will be filtered on the timestamp.
    # This is step is relatively slow.
    measure_infos(case=case, group="extraction", data=data, info=info)
    quality_filter(data=data, side="left", quality=quality)
    quality_filter(data=data, side="right", quality=quality)
    measure_infos(case=case, group="extraction_filtered", data=data, info=info)
    return data


def write_extracted_data(out_dir, case, data, with_csv):
    for key, df in data.items():
        name_csv = case + "_" + key.lower().replace("/", "_") + ".csv"
        path_csv = out_dir / "csv" / case / name_csv
        path_hdf = out_dir / "store" / (case+".h5")
        # Writing .csv is relatively slow
        if with_csv:
            write_csv(df=df, path=path_csv)
        write_hdf(df=df, path=path_hdf, key=key)


def run(args):
    data_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    delta_minutes = args.margin
    quality = args.quality
    max_gap = args.max_gap
    with_csv = args.csv
    dump_context(out_dir=out_dir)

    print_title("Extracting De Morton data:")
    print("    data_dir:", data_dir)
    print("    out_dir:", out_dir)
    print("    margin: ±%ss" % delta_minutes)
    print()

    files = list(sorted((data_dir/"store").glob("*.h5")))
    if not files:
        print("Error: No files in data folder:", data_dir)
    out_dir_store = out_dir / "store"
    ensure_dir(out_dir_store)
    progress = create_progress_bar(label=None,
                                   size=len(files),
                                   prefix="Patient {variables.file:<3}... ",
                                   variables={"file": "Processing... "})
    progress.start()
    info = defaultdict(dict)
    for i, filepath in enumerate(files):
        progress.update(i, file=filepath.stem)
        data = extract_data_store(filepath=filepath,
                                  delta_minutes=delta_minutes,
                                  quality=quality, max_gap=max_gap,
                                  info=info)
        write_extracted_data(out_dir=out_dir, case=filepath.stem,
                             data=data, with_csv=with_csv)
    progress.finish()

    # Copy the exercises file as well
    src = data_dir / "exercises" / "_all.csv"
    dst = out_dir / "exercises.csv"
    shutil.copy(src=src, dst=dst)

    print("Done!")

    info = pd.DataFrame(info).T
    info.index.names = ["Patient", "Mode", "Side"]
    for mode in info.index.levels[1]:
        for group in info.columns.levels[0]:
            out_path = out_dir / f"summary_{mode}_{group}.csv"
            df = info.loc[pd.IndexSlice[:, mode], group]
            df = df.dropna(how="all", axis=1)
            df.to_csv(out_path)


def parse_args():
    description = ("Extract data for the De Morton exercise sessions.")
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(add_help=False,
                                     formatter_class=formatter,
                                     description=description)
    parser.add_argument("-h", "--help", action="help",
                        help="Show this help text")
    parser.add_argument("-i", "--in-dir", required=True,
                        help="Input directory")
    parser.add_argument("-o", "--out-dir", default="../output/extracted",
                        help="Output directory")
    parser.add_argument("--csv", action="store_true",
                        help="Write also .csv files, besides HDF stores.")
    parser.add_argument("--quality", default=50, type=float,
                        help="Threshold for quality filtering")
    parser.add_argument("--margin", default=15,
                        type=lambda x: None if x in ("", "None", "none") else float(x),
                        help=("Time margin in minutes to collect extra before "
                              "and after the De Morton exercise sessions. "
                              "Empty string or 'None' disables the clipping "
                              "of data around De Morton sessions."))
    parser.add_argument("--max-gap", default=36, type=float,
                        help=("Maximal time gap tolerated, in hours. Data "
                              "after an extremal time gap are clipped."))
    parser.set_defaults(func=run)
    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

