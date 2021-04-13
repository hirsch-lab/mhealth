import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
from collections import defaultdict

import context
from mhealth.utils.commons import create_progress_bar


KEYS = [
    "vital/left",
    "vital/right",
    "raw/left",
    "raw/right"
]

QUALITY_COLS = [ "QualityClassification", "HRQ" ]

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


def quality_filter(data, side, quality):
    # See the notes in preprocessing.md!
    vital_key = f"vital/{side}"
    raw_key = f"raw/{side}"
    if vital_key not in data:
        # This apparently holds always
        assert raw_key not in data
        # Nothing to do
        return
    # Filter vital.
    vital = data[vital_key].set_index("timestamp")
    mask = (vital[QUALITY_COLS] > quality).all(axis=1)
    mask &= (vital["HR"] > 0)
    vital = vital[mask]
    data[vital_key] = vital.reset_index().copy()
    if raw_key not in data:
        # Nothing to do
        return
    # Filter raw.
    raw = data[raw_key].set_index("timestamp")
    index = raw.index.intersection(vital.index)
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

    def _measure_vital(df, info, group, key):
        COLS = ["HR", "HRQ", "SpO2", "SpO2Q", "Activity",
                "QualityClassification", "ObjTemp",]
        for col in COLS:
            info[key][(group, col, "mean")] = df[col].mean()
            info[key][(group, col, "std")] = df[col].std()
            info[key][(group, col, "min")] = df[col].min()
            info[key][(group, col, "25%")] = df[col].quantile(0.25)
            info[key][(group, col, "50%")] = df[col].median()
            info[key][(group, col, "75%")] = df[col].quantile(0.75)
            info[key][(group, col, "max")] = df[col].max()

    def _measure_both(df, info, group, key, sr):
        n_samples = len(df)
        ts_min = df["timestamp"].min()
        ts_max = df["timestamp"].max()
        ts_diff = (ts_max-ts_min).total_seconds()

        info[key][(group, "Counts", "nSamples")] = n_samples
        info[key][(group, "Time", "start")] = ts_min
        info[key][(group, "Time", "stop")] = ts_max
        info[key][(group, "Time", "validHours")] = n_samples/3600./sr
        info[key][(group, "Time", "totalHours")] = ts_diff/3600

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


def extract_data_store(filepath, out_dir,
                       delta_minutes, quality,
                       info):
    # Structure: {pat_id}.h5/{mode}/{side}"
    store = pd.HDFStore(filepath)
    case = filepath.stem
    data = {}
    for key in KEYS:
        if key not in store:
            continue
        df = store[key]
        measure_info(key=key, case=case, group="initial", df=df, info=info)
        df = extract_data(df, delta_minutes=delta_minutes)
        data[key] = df
    store.close()

    measure_infos(case=case, group="before", data=data, info=info)
    quality_filter(data=data, side="left", quality=quality)
    quality_filter(data=data, side="right", quality=quality)
    measure_infos(case=case, group="after", data=data, info=info)


def run(data_dir, out_dir, delta_minutes, quality):
    files = list(sorted((data_dir/"store").glob("*.h5")))
    if not files:
        print("Error: No files in data folder:", data_dir)
    out_dir_store = out_dir / "store"
    if not out_dir_store.is_dir():
        out_dir_store.mkdir(parents=True, exist_ok=True)
    progress = create_progress_bar(label=None,
                                   size=len(files),
                                   prefix="Patient {variables.file:<3}... ",
                                   variables={"file": "Processing... "})
    progress.start()
    info = defaultdict(dict)
    for i, filepath in enumerate(files):
        if i == 1:
            break
        progress.update(i, file=filepath.stem)
        extract_data_store(out_dir=out_dir_store,
                           filepath=filepath,
                           delta_minutes=delta_minutes,
                           quality=quality, info=info)
    progress.finish()
    print("Done!")


    info = pd.DataFrame(info).T
    info.index.names = ["Patient", "Mode", "Side"]
    for mode in info.index.levels[1]:
        for group in info.columns.levels[0]:
            out_path = out_dir / f"filtering_{mode}_{group}.csv"
            info.loc[pd.IndexSlice[:, mode], group].to_csv(out_path)


if __name__ == "__main__":
    data_dir = Path("../results/preprocessed")
    out_dir = Path("../results/extraction")
    delta_minutes = 15
    quality = 50
    run(data_dir=data_dir, out_dir=out_dir,
        delta_minutes=delta_minutes, quality=quality)
