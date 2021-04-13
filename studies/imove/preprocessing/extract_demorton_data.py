import pandas as pd
from pathlib import Path
from datetime import timedelta

import context
from mhealth.utils.commons import create_progress_bar


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


def extract_data_store(filepath, out_dir, delta_minutes):
    keys = [
        "vital/left",
        "vital/right",
        "raw/left",
        "raw/right"
    ]
    # Structure: {pat_id}.h5/{mode}/{side}"
    store = pd.HDFStore(filepath)
    for key in keys:
        if key not in store:
            continue
        df = extract_data(store[key], delta_minutes=delta_minutes)
        df.to_hdf(out_dir/filepath.name, key=key, format="table")
    store.close()


def run(data_dir, out_dir, delta_minutes):
    files = list(sorted((data_dir/"store").glob("*.h5")))
    if not files:
        print("Error: No files in data folder:", data_dir)
    out_dir = out_dir / "extraction"
    if not out_dir.is_dir():
        out_dir.mkdir(parents=True, exist_ok=True)
    progress = create_progress_bar(label=None,
                                   size=len(files),
                                   prefix='Patient {variables.file:<3}... ',
                                   variables={"file": "Processing... "})
    progress.start()
    for i, filepath in enumerate(files):
        progress.update(i, file=filepath.stem)
        extract_data_store(out_dir=out_dir,
                           filepath=filepath,
                           delta_minutes=delta_minutes)
    progress.finish()
    print("Done!")


if __name__ == "__main__":
    data_dir = Path("../results/preprocessed")
    out_dir = Path("../results/extraction")
    delta_minutes = 15
    run(data_dir=data_dir, out_dir=out_dir,
        delta_minutes=delta_minutes)
