"""
Visualize the sensor data for the De Morton exercises.
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import context
from mhealth.utils.commons import print_title
from mhealth.utils.context_info import dump_context
from mhealth.utils.commons import create_progress_bar
from mhealth.patient.imove_label_loader import merge_labels
from mhealth.utils.file_helper import ensure_dir, write_hdf

DEFAULT_COLUMNS = [ "HR", "AX", "AY", "AZ", "A" ]

# Update if metrics are not available.
METRICS_AT_50HZ = { "AX", "AY", "AZ", "A" }
METRICS_AT_01HZ = { "HR" }




def read_data(data_dir, out_dir, columns, resample,
              side="both", forced=False):

    def _split_by_sampling_rate(columns):
        _01Hz, _50Hz = [], []
        for col in columns:
            if col in METRICS_AT_50HZ:
                _50Hz.append(col)
            elif col in METRICS_AT_01HZ:
                _01Hz.append(col)
            else:
                msg = "Update METRICS_AT_01HZ or METRICS_AT_50HZ with: %s"
                assert False, msg % col
        return _01Hz, _50Hz

    def _append_data_side(dfs, store, group, side, key,
                          pat_id, cols, resample):
        if key in store:
            df = store.get(key)
            df = df.reset_index(drop=True)
            df = df.set_index("timestamp")
            df = _derived_metrics_01Hz(df=df, group=group)
            df = _derived_metrics_50Hz(df=df, group=group)
            df = df[cols].copy()
            # Only a bit slow (1-2s, if not a no-op)
            df = _resample(df=df, resample=resample, group=group)
            assert("exercises" in store)
            # We cannot resample labels. Therefore re-extract the
            # De Morton labels. Unfortunately, this step is slow.
            df = merge_labels(df=df, df_labels=store.get("exercises"))
            df["Side"] = side
            df["Patient"] = pat_id
            dfs.append(df)

    def _append_data(dfs, store, group, side, pat_id, cols, resample):
        assert group in ("vital", "raw")
        if side in ("left", "both"):
            key = f"{group}/left"
            _append_data_side(dfs=dfs, store=store, group=group, side="left",
                              key=key, pat_id=pat_id, cols=cols,
                              resample=resample)
        if side in ("right", "both"):
            key = f"{group}/right"
            _append_data_side(dfs=dfs, store=store, group=group, side="right",
                              key=key, pat_id=pat_id, cols=cols,
                              resample=resample)
        else:
            assert False

    def _resample(df, resample, group):
        if group == "vital" and resample<1:
            return df
        if group == "raw" and resample<(1/50):
            return df
        msg = "Currently, only integral values for integral are possible."
        assert resample==int(resample), msg
        if resample and resample>0:
            df = df.resample("%ds" % resample).mean()

        return df

    def _derived_metrics_01Hz(df, group):
        if group=="vital":
            pass
        return df

    def _derived_metrics_50Hz(df, group):
        if group=="raw":
            df.loc[:, "A"] = np.linalg.norm(df[["AX", "AY", "AZ"]].values,
                                            axis=1)
        return df

    def _read_data_stores(data_dir, cols_01Hz, cols_50Hz,
                          resample, side):
        dfs_01Hz = []
        dfs_50Hz = []

        print("Reading data...")
        files = list(sorted(Path(data_dir).glob("*.h5")))
        prefix = "Patient {variables.pat_id:<3}... "
        progress = create_progress_bar(label=None,
                                       size=len(files),
                                       prefix=prefix,
                                       variables={"pat_id": "N/A"})
        progress.start()
        for i, path in enumerate(files):
            pat_id = path.stem
            progress.update(i, pat_id=pat_id)
            store = pd.HDFStore(path, mode="r")
            if cols_01Hz:
                _append_data(dfs=dfs_01Hz, store=store,
                             group="vital", side=side,
                             pat_id=pat_id, cols=cols_01Hz,
                             resample=resample)
            if cols_50Hz:
                _append_data(dfs=dfs_50Hz, store=store,
                             group="raw", side=side,
                             pat_id=pat_id, cols=cols_50Hz,
                             resample=resample)
            store.close()
        progress.finish()
        print("Done!")

        print("Concatenating data...")
        dfs_01Hz = [df for df in dfs_01Hz if df is not None]
        dfs_50Hz = [df for df in dfs_50Hz if df is not None]
        if dfs_01Hz:
            df_01Hz = pd.concat(dfs_01Hz, axis=0)
        if dfs_50Hz:
            df_50Hz = pd.concat(dfs_50Hz, axis=0)
        print("Done!")
        return df_01Hz, df_50Hz

    def _read_data_lazily(out_dir, cols_01Hz, cols_50Hz):
        filepath = out_dir / "demorton.h5"
        if not filepath.is_file():
            return None, None
        print("Reading data lazily...")
        store = pd.HDFStore(filepath, mode="r")
        df_vital = store["vital"]
        df_raw = store["raw"]
        store.close()
        if set(cols_01Hz) - set(df_vital.columns):
            # Force re-reading.
            df_vital = None
        if set(cols_50Hz) - set(df_raw.columns):
            # Force re-reading.
            df_raw = None
        print("Done!")
        return df_vital, df_raw

    def _save_data(out_dir, df_vital, df_raw):
        print("Writing data...")
        filepath = out_dir / "demorton.h5"
        write_hdf(df=df_vital, path=filepath, key="vital")
        write_hdf(df=df_raw, path=filepath, key="raw")
        print("Done!")


    #######################################################

    if columns is None:
        columns = list(DEFAULT_COLUMNS)
    assert side in ("left", "right", "both")
    cols_01Hz, cols_50Hz = _split_by_sampling_rate(columns)

    df_vital = df_raw = None
    if not forced:
        df_vital, df_raw = _read_data_lazily(out_dir=out_dir,
                                             cols_01Hz=cols_01Hz,
                                             cols_50Hz=cols_50Hz)
    if df_vital is None or df_raw is None:
        df_vital, df_raw = _read_data_stores(data_dir=data_dir,
                                             cols_01Hz=cols_01Hz,
                                             cols_50Hz=cols_50Hz,
                                             resample=resample,
                                             side=side)
        # Save for lazy loading.
        _save_data(out_dir=out_dir,
                   df_vital=df_vital,
                   df_raw=df_raw)

    return df_vital, df_raw


def run(args):
    data_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    metrics = args.metrics
    forced = args.force_read
    dump_context(out_dir=out_dir)

    print_title("Analyzing De Morton exercises:")
    print("    data_dir:", data_dir)
    print("    out_dir:", out_dir)
    print()
    df_vital, df_raw = read_data(data_dir=data_dir,
                                 out_dir=out_dir,
                                 columns=metrics,
                                 resample=args.resample,
                                 side=args.side,
                                 forced=forced)


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
    parser.add_argument("--resample", type=float, default=1.,
                        help="Resampling period, in seconds. Default: 1.")
    parser.add_argument("--metrics", default=None, nargs="+",
                        help="Select a subset of metrics for the analysis")
    parser.add_argument("--side", default="both", type=str,
                        choices=("left", "right", "both"),
                        help="Select the side of the device")
    parser.add_argument("-f", "--force-read", action="store_true",
                        help="Enforce reading of data-stores")
    parser.set_defaults(func=run)
    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
