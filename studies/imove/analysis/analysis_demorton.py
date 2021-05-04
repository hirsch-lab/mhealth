"""
Visualize the sensor data for the De Morton exercises.

TODOs:
- Should df_vital and df_raw be merged after resampling?
"""
import logging
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

import context
from mhealth.utils.maths import split_contiguous
from mhealth.utils.context_info import dump_context
from mhealth.utils.commons import create_progress_bar
from mhealth.patient.imove_label_loader import merge_labels
from mhealth.utils.file_helper import ensure_dir, write_hdf
from mhealth.utils.commons import print_title, setup_logging
from mhealth.utils.plotter_helper import save_figure, setup_plotting

DEFAULT_COLUMNS = [ "HR", "AX", "AY", "AZ", "A" ]

# Update if metrics are not available.
METRICS_AT_50HZ = { "AX", "AY", "AZ", "A" }
METRICS_AT_01HZ = { "HR" }

logger = logging.getLogger("imove")


###############################################################################

def read_data(data_dir, out_dir, columns, resample,
              side="both", forced=False, n_pats=None,
              pats=None, labels=None):

    def _resample(df, resample, group):
        if group == "vital" and (resample is None or resample<=1):
            return df
        if group == "raw" and (resample is None or resample<=(1/50)):
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

    def _set_timestamp_index(df, group):
        # Before setting the index, introduce sub-second resolution.
        # This applies mainly to the 50Hz data, for which the timestamps
        # unfortunately are resolved only up to seconds.
        # This could go to preprocessing as well, but will also increase
        # (I suppose) the file sizes.
        if group=="raw":
            def _subseconds(series):
                shifts = np.linspace(0,1,len(series), endpoint=False)
                series += pd.to_timedelta(shifts, unit="second")
                return series
            tol = pd.Timedelta(seconds=1)
            chunks = split_contiguous(df["timestamp"], tol=tol, inclusive=True)
            chunks = map(_subseconds, chunks)
            df["timestamp"] = pd.concat(chunks, axis=0)
        elif group=="vital":
            pass
        else:
            assert False, "This case is not implemented yet"
        df = df.set_index("timestamp")
        return df

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
            df = _set_timestamp_index(df=df, group=group)
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

    def _read_data_stores(data_dir, cols_01Hz, cols_50Hz,
                          resample, side, n_pats=None,
                          pats=None, labels=None):
        """
        n_pats is ignored if pats is not None
        """
        dfs_01Hz = []
        dfs_50Hz = []

        logger.info("Reading data...")
        files = list(sorted(Path(data_dir).glob("*.h5")))
        if len(files) == 0:
            msg = "No files HDF stores found under path: %s"
            raise RuntimeError(msg % data_dir)
        prefix = "Patient {variables.pat_id:<3}... "
        n_files = n_pats if n_pats is not None else len(files)
        progress = create_progress_bar(label=None,
                                       size=n_files,
                                       prefix=prefix,
                                       variables={"pat_id": "N/A"})
        progress.start()
        for i, path in enumerate(files):
            pat_id = path.stem
            if pats is not None and pat_id not in pats:
                continue
            elif n_pats is not None and n_pats and i>=n_pats:
                break
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
        logger.info("Done!")

        logger.info("Concatenating data...")
        dfs_01Hz = [df for df in dfs_01Hz if df is not None]
        dfs_50Hz = [df for df in dfs_50Hz if df is not None]
        df_01Hz = None
        df_50Hz = None
        if dfs_01Hz:
            df_01Hz = pd.concat(dfs_01Hz, axis=0)
        if dfs_50Hz:
            df_50Hz = pd.concat(dfs_50Hz, axis=0)
        if labels is not None:
            logger.info("Reading data for De Morton labels: %s",
                        ", ".join(labels))
            df_01Hz = df_01Hz[df_01Hz["DeMortonLabel"].isin(labels)]
            df_50Hz = df_50Hz[df_50Hz["DeMortonLabel"].isin(labels)]
        logger.info("Done!")
        return df_01Hz, df_50Hz

    def _read_data_lazily(out_dir, cols_01Hz, cols_50Hz,
                          n_pats=None, pats=None, labels=None,
                          mode="from_exercises"):
        """
        Arguments:
            mode:   Two modes are available:
                        - "all": read store/demorton.h5
                        - "from_exercises": read from store/ex-*.h5
                    Note: demorton.h5 contains all data that was extracted by
                    preprocessing/extract_demorton_data.py. It allows to select
                    data ±margin around the exercise sessions.
            labels: Optional list of exercises/labels to include. Is ignored
                    if mode != "from_exercises"
        """
        assert mode in ("all", "from_exercises")
        if mode == "all":
            filepath = out_dir / "store" / "demorton.h5"
            if not filepath.is_file():
                return None, None
            logger.info("Reading data lazily...")
            store = pd.HDFStore(filepath, mode="r")
            df_vital = store["vital"]
            df_raw = store["raw"]
        if mode == "from_exercises":
            files = sorted((out_dir / "store").glob("ex*.h5"))
            if not files:
                return None, None
            dfs_vital = []
            dfs_raw = []
            if labels is not None:
                logger.info("Lazy loading data for De Morton labels: %s",
                            ", ".join(labels))
            for filepath in files:
                label = filepath.stem
                label = label.replace("ex-", "")
                if labels is not None and label not in labels:
                    continue
                store = pd.HDFStore(filepath, mode="r")
                dfs_vital.append(store["vital"])
                dfs_raw.append(store["raw"])
            df_vital = pd.concat(dfs_vital, axis=0)
            df_raw = pd.concat(dfs_raw, axis=0)

        pats_vital = df_vital["Patient"]
        pats_unique = pats_vital.unique()  # keeps order of appearance!
        choice = None
        if pats is not None:
            diff = set(pats) - set(pats_unique)
            if len(diff)>0 and labels is None:
                msg = ("Requested patients cannot be loaded lazily.\n"
                       "      Use flag --force-read to avoid this warning.\n"
                       "      Missing patients: %s")
                logger.warning(msg % ", ".join(diff))
            elif len(diff)>0 :
                msg = ("Requested patients cannot be loaded lazily. "
                       "Try flag --force-read to avoid this warning.\n"
                       "      Note: It could be that no data is "
                       "available for the selected labels.\n"
                       "      Missing patients: %s\n"
                       "      Selected labels:  %s")
                logger.warning(msg, ", ".join(diff), ", ".join(labels))
            choice = pats
        elif n_pats is not None:
            if len(pats_unique) < n_pats:
                msg = ("WARNING: Requested %d patients, but the lazily loaded "
                       "data contains only data from %d patients.")
                logger.warning(msg % (n_pats, len(pats_unique)))
            choice = pats_unique[:n_pats]
        if choice is not None:
            df_vital = df_vital[pats_vital.isin(choice)]
            pats_raw = df_raw["Patient"]
            df_raw = df_raw[pats_raw.isin(choice)]

        store.close()
        if set(cols_01Hz) - set(df_vital.columns):
            # Force re-reading.
            df_vital = None
        if set(cols_50Hz) - set(df_raw.columns):
            # Force re-reading.
            df_raw = None
        logger.info("Done!")
        return df_vital, df_raw

    def _save_data(out_dir, df_vital, df_raw, split_exercises=True):
        logger.info("Writing data...")
        if split_exercises:
            # This drops all data for which no De Morton label is set.
            for label, dfv in df_vital.groupby("DeMortonLabel"):
                filepath = out_dir / "store" / f"ex-{label}.h5"
                write_hdf(df=dfv, path=filepath, key="vital")
            for label, dfr in df_raw.groupby("DeMortonLabel"):
                filepath = out_dir / "store" / f"ex-{label}.h5"
                write_hdf(df=dfr, path=filepath, key="raw")
        # This stores all data!
        filepath = out_dir / "store" / "demorton.h5"
        write_hdf(df=df_vital, path=filepath, key="vital")
        write_hdf(df=df_raw, path=filepath, key="raw")
        logger.info("Done!")


    #######################################################

    if columns is None:
        columns = list(DEFAULT_COLUMNS)
    assert side in ("left", "right", "both")
    cols_01Hz, cols_50Hz = _split_by_sampling_rate(columns)

    df_vital = df_raw = None
    if not forced:
        lazy_mode = "from_exercises" if labels else "all"
        df_vital, df_raw = _read_data_lazily(out_dir=out_dir,
                                             cols_01Hz=cols_01Hz,
                                             cols_50Hz=cols_50Hz,
                                             n_pats=n_pats,
                                             pats=pats,
                                             labels=labels,
                                             mode=lazy_mode)
    if df_vital is None or df_raw is None:
        df_vital, df_raw = _read_data_stores(data_dir=data_dir/"store",
                                             cols_01Hz=cols_01Hz,
                                             cols_50Hz=cols_50Hz,
                                             resample=resample,
                                             side=side,
                                             n_pats=n_pats,
                                             pats=pats,
                                             labels=labels)
        # Save for lazy loading.
        _save_data(out_dir=out_dir,
                   df_vital=df_vital,
                   df_raw=df_raw)

    df_ex = pd.read_csv(data_dir/"exercises.csv")
    df_ex["Patient"] = df_ex["Patient"].map("{:03d}".format)
    df_ex["StartDate"] = pd.to_datetime(df_ex["StartDate"], utc=True)
    df_ex["EndDate"] = pd.to_datetime(df_ex["EndDate"], utc=True)
    df_ex["Duration"] = pd.to_timedelta(df_ex["Duration"])
    df_ex = df_ex.set_index(["Patient", "Day", "Task"])

    return df_vital, df_raw, df_ex


###############################################################################

def plot_data_availability(df, df_ex, column, title, labels,
                           show_ex=True, out_dir=None):
    def plot_exercises(ax, df_ex, x, offset, width, **kwargs):
        colors = sns.color_palette("hls", 20)
        order = ["1", "2a", "2b", "2c", "2d", "3a", "3b", "4", "5a", "5b",
                 "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
        patches = []

        for o, c in zip(order, colors):
            if o not in df_ex.index:
                continue
            start = df_ex.loc[o, "Start"]
            stop = df_ex.loc[o, "Stop"]
            rect = plt.Rectangle(xy=(start, x+offset-width/2),
                                 width=stop-start,
                                 height=width, color=c,
                                 **kwargs)
            ax.add_patch(rect)
            patches.append(rect)
        return patches

    def plot_total_patch(ax, t0, t1, x, offset, width, **kwargs):
        rect = plt.Rectangle(xy=(t0, x+offset-width/2),
                             width=t1-t0,
                             height=width,
                             label="Total",
                             **kwargs)
        ax.add_patch(rect)
        return rect

    def plot_contiguous(ax, indices, t, x, offset, width, **kwargs):
        patches = []
        for i, j in indices:
            t0, t1 = t[i], t[j-1]
            if t0 > t1:
                logger.warning(f"Found non-monotonic step: "
                               "t0={t0:.1f}s, t1={t1:.1f}s")
                continue
            rect = plt.Rectangle(xy=(t0, x+offset-width/2),
                                 width=t1-t0,
                                 height=width,
                                 **kwargs)
            ax.add_patch(rect)
            patches.append(rect)
        return patches

    def plot_bracket(ax, x, offset, offsets, width, height, **kwargs):
        n_days = 3
        rect = plt.Rectangle(xy=(-height-height/4, x-offset-width/2),
                             width=height,
                             height=n_days*width+2*(offset-width),
                             **kwargs)
        ax.add_patch(rect)
        for day in range(1, n_days+1):
            plt.annotate(xy=(-2*height,
                             x+offsets[day]),
                         text="%d" % day,
                         horizontalalignment="right",
                         verticalalignment="center_baseline",
                         fontproperties=dict(size=6))

    # Only consider valid exercises from 1-15.
    tasks = df_ex.index.get_level_values("Task")
    df = df[~df["DeMortonLabel"].isin(["temp", "default"])].copy()
    df_ex = df_ex[~tasks.isin(["temp", "default"])].copy()

    grouping = df.groupby(["Patient", "Side"])
    offset = 0.2
    offsets = {1:offset, 2:0, 3:-offset}    # map: day->offset
    width = 0.15
    tol = 1
    yticks = {}
    fig, ax = plt.subplots(figsize=(6.4, 4.8/8*len(grouping)))
    legend_items = {}
    for i, ((pat_id, side), dfg) in enumerate((grouping)):
        x = len(grouping)-i
        yticks[x] = f"{pat_id}/{side[0].upper()}"
        days = dfg["DeMortonDay"].dropna().unique()
        days = sorted(map(int, days))
        for day in days:
            ex = df_ex.loc[(pat_id, day)].copy()
            # Reference time: StartDate of exercise 1.
            assert ex.loc["1", "StartDate"] == ex["StartDate"].min()
            session_start = ex["StartDate"].min()
            session_end = ex["EndDate"].max()
            delta_total = (session_end - session_start).total_seconds()
            # Time data in seconds since session start.
            ex["Start"] = (ex["StartDate"] - session_start).dt.total_seconds()
            ex["Stop"] = (ex["EndDate"] - session_start).dt.total_seconds()
            # Extract sensor data.
            mask = (dfg.index >= session_start) & (dfg.index <= session_end)
            df_day = dfg[mask]
            ts = df_day.index
            ts = (ts - session_start).total_seconds()
            ts_nona = df_day[column].dropna().index
            ts_nona = (ts_nona - session_start).total_seconds()
            indices = split_contiguous(arr=ts_nona, tol=tol, indices=True)
            h_total = plot_total_patch(ax=ax, t0=0, t1=delta_total, x=x,
                                       width=width, offset=offsets[day],
                                       color=[0.8]*3, edgecolor=None)
            if show_ex:
                # linewidth == 0 creates a small border (only in pdf).
                h_exs = plot_exercises(ax=ax, df_ex=ex, x=x, width=width,
                                       offset=offsets[day], alpha=0.7,
                                       edgecolor=None, linewidth=0.5)
                h_cts = plot_contiguous(ax=ax, indices=indices, t=ts_nona, x=x,
                                        offset=offsets[day]-width/4,
                                        width=width/2, linewidth=0.5,
                                        facecolor=(0.4,0.4,0.4,0.7),
                                        edgecolor="black")
            else:
                h_exs = None
                h_cts = plot_contiguous(ax=ax, indices=indices, t=ts_nona, x=x,
                                        offset=offsets[day], width=width,
                                        edgecolor=None, linewidth=0,
                                        color="seagreen", alpha=0.7)

            # Collect legend entries:
            if h_total:
                legend_items.setdefault("Complete session", h_total)
            if h_exs:
                legend_items.setdefault("Exercises", tuple(h_exs))
            if h_cts:
                colors = sns.color_palette("hls", 20)
                legend_items.setdefault("Data available", h_cts[0])

        plot_bracket(ax=ax, x=x, offset=offset, offsets=offsets, width=width,
                     height=5, facecolor=[0.2]*3, edgecolor="black",
                     linewidth=1)
    ax.set_yticks(list(yticks.keys()))
    ax.set_yticklabels(list(yticks.values()))
    plt.autoscale(enable=True)
    ax.tick_params(top=False, bottom=True, left=False, right=False,
                   labelleft=True, labelbottom=True)
    ax.tick_params(axis="y", pad=-10)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("Time [s]")
    title = "Data availability: %s" % title
    if labels:
        title += ("\n*** Sensor data only for labels: %s ***"
                  % ", ".join(labels))
    ax.set_title(title)
    plt.tight_layout()

    plt.legend(legend_items.values(), legend_items.keys(),
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
               bbox_to_anchor=(1.04,1), loc="upper left")
    if out_dir:
        save_figure(out_dir/"data_availability.pdf", fig=fig,
                    override=False)

###############################################################################

def visualize_per_exercise(df, column, exercises=None):
    if exercises:
        mask = df["DeMortonLabel"].isin(exercises)
    else:
        mask = ~df["DeMortonLabel"].isna()
        mask &= ~df["DeMortonLabel"].isin(["temp", "default"])
    df = df[mask].copy()
    df = df.reset_index()

    def _zero_time(df):
        return (df["timestamp"] - df["timestamp"].min()).dt.total_seconds()

    group_cols = ["Patient", "Side", "DeMortonDay", "DeMortonLabel"]
    t = df.groupby(group_cols).apply(_zero_time)
    t = t.droplevel(level=None, axis=0)
    df = df.set_index(group_cols)
    df["Seconds"] = t
    df = df.reset_index()
    g = sns.relplot(x="Seconds", y=column, hue="Patient",
                    style="DeMortonDay", col="DeMortonLabel",
                    data=df, kind="line", style_order=["1","2","3"],
                    estimator=None)
    plt.tight_layout()
    plt.show()

###############################################################################

def run(args):
    setup_logging(verbosity=2, logger=logger)
    data_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    metrics = args.metrics
    forced = args.force_read
    n_pats = args.n_pats
    pats = args.patients
    labels = args.labels
    dump_context(out_dir=out_dir)

    print_title("Analyzing De Morton exercises:")
    print("    data_dir: %s" % data_dir)
    print("    out_dir: %s" % out_dir)
    print()
    setup_plotting()

    df_vital, df_raw, df_ex = read_data(data_dir=data_dir,
                                        out_dir=out_dir,
                                        columns=metrics,
                                        resample=args.resample,
                                        side=args.side,
                                        forced=forced,
                                        n_pats=n_pats,
                                        pats=pats,
                                        labels=labels)
    plot_data_availability(df=df_raw, df_ex=df_ex, column="A",
                           title="Acceleration (magnitude)",
                           labels=labels, out_dir=out_dir)
    # visualize_per_exercise(df=df_raw,
    #                        column="A",
    #                        exercises=["2a", "2b"])

###############################################################################

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
    parser.add_argument("--resample", type=float, default=None,
                        help="Resampling period, in seconds. Default: 1.")
    parser.add_argument("--metrics", default=None, nargs="+",
                        help="Select a subset of metrics for the analysis")
    parser.add_argument("--side", default="both", type=str,
                        choices=("left", "right", "both"),
                        help="Select the side of the device")
    parser.add_argument("-f", "--force-read", action="store_true",
                        help="Enforce reading of data-stores")
    parser.add_argument("-n", "--n-pats", default=None, type=int,
                        help=("Number of patients to be loaded. Make sure to "
                              "use the --force-read command."))
    parser.add_argument("-p", "--patients", default=None, type=str, nargs="*",
                        help="List of patients to include (format: '%%03d')")
    parser.add_argument("-l", "--labels", default=None, type=str, nargs="*",
                        help="List of De Morton labels to include")
    parser.set_defaults(func=run)
    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
