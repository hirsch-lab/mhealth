import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import timezone, datetime, timedelta

import context
from mhealth.utils.file_helper import ensure_dir
from mhealth.utils.maths import split_contiguous
from mhealth.utils.context_info import dump_context
from mhealth.utils.commons import (print_title, print_subtitle,
                                   create_progress_bar)
from mhealth.utils.plotter_helper import save_figure, setup_plotting

from mhealth.utils.io_manager import IOManager
from mhealth.utils.file_helper import write_csv, write_hdf, read_hdf

################################################################################

VITALS_DESCRIPTION = {
    "HR" : "Heart rate",
    "SPo2": "Oxygen saturation",
    "BloodPressure": "Blood pressure",
    "RespRate": "Respiratory rate",
    "objtemp": "Skin temperature"
}

VITALS_DESCRIPTION_MASIMO = {
    "HR" : "Heart rate",
    "Oxy": "Oxygen saturation",
    "BPsys": "Blood pressure (systolic)",
    "BPdia": "Blood pressure (diastolic)",
    "Resp Rate": "Respiratory rate",
    "Temp": "Skin temperature"
}

VITALS_TO_MASIMO = {
    "HR": "HR",
    "SPo2": "Oxy",
    "BloodPressure": ["BPsys", "BPdia"],
    "RespRate": "Resp Rate",
    "objtemp": "Temp"
}

VITAL_COLS = [
    "HR", "SPo2", "BloodPressure", "RespRate", "objtemp",
]

QUALITY_COLS = [ "HRQ" ]

BIN_PARAMS = {
    "HR":            {"fix_bins": True, "bins": 50, "xlim": (40, 180), "ylim": (0, 0.188), "bw_adjust": 3},
    "SPo2":          {"fix_bins": True, "bins": 50, "xlim": (0, 180),  "ylim": (0,1),      "bw_adjust": 3},
    "BloodPressure": {"fix_bins": True, "bins": 50, "xlim": (0, 255),  "ylim": (0, 0.16),  "bw_adjust": 3},
    "RespRate":      {"fix_bins": True, "bins": 50, "xlim": (0,50),    "ylim": (0, 0.3),   "bw_adjust": 3},
    "objtemp":       {"fix_bins": True, "bins": 50, "xlim": (30,40),   "ylim": (0, 0.225), "bw_adjust": 3}
}

MASIMO_COLORS = [ "#39B069",  "#D9605A" ]


################################################################################
# LOGGING
################################################################################

def log_info(dataset_id, msg):
    log_msg("INFO", dataset_id=dataset_id, msg=msg)
def log_warn(dataset_id, msg):
    log_msg("WARN", dataset_id=dataset_id, msg=msg)
def log_msg(level, dataset_id, msg):
    idstr = "id=%s " % dataset_id
    print("%-4s: %-7s %s" % (level, idstr, msg))


################################################################################
# FILTERING
################################################################################

def fix_time_gaps(df, max_gap_hours, dataset_id):
    def _first_loc_index_where(mask):
        return next((idx for idx, x in zip(mask.index, mask) if x), None)
    def _first_iloc_index_where(mask):
        return next((idx for idx, x in zip(range(len(mask)), mask) if x), None)

    if max_gap_hours <= 0:
        return df

    assert df.index.name == "timestamp"
    diff = df.index.to_series().diff()
    gaps = diff.dt.total_seconds().abs() > (max_gap_hours*3600)
    # idx is None if no gaps are present
    # .loc[] is inclusive for both upper and lower bound.
    # .iloc[] is not inclusive for upper bound (as usual)
    # Consequences:
    #   .loc[idx:] and .loc[:idx] overlap at idx!
    #   .iloc[idx:] and .iloc[:idx] don't overlap
    # Conclusion: better use iloc[]
    idx = _first_iloc_index_where(gaps)
    if gaps.sum() > 1:
        msg = "More than one time leap found (n=%d)"
        log_warn(dataset_id, msg % gaps.sum())
    if idx is not None:
        delta = diff.iloc[idx].total_seconds()/3600
        mask = pd.Series(False, index=df.index)
        if idx < len(df)/2:
            mask[idx+1:] = True
        else:
            mask[:idx] = True
        msg = "Found a gap of %d hours, clipped %d rows (%.3f%%)"
        log_info(dataset_id, msg % (delta, sum(~mask), (~mask).mean()*100))
        df = df.loc[mask].copy()

    # Check if we have a larger gap at the end of the sequence.
    # This gap may be smaller than the max_gap_hours.
    n_last_steps = 1000
    trail_gap_thr = 1*3600
    diff = df.index[-n_last_steps:].to_series().diff()
    diff = diff[~diff.isna()]  # to get rid of initial NaT (not a time)
    diff = diff.dt.total_seconds()
    rank = diff.sort_values(ascending=False)
    key, diff_max = next(rank.items())
    if diff_max > trail_gap_thr:
        n_before = len(df)
        df = df[:key][:-1].copy()
        n_after = len(df)
        if n_before != n_after:
            msg = "Clipped a trailing gap of %.1f hours:"
            log_info(dataset_id, msg % (diff_max/3600))
    return df


def fix_duplicate_timestamps(df, mode, dataset_id):
    assert df.index.name == "timestamp"
    assert mode in ["mean", "last", "first"], "Mode doesn't exist: %s" % mode
    n = len(df)
    if mode == "mean":
        dtypes = df.dtypes
        df = df.groupby("timestamp").mean()
        df = df.astype(dtypes)
    elif mode == "last":
        mask = df.index.duplicated(keep="last")
        df = df[~mask].copy()
    elif mode == "first":
        mask = df.index.duplicated(keep="first")
        df = df[~mask].copy()
    if n!=len(df):
        msg = "Removed %d records with duplicate timestamp (mode=%s)"
        log_info(dataset_id, msg % (n-len(df), mode))
    return df


def fix_non_monotonic_index(df, mode, max_gap_hours, dataset_id):
    """
    Occasionally, the readouts of the Everion device jumps in time: The
    clock of the devices micro-controller might run a bit faster than
    the actual time. The leaps in time occur when the device occasionally
    synchronizes its time with the global system time. These time jumps are
    usually relatively small.
    In some datasets (see for example case 014), larger jumps occur, for
    unclear reasons.

    mode:   mean - take the mean of the duplicate entries (slow)
            last - take the last occurrence the duplicate
    """
    df = fix_duplicate_timestamps(df=df, mode=mode, dataset_id=dataset_id)
    if not df.index.is_monotonic_increasing:
        msg = "The index is not monotonically increasing. Sorting the index."
        log_warn(dataset_id, msg)
        df = df.sort_index()
    df = fix_time_gaps(df=df, max_gap_hours=max_gap_hours,
                       dataset_id=dataset_id)
    assert df.index.is_monotonic_increasing
    return df


def filter_quality(df, dataset_id,
                   quality, quality_cols,
                   non_zero_cols=None,
                   enabled=True):
    if not enabled:
        return df.copy()

    n_old = len(df)
    mask = (df[quality_cols] > quality).all(axis=1)
    mask &= (df[non_zero_cols] != 0).all(axis=1)
    df = df[mask]

    if n_old!=len(df) and dataset_id is not None:
        msg = ("Filtered %d (%0.2f%%) samples by quality (thr=%d)."
               % (n_old-len(df), (n_old-len(df))/n_old*100, quality))
        log_warn(dataset_id, msg)
    return df.copy()


def get_quality_cols(parameter):
    if parameter.lower() in "spo2":
        return ["SPO2Q"]   # or ["SPO2Q", "HR"]
    elif parameter in ("HR", "objtemp"):
        return ["HRQ"]
    else:
        return ["HRQ"]   # ["SPO2Q", "HR"]


################################################################################
# DATA IO
################################################################################

def get_io_manager(out_dir):
    # The filenames look as follows:
    #   - 001_raw_ukbb.csv

    # Extract information from the filenames.
    info_patterns = {
        "pat_id": "([0-9]{3})_raw_ukbb",
    }
    # Write methods.
    target_writers = {
        ".h5": write_hdf
    }
    # Read methods (for lazy loading)
    target_readers = {
        ".h5": read_hdf
    }
    #target_writers[".csv"] = write_csv

    # Construct filenames out of parts.
    target_names = {
        ".csv": "sensor/{pat_id}.csv",
        ".h5":  "store/{pat_id}.h5/data"
    }
    # Works only for target .csv. (Checks inside a .h5 file not possible yet)
    skip_existing = False
    iom = IOManager(out_dir=out_dir,
                    info_patterns=info_patterns,
                    target_writers=target_writers,
                    target_readers=target_readers,
                    target_names=target_names,
                    skip_existing=skip_existing,
                    dry_run=False)
    return iom


def read_patient_data(path):
    df = pd.read_csv(path,
                     parse_dates=["Studyentry"],
                     dayfirst=True,
                     dtype={"Record_id": str})
    df = df.rename({"Record_id": "pat_id"}, axis=1)
    df["pat_id"] = df["pat_id"].astype(str).str.pad(width=3, fillchar="0")
    df = df.set_index("pat_id")
    return df


def read_masimo(path, sep=";"):
    df = pd.read_csv(path,
                     sep=sep,
                     dtype={"Record_id": str})
    df["DateTime"] = pd.to_datetime(df["Date"]+" "+df["Time"],
                                    dayfirst=True, utc=True)
    return df


def read_data(path, iom,
              n_files=None, sep=";",
              lazy_load=True,
              max_gap_hours=24,
              fix_non_monotonic="last"):
    data = {}
    files = list(sorted(path.glob("*.csv")))
    n_files = len(files) if n_files is None else min(n_files, len(files))
    label = "Loading data lazily..." if lazy_load else "Loading data..."
    with create_progress_bar(size=n_files,
                             label=label) as progress:
        for i, filepath in enumerate(files):
            if i > 5:
                pass #break

            progress.update(i)
            iom.set_current(filepath)
            pat_id = iom.info.get("pat_id")
            if iom.skip_existing():
                continue

            if lazy_load and iom.check_out_file(target=".h5"):
                # Lazy loading...
                df = iom.read_target(target=".h5")
            else:
                # Reading...
                df = pd.read_csv(filepath, sep=sep,
                                 parse_dates=["timestamp"],
                                 index_col="timestamp")
                # Processing...
                df = fix_non_monotonic_index(df=df,
                                             mode=fix_non_monotonic,
                                             max_gap_hours=max_gap_hours,
                                             dataset_id=pat_id)
                iom.write_data(data=df)

            data[pat_id] = df
    return data


################################################################################
# PLOTTING
################################################################################

def plot_data_availability(data, masimo, patients, column, out_dir):
    """
    Data before quality filtering!
    """

    def plot_total_patch(ax, t0, t1, x, offset, width, **kwargs):
        rect = plt.Rectangle(xy=(t0, x+offset-width/2),
                             width=t1-t0,
                             height=width,
                             label="Total",
                             **kwargs)
        ax.add_patch(rect)
        return rect

    def plot_contiguous(ax, pat_id, indices, t, x, offset, width, **kwargs):
        patches = []
        for i, j in indices:
            t0, t1 = t[i], t[j-1]
            if t0 > t1:
                msg = f"Found non-monotonic step: t0={t0:.1f}s, t1={t1:.1f}s"
                log_warn(pat_id, msg)
                continue
            rect = plt.Rectangle(xy=(t0, x+offset-width/2),
                                 width=t1-t0,
                                 height=width,
                                 **kwargs)
            ax.add_patch(rect)
            patches.append(rect)
        return patches


    def plot_lines(ax, t, x, offset, width, **kwargs):
        from matplotlib import collections  as mc
        lines = [[(tt, x+offset-width/2), (tt, x+width+offset-width/2)] for tt in t]
        lc = mc.LineCollection(lines, **kwargs)
        ax.add_collection(lc)
        return lc


    def plot_day_night(ax, tlim, height, **kwargs):
        margin = 0.5
        t_morning = 6
        t_evening = 21
        t_mornings = np.arange(tlim[0]//24*24 + t_morning, tlim[1]+24, 24)
        t_evenings = np.arange((tlim[0]//24-1)*24 + t_evening, tlim[1], 24)

        zorder = -1
        col_night = [0.2, 0.2, 0.5, 0.1]
        col_day = [1., 1., 1., 1.,]

        for tm, te in zip(t_mornings, t_evenings):
            # Night
            rect = plt.Rectangle(xy=(te,-margin),
                                 width=tm-te,
                                 height=height+2*margin,
                                 facecolor=col_night,
                                 edgecolor="none",
                                 zorder=zorder,
                                 **kwargs)
            ax.add_patch(rect)
            # Day
            rect = plt.Rectangle(xy=(tm, 0),
                                 width=te-tm+24,
                                 height=height,
                                 facecolor=col_day,
                                 edgecolor="none",
                                 zorder=zorder,
                                 **kwargs)
            ax.add_patch(rect)

    def set_ticks(ax, patients, data, xlim):
        if False:
            # Hourly ticks
            xticks = np.arange(0, xlim[1], 6)
            xhours = xticks % 24
            #xdays = xticks / 24
            ax.set_xticks(xticks)
            ax.set_xticklabels(("%d:00" % h for h in xhours),
                               rotation=40, ha="right")
        else:
            # Daily ticks
            from matplotlib.ticker import (MultipleLocator, FixedLocator,
                                           NullFormatter, FixedFormatter)
            xnoons = np.arange(12, xlim[1], 24)
            xdays = xnoons.astype(int) // 24 + 1
            ax.xaxis.set_major_locator(MultipleLocator(24))
            ax.xaxis.set_minor_locator(FixedLocator(xnoons))
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.xaxis.set_minor_formatter(FixedFormatter(xdays))
            ax.tick_params(which="minor", length=0)
            ax.set_xlabel("Study days")
            #ax.set_xticklabels(("%d" % h for h in xdays))
            #ax.tick_params(axis="x", which="minor", bottom=True)

        ax.set_yticks(np.arange(len(data),0, -1))
        if False:
            # Only id
            ax.set_yticklabels(data.keys())
        else:
            # Id and patient data
            import matplotlib
            age = patients.loc[data.keys(), "Age"]  # age in months
            age = (age/12).round().astype(int)
            sex = patients.loc[data.keys(), "Sex"].str[:1]
            args = zip(data.keys(), age, sex)
            ax.set_yticklabels(("%s (%dy, %s)" % arg for arg in args),
                               ha="left")
            ax.yaxis.set_tick_params(pad=70)

    def plot_masimo(ax, masimo, pat_id, column, start_zero, x, width):
        masimo = masimo[masimo["Record_id"]==pat_id].copy()
        time = (masimo["DateTime"]-start_zero).dt.total_seconds() / 3600
        plot_total_patch(ax=ax, t0=time.min(), t1=time.max(), x=x,
                         width=width/2, offset=-width/2,
                         facecolor="white", edgecolor=[0.6]*3, alpha=0.7)
        plot_lines(ax=ax, t=time, x=x, offset=-width/2,
                   width=width/2, color="k", lw=0.75)
        #ax.plot(time, np.ones_like(time)*x, "k|")
        column = VITALS_TO_MASIMO[column]
        if isinstance(column, list):
            column = column[0]
        mask = ~masimo[column].isna()
        time = time[mask]


        # ax.plot(time, np.ones_like(time)*x, marker="|", color="#3498DB",
        #         linestyle="None")


    width = 0.5
    tol = 1/12   # tol in hours
    fig, ax = plt.subplots(figsize=(6.4, 4.8/14*len(data)))
    for i, (pat_id, df) in enumerate(data.items()):
        x = len(data)-i

        # Zero time.
        start_time = df.index.min()
        start_zero = pd.Timestamp(start_time.date(), tz="UTC")  # Start at 0:00
        time = (df.index - start_zero).total_seconds() / 3600
        # Total time.
        h_total = plot_total_patch(ax=ax, t0=time.min(), t1=time.max(), x=x,
                                   width=width, offset=0,
                                   facecolor=[0.8]*3, edgecolor=[0.2]*3)

        ts_nona = df[column].dropna().index
        ts_nona = (ts_nona - start_zero).total_seconds() / 3600
        indices = split_contiguous(arr=ts_nona, tol=tol, indices=True)
        h_cts = plot_contiguous(ax=ax, pat_id=pat_id,
                                indices=indices, t=ts_nona, x=x,
                                offset=0, width=width,
                                edgecolor=None, linewidth=0,
                                facecolor="#CD6155", alpha=0.7)

        # Filter data. Quality metrics differ for different vital parameters.
        # Always use heart rate > 0 as condition to see if the wearable is worn.
        df_filtered = filter_quality(df.copy(),
                                     quality=50,
                                     quality_cols=get_quality_cols(column),
                                     non_zero_cols=[column, "HR"],
                                     dataset_id=None)
        ts_nona = df_filtered[column].dropna().index
        ts_nona = (ts_nona - start_zero).total_seconds() / 3600
        indices = split_contiguous(arr=ts_nona, tol=tol, indices=True)
        h_cts = plot_contiguous(ax=ax, pat_id=pat_id,
                                indices=indices, t=ts_nona, x=x,
                                offset=0, width=width,
                                edgecolor=None, linewidth=0,
                                color="#609E7C")

        # Plot Masimo data
        if masimo is not None:
            plot_masimo(ax=ax, masimo=masimo, pat_id=pat_id,
                        column=column, x=x, start_zero=start_zero, width=width)

    plt.autoscale(enable=True)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    set_ticks(ax=ax, patients=patients, data=data, xlim=xlim)
    plot_day_night(ax, tlim=xlim, height=ylim[1]-ylim[0])
    #ax.grid(axis="x")
    ax.set_axisbelow(True)
    plt.tight_layout()
    filename = ("availability-%s.pdf" % column).lower()
    save_figure(out_dir / filename, fig=fig, override=True)


################################################################################
def plot_data_availability_all(data, patients, masimo, out_dir):
    """
    Best results expected for unfiltered data.
    """
    out_dir_basic = out_dir/"availability/basic"
    plot_data_availability(data=data, masimo=None, patients=patients,
                           column="HR", out_dir=out_dir_basic)
    plot_data_availability(data=data, masimo=None, patients=patients,
                           column="SPo2", out_dir=out_dir_basic)
    plot_data_availability(data=data, masimo=None, patients=patients,
                           column="RespRate", out_dir=out_dir_basic)
    plot_data_availability(data=data, masimo=None, patients=patients,
                           column="BloodPressure", out_dir=out_dir_basic)
    plot_data_availability(data=data, masimo=None, patients=patients,
                           column="objtemp", out_dir=out_dir_basic)

    out_dir_masimo = out_dir/"availability/masimo"
    plot_data_availability(data=data, masimo=masimo, patients=patients,
                           column="HR", out_dir=out_dir_masimo)
    plot_data_availability(data=data, masimo=masimo, patients=patients,
                           column="SPo2", out_dir=out_dir_masimo)
    plot_data_availability(data=data, masimo=masimo, patients=patients,
                           column="RespRate", out_dir=out_dir_masimo)
    plot_data_availability(data=data, masimo=masimo, patients=patients,
                           column="BloodPressure", out_dir=out_dir_masimo)
    plot_data_availability(data=data, masimo=masimo, patients=patients,
                           column="objtemp", out_dir=out_dir_masimo)


################################################################################
def plot_histograms(data, masimo, out_dir, combine=False):
    def with_margin(rng, margin):
        margin = (rng[1]-rng[0])*margin
        return rng[0]-margin, rng[1]+margin

    fig, ax = plt.subplots()
    label = "Plotting (combined)..." if combine else "Plotting...           "
    with create_progress_bar(size=len(data)*len(VITAL_COLS),
                             label=label) as progress:
        for j, col in enumerate(VITAL_COLS):
            out_dir_col = out_dir / col
            ensure_dir(out_dir_col)
            col_description = VITALS_DESCRIPTION[col]
            for i, (pid, df) in enumerate(data.items()):
                progress.update(j*len(data)+i)
                bw_adjust = BIN_PARAMS[col]["bw_adjust"]
                fix_bins = BIN_PARAMS[col]["fix_bins"]
                xlim = BIN_PARAMS[col]["xlim"]
                ylim = BIN_PARAMS[col]["ylim"]
                bins = BIN_PARAMS[col]["bins"]
                bins = np.linspace(*xlim, bins+1) if fix_bins else bins
                sns.histplot(x=col, data=df, ax=ax,
                             bins=bins,
                             kde_kws=dict(bw_adjust=bw_adjust,
                                          clip=xlim),
                             kde=True,
                             alpha=0.1 if combine else 0.5,
                             stat="probability",
                             cbar_kws=dict(alpha=(0. if combine else 1.)))
                if masimo is not None:
                    masimo_col = VITALS_TO_MASIMO[col]
                    masimo_col = ([masimo_col] if (isinstance(masimo_col, str))
                                  else masimo_col)
                    color_palette = MASIMO_COLORS
                    for k, mcol in enumerate(masimo_col):
                        pat_mask = masimo["Record_id"]==pid
                        vals = masimo.loc[pat_mask,mcol]
                        if not vals.isna().all():
                            ax.vlines(vals, ymin=ylim[0], ymax=ylim[1],
                                      color=color_palette[k],
                                      alpha=0.4)
                ax.set_xlim(with_margin(xlim, 0.05))
                ax.set_ylim(ylim)
                ax.set_xlabel(col_description)
                if not combine:
                    ax.grid(axis="y")
                    ax.set_title("Patient %s: %s" % (pid, col_description))
                    save_figure(out_dir_col / ("%s-%s.pdf" % (pid, col)).lower(),
                                fig=fig, override=True)
                    ax.cla()
                else:
                    # Remove bars... (hacky)
                    #ax.containers[0].remove()
                    pass
            if combine:
                ax.grid(axis="y")
                ax.set_title("All patients")
                save_figure(out_dir_col / ("%s-combined.pdf" % col).lower(),
                            fig=fig, override=True)
                ax.cla()
    plt.close(fig)


################################################################################
def plot_boxplots(data, masimo, out_dir, prefix="", outliers=False):
    fig, ax = plt.subplots()
    label = "Plotting (boxplots)..."
    with create_progress_bar(size=len(data)*len(VITAL_COLS),
                             label=label) as progress:
        n_pats = len(data)
        x_ticks = range(n_pats)
        x_tick_labels = data.keys()
        for j, col in enumerate(VITAL_COLS):
            col_description = VITALS_DESCRIPTION[col]
            for i, (pid, df) in enumerate(data.items()):
                progress.update(j*len(data)+i)
                box_width = 0.6 if masimo is None else 0.3
                offset = 0 if masimo is None else box_width/2
                plt.boxplot([df[col].values],
                            positions=[x_ticks[i]-offset],
                            patch_artist=True,
                            boxprops=dict(facecolor=(.95,.95,.95,.8)),
                            medianprops=dict(lw=2., color="red"),
                            flierprops=dict(marker="o",
                                            markerfacecolor=(.3,.3,.3,.2),
                                            markeredgecolor=None),
                            showfliers=outliers,
                            widths=box_width)
                if masimo is not None:
                    masimo_col = VITALS_TO_MASIMO[col]
                    masimo_col = ([masimo_col] if (isinstance(masimo_col, str))
                                  else masimo_col)
                    color_palette = MASIMO_COLORS
                    for k, mcol in enumerate(masimo_col):
                        pat_mask = masimo["Record_id"]==pid
                        vals = masimo.loc[pat_mask,mcol]
                        vals = vals[~vals.isna()]
                        color = color_palette[k]
                        rgba = lambda rgb, a: mpl.colors.to_rgba(rgb, a)
                        plt.boxplot(vals,
                                    positions=[x_ticks[i]+offset],
                                    patch_artist=True,
                                    boxprops=dict(facecolor=rgba(color, .5)),
                                    medianprops=dict(lw=2., color="red"),
                                    flierprops=dict(marker="o",
                                                    markerfacecolor=rgba(color, .5),
                                                    markeredgecolor=None),
                                    showfliers=True,
                                    widths=box_width)
            plt.xticks(ticks=x_ticks,
                       labels=x_tick_labels,
                       rotation=45)
            ax.grid(axis="y")
            ax.set_title("Vital parameter: %s" % col_description)
            ax.set_ylabel(col_description)
            name = "%s%s" % (prefix, col,)
            save_figure(out_dir / (name + ".pdf").lower(),
                        fig=fig, override=True)
            ax.cla()
    plt.close(fig)


################################################################################
def visualize_bland_altman(data, masimo, col, out_dir,
                           delta_min=15,
                           distinguish_pats=False):

    def compute_means(sensor, valid, col, delta_min):
        valid = valid[~valid.isna()]

        means = []
        for exam_time in valid.index:
            time_delta = timedelta(minutes=delta_min)
            start_time = exam_time - time_delta
            stop_time = exam_time + time_delta
            mask = (sensor.index >= start_time) & (sensor.index <= stop_time)
            mean = sensor[mask].mean(axis=0)
            means.append(mean)
        return pd.Series(means, index=valid.index, dtype=float)

    def plot_bland_altman(sensor, valid, col, out_dir, distinguish_pats=True):
        # Align the data
        diff = sensor-valid;    diff.name = "Difference"
        avg = (valid+sensor)/2; avg.name = "Mean"
        diff_mean = diff.mean()
        diff_std = diff.std()
        offset_ci = 1.96*diff_std
        offset_miss = diff.abs().max()*1.2
        y_off = lambda x, y: y*np.ones_like(x)

        # Info
        n_sensor = sensor.notna().sum()
        n_valid = valid.notna().sum()
        n_diff = diff.notna().sum()
        info = "Info:\n#sensor: %d\n#masimo: %d\n#diff:   %d\ndelta:   %d min"
        info = info % (n_sensor, n_valid, n_diff, delta_min)

        fig, ax = plt.subplots()
        if distinguish_pats:
            comparison_data = pd.concat([avg, diff], axis=1)
            comparison_data.index.names = ["pat_id", "DateTime"]
            comparison_data = comparison_data.reset_index(level=0)
            #comparison_data["pat_id"] = comparison_data["pat_id"].astype(int)
            palette = sns.color_palette("hls",
                                        comparison_data["pat_id"].nunique())
            sns.scatterplot(x="Mean", y="Difference", hue="pat_id",
                            data=comparison_data, ax=ax, palette=palette,
                            edgecolor="none", alpha=0.7)
            h_valid = ax.get_children()[0]  # Best guess hack
        else:
            h_valid = ax.scatter(avg, diff, c="black", alpha=0.2)
        xlim = ax.get_xlim()
        h_mean, = ax.plot(xlim, y_off(xlim, diff_mean), "b", zorder=100)
        h_cip, = ax.plot(xlim, y_off(xlim, diff_mean+offset_ci), ":r", zorder=100)
        h_cim, = ax.plot(xlim, y_off(xlim, diff_mean-offset_ci), ":r", zorder=100)
        # h_ci, = ax.hlines([diff_mean+offset_ci, diff_mean-offset_ci],
        #                  colors="r", linestyle="dotted",
        #                  xmin=xlim[0], xmax=xlim[1], zorder=100)
        h_dummy, = plt.plot([avg.mean()],[0], color="w", alpha=0)

        ax.grid(True)
        ax.set_axisbelow(True)
        ax.set_title(f"Bland-Altman ({col})")
        ax.set_xlabel("Mean: (Sensor+Masimo)/2")
        ax.set_ylabel("Difference: (Sensor-Masimo)")
        legend = [(h_mean,   "Mean: %.2f" % diff_mean),
                  (h_cip,    "95%% CI: ±%.2f" % (1.96*diff_std)),
                  (h_dummy,  ""),
                  (h_valid,  "Patients")]
        leg = ax.legend(*zip(*legend),
                        title="Difference:",
                        loc="upper left",
                        bbox_to_anchor=(1.05, 1.02))

        # these are matplotlib.patch.Patch properties
        props = dict(alpha=0.5, family="DejaVu Sans Mono", fontsize=9)
        # place a text box in upper left in axes coords
        ax.text(1.09, 0.6, info, transform=ax.transAxes,
                verticalalignment="top", **props)

        plt.tight_layout()
        leg._legend_box.align = "left"
        for lh in leg.legendHandles:
            lh.set_alpha(1)
        file_path = out_dir / ("bland_altman_%s_%dmin.png" % (col, delta_min))
        save_figure(path=file_path, fig=fig, dpi=300)


    sensor_data = {}
    valid_data = {}
    for pat_id, df_masimo in masimo.groupby("Record_id"):
        df_masimo = df_masimo.set_index("DateTime")
        df_everion = data[pat_id]
        sensor = df_everion[col]
        if col == "BloodPressure":
            valid = df_masimo["BPsys"]
        else:
            valid = df_masimo[VITALS_TO_MASIMO[col]]
        sensor = compute_means(sensor=sensor, valid=valid,
                               col=col, delta_min=delta_min)
        sensor_data[pat_id] = sensor
        valid_data[pat_id] = valid

    sensor_data = pd.concat(sensor_data)
    valid_data = pd.concat(valid_data)
    plot_bland_altman(sensor=sensor_data, valid=valid_data,
                      col=col, out_dir=out_dir,
                      distinguish_pats=distinguish_pats)


def visualize_bland_altman_all(data, masimo, out_dir, delta_min=15):
    for vital in VITAL_COLS:
        visualize_bland_altman(data, masimo, out_dir=out_dir/"ba",
                               col=vital, delta_min=delta_min,
                               distinguish_pats=False)
        visualize_bland_altman(data, masimo, out_dir=out_dir/"ba-per-patient",
                               col=vital, delta_min=delta_min,
                               distinguish_pats=True)



################################################################################
# SUMMARY
################################################################################

def summarize_data(data, masimo, patients, quality, out_dir):

    def _nicify(df, with_counts=False):
        df = df.droplevel(0, axis=1)
        if "count" in df and with_counts:
            def format_str(x):
                if pd.isna(x["mean"]):
                    return "n/a"
                elif pd.isna(x["std"]):
                    return f"{x['mean']:.2f} (n={int(x['count'])})"
                else:
                    return f"{x['mean']:.2f} ± {x['std']:.2f} (n={int(x['count'])})"
        else:
            def format_str(x):
                return f"{x['mean']:.2f} ± {x['std']:.2f}"

        return df.apply(format_str, axis=1)

    def _time_delta(df):
        start = df.index.get_level_values("timestamp").min()
        stop = df.index.get_level_values("timestamp").max()
        return stop-start

    def _format_delta(delta):
        h = (delta.dt.total_seconds() / 3600).astype(int)
        d = delta.dt.components["days"]
        ret = pd.concat([d,h], keys=["days", "hours"], axis=1)
        ret = ret.apply(lambda r: f"{r['days']}d ({r['hours']}h)", axis=1)
        return ret

    def _summarize(df, patients, out_dir, name, quality):
        def _filtered_agg(df, quality):
            ret = []
            for col in VITAL_COLS:
                df_filt = filter_quality(df, dataset_id=None,
                                         quality=quality,
                                         quality_cols=get_quality_cols(col),
                                         non_zero_cols=[col, "HR"],
                                         enabled=True)
                ret.append(df_filt[col].describe())
            ret = pd.concat(ret, keys=VITAL_COLS, axis=0)
            return ret


        patients = patients[["Sex", "Age", "Diagnosis", "Height", "Weight"]].copy()
        patients["Age"] = (patients["Age"] / 12).round().astype(int)
        summary = df.groupby("pat_id").apply(_filtered_agg, quality=50)
        summary.columns.names = ["vital", "measure"]
        summary = summary.rename(VITALS_DESCRIPTION, axis=1)

        # Compute duration
        delta = df.groupby("pat_id").apply(_time_delta)
        delta = delta.dt.round("H")
        delta = _format_delta(delta)
        delta.name = ("Time", "Time measured")

        ret = pd.concat([patients], keys=["Patient"], axis=1)
        ret = ret.join(summary, how="right")
        ret = ret.join(delta)
        ret.to_csv(out_dir / (name+".csv"))

        summary2 = summary.groupby("vital", axis=1).apply(_nicify,
                                                          with_counts=True)
        delta.name = "Time measured"
        summary2 = summary2.join(delta)
        summary2["counts"] = summary.loc[:,("Heart rate", "count")].astype(int)
        ret2 = patients.join(summary2, how="right")
        ret2.to_csv(out_dir / (name+"_short"+".csv"))
        return ret2


    def _summarize_massimo(patients, masimo, out_dir, name):
        summary = masimo.rename(VITALS_DESCRIPTION_MASIMO, axis=1)
        summary = summary.rename({"Record_id": "pat_id"}, axis=1)
        summary = summary.groupby("pat_id").describe()
        summary.columns.names = ["vital", "measure"]

        patients = patients[["Sex", "Age", "Diagnosis", "Height", "Weight"]].copy()
        patients["Age"] = (patients["Age"] / 12).round().astype(int)

        ret = pd.concat([patients], keys=["Patient"], axis=1)
        ret = ret.join(summary, how="right")
        #ret = ret.sort_index(level=0, axis=1)
        ret.to_csv(out_dir / (name+".csv"))

        summary2 = summary.groupby("vital", axis=1).apply(_nicify,
                                                          with_counts=True)
        ret2 = patients.join(summary2, how="right")
        ret2.to_csv(out_dir / (name+"_short"+".csv"))
        return ret2


    df = pd.concat(data, axis=0, names=("pat_id", "timestamp"))
    #df = df[VITAL_COLS]
    #df = df.rename(VITALS_DESCRIPTION, axis=1)

    print_title("Summary Masimo:")
    summary = _summarize_massimo(patients=patients, masimo=masimo,
                                 out_dir=out_dir, name="masimo")
    print(summary)

    print_title("Summary Wearables Data:")
    print_subtitle("Overall")
    summary = _summarize(df=df, patients=patients, quality=quality,
                         out_dir=out_dir, name="overall")
    print(summary)

    print_subtitle("Daytime")
    ts = df.index.get_level_values("timestamp")
    df_day = df[(ts.hour>=9) & (ts.hour<=18)]
    summary = _summarize(df=df_day, patients=patients, quality=quality,
                         out_dir=out_dir, name="day")
    print(summary)

    print_subtitle("Nighttime")
    df_night = df[(ts.hour<=6) | (ts.hour>=22)]
    summary = _summarize(df=df_night, patients=patients, quality=quality,
                         out_dir=out_dir, name="night")
    print(summary)

    # log_info()


################################################################################
# MAIN
################################################################################

def run(args):
    n_files = args.n_files
    in_dir = Path(args.in_dir)
    data_dir = in_dir / "cleaned_data" / "data_short_header"
    masimo_file = in_dir / "masimo" / "vitals_pheonix_main.csv"
    pat_file = in_dir / "patients.csv"
    out_dir = Path(args.out_dir)
    lazy_load = not args.forced_read
    quality = args.quality
    max_gap = args.max_gap
    dump_context(out_dir=out_dir)
    setup_plotting()
    iom = get_io_manager(out_dir=out_dir)

    print_title("Analysis of UKBB data:")
    print("    data_dir:", data_dir)
    print("    out_dir:", out_dir)
    print()
    masimo = read_masimo(path=masimo_file)
    patients = read_patient_data(path=pat_file)

    # Read data, no quality filtering!
    data = read_data(data_dir, iom=iom,
                     n_files=n_files,
                     lazy_load=lazy_load,
                     max_gap_hours=max_gap,
                     fix_non_monotonic="last")

    plot_data_availability_all(data=data, patients=patients,
                               masimo=masimo, out_dir=out_dir)

    summarize_data(data=data, masimo=masimo, patients=patients,
                   quality=50, out_dir=out_dir)

    plot_histograms(data=data, masimo=None,
                    out_dir=out_dir/"histograms", combine=True)
    plot_histograms(data=data, masimo=masimo,
                    out_dir=out_dir/"histograms")
    plot_boxplots(data=data, masimo=None, prefix="plain-",
                  out_dir=out_dir/"boxplots")
    plot_boxplots(data=data, masimo=masimo, prefix="masimo-",
                  out_dir=out_dir/"boxplots")
    plot_boxplots(data=data, masimo=None, prefix="outliers-",
                  out_dir=out_dir/"boxplots",
                  outliers=True)

    visualize_bland_altman_all(data=data, masimo=masimo,
                               out_dir=out_dir, delta_min=15)


def parse_args():
    description = ("Analysis of UKBB data.")
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(add_help=False,
                                     formatter_class=formatter,
                                     description=description)
    parser.add_argument("-h", "--help", action="help",
                        help="Show this help text")
    parser.add_argument("-i", "--in-dir", required=True,
                        help="Input directory")
    parser.add_argument("-o", "--out-dir", help="Output directory",
                        default="output/")
    parser.add_argument("-n", "--n-files", default=None, type=int,
                        help="Number of datasets to include")
    parser.add_argument("-f", "--forced-read", action="store_true",
                        help="Disable lazy-loading of data")
    parser.add_argument("-q", "--quality", type=int, default=50,
                        help="Threshold for quality filtering")
    parser.add_argument("--max-gap", type=int, default=24,
                        help="Maximal gap in hours tolerated")
    parser.set_defaults(func=run)
    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
