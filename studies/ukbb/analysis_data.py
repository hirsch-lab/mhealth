import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

import context
from mhealth.utils.file_helper import ensure_dir
from mhealth.utils.context_info import dump_context
from mhealth.utils.commons import print_title, create_progress_bar
from mhealth.utils.plotter_helper import save_figure, setup_plotting

from mhealth.utils.io_manager import IOManager
from mhealth.utils.file_helper import write_csv, write_hdf, read_hdf

VITALS_DESCRIPTION = {
    "HR" : "Heart rate",
    "SPo2": "Oxygen saturation",
    "BloodPressure": "Blood pressure",
    "RespRate": "Respiratory rate",
    "objtemp": "Skin temperature"
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

def log_info(dataset_id, msg):
    log_msg("INFO", dataset_id=dataset_id, msg=msg)
def log_warn(dataset_id, msg):
    log_msg("WARN", dataset_id=dataset_id, msg=msg)
def log_msg(level, dataset_id, msg):
    idstr = "id=%s " % dataset_id
    print("%-4s: %-7s %s" % (level, idstr, msg))

def fix_time_gaps(df, max_gap_hours, dataset_id):
    def _first_loc_index_where(mask):
        return next((idx for idx, x in zip(mask.index, mask) if x), None)
    def _first_iloc_index_where(mask):
        return next((idx for idx, x in zip(range(len(mask)), mask) if x), None)

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


def filter_quality(df, quality, dataset_id):
    n = len(df)
    mask = (df[QUALITY_COLS] > quality).all(axis=1)
    mask &= (df["HR"] > 0)
    df = df[mask]
    if n!=len(df):
        msg = ("Filtered %d (%0.2f%%) samples by quality (thr=%d)."
               % (n-len(df), (n-len(df))/n*100, quality))
        log_warn(dataset_id, msg)
    return df.copy()


def read_masimo(path, sep=";"):
    df = pd.read_csv(path,
                     sep=sep,
                     dtype={"Record_id": str})
    df["DateTime"] = pd.to_datetime(df["Date"]+" "+df["Time"],
                                    dayfirst=True)
    return df


def read_data(path, iom,
              n_files=None, sep=";",
              lazy_load=True,
              max_gap_hours=24,
              fix_non_monotonic="last",
              quality=50):
    data = {}
    files = list(sorted(path.glob("*.csv")))
    n_files = len(files) if n_files is None else min(n_files, len(files))
    label = "Loading data lazily..." if lazy_load else "Loading data..."
    with create_progress_bar(size=n_files,
                             label=label) as progress:
        for i, filepath in enumerate(files):
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
                df = filter_quality(df=df, quality=quality,
                                    dataset_id=pat_id)
                iom.write_data(data=df)

            data[pat_id] = df
    return data


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


def run(args):
    n_files = args.n_files
    data_dir = Path(args.in_dir)
    masimo_file = Path(args.masimo_file)
    out_dir = Path(args.out_dir)
    lazy_load = not args.forced_read
    quality = args.quality
    dump_context(out_dir=out_dir)
    setup_plotting()
    iom = get_io_manager(out_dir=out_dir)

    print_title("Analysis of UKBB data:")
    print("    data_dir:", data_dir)
    print("    out_dir:", out_dir)
    print()
    masimo = read_masimo(path=masimo_file)
    data = read_data(data_dir, iom=iom,
                     n_files=n_files,
                     lazy_load=lazy_load,
                     max_gap_hours=12,
                     fix_non_monotonic="last",
                     quality=quality)
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
    parser.add_argument("-m", "--masimo-file", required=True,
                        help="Path to file with Masimo data")
    parser.add_argument("-o", "--out-dir", help="Output directory",
                        default="output/")
    parser.add_argument("-n", "--n-files", default=None, type=int,
                        help="Number of datasets to include")
    parser.add_argument("-f", "--forced-read", action="store_true",
                        help="Disable lazy-loading of data")
    parser.add_argument("--quality", type=int, default=50,
                        help="Threshold for quality filtering")
    parser.set_defaults(func=run)
    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
