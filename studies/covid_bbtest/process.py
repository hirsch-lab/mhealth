import time
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

import context
from mhealth.utils.plotter_helper import save_figure
from mhealth.utils.commons import create_progress_bar

# Used if command-line option --parameters is not provided.
DEFAULT_PARAMETERS = ["Temperatur", "Herzfrequenz", "Atemfrequenz"]
# Data sources included in HF-AF_25052021.csv.
VALIDATION_DATA_SOURCES = ["WELCHALLYN_MONITOR", "PHILIPS_GATEWAY"]
# Half-ranges relevant for the validation: x +/- delta
DELTAS = {
    "Atemfrequenz": 3,      # ±3bpm
    "Herzfrequenz": 10,     # ±10bpm
    "Temperatur":   0.5     # ±0.5°C
}
# Half-range of the for the timestamp delta, in minutes.
DELTA_TS = 2.5              # ±2.5min
# Devices are identified by the bed number they are used with.
# In case of device breakdown (or other problems), some devices
# were replaced by a device of another room. The below lookup
# specifies which the bed ids (devices) must be renamed, as well
# as the time range, between which the lookup applies.
DEVICE_REPLACEMENT_LOOKUP = {
    # Alias     True       From                        To
    "2653F"  : ("2655F",  "2021-05-14 12:00:00+02:00", None),
    "2652F"  : ("2656FL", "2021-05-18 00:00:00+02:00", None),
    "2661TL" : ("2661FL", "2021-05-20 00:00:00+02:00", None),
    "2664T"  : ("2664F",  "2021-05-12 00:00:00+02:00", None),
    "2665T"  : ("2665F",  None, "2021-05-19 10:30:00+02:00"),
}
# Expected value ranges per vital parameter.
VALUE_RANGES = {
    "Atemfrequenz": [0, 35],
    "Herzfrequenz": [30, 130],
    "Temperatur":   [35, 40],
}
BIN_WIDTHS = {
    "Atemfrequenz": 0.5,
    "Herzfrequenz": 1,
    "Temperatur":   0.01,
}
BIN_WIDTHS_VALID = {
    "Atemfrequenz": 1,
    "Herzfrequenz": 2,
    "Temperatur":   0.1,
}


def tic():
    return time.time()


def toc(label, start):
    diff = time.time()-start
    print(label + (": %.3f" % diff))


def check_dir(path):
    if not path.is_dir():
        msg = "Requested folder does not exist: %s"
        raise FileNotFoundError(msg % path)


def ensure_dir(path, exist_ok=True):
    path = Path(path)
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=exist_ok)
    return path.is_dir()


def apply_replacement_lookup(df):
    print("Applying device replacements...")
    def dt_to_str(dt):
        return "--" if dt is None else dt.strftime("%m.%d.%y %H:%M")
    for id_alias, replace_data in DEVICE_REPLACEMENT_LOOKUP.items():
        id_true, repl_start, repl_stop = replace_data
        repl_start = pd.to_datetime(repl_start)
        repl_stop = pd.to_datetime(repl_stop)
        mask = ((df["Bettenstellplatz"]==id_alias) &
                ((repl_start is None) or df["Timestamp"]>=repl_start) &
                ((repl_stop is None) or df["Timestamp"]<=repl_stop))
        df.loc[mask, "Bettenstellplatz"] = id_true
        print("%-6s => %-6s: %6d affected values in time range (%s, %s)"
              % (id_alias, id_true, mask.sum(),
                 dt_to_str(repl_start), dt_to_str(repl_stop)))
    print()


def read_validation_data(data_dir):
    def no_whitespace(s):
        return s.replace(" ", "")
    def fix_time(s):
        return s.replace(".", ":")

    def form_timestamp(df, col_date, col_time):
        timestamp = df[col_date] + " " + df[col_time]
        timestamp = pd.to_datetime(timestamp, dayfirst=True)
        timestamp = timestamp.dt.tz_localize("Europe/Zurich").copy()
        timestamp[(df[col_date]=="") | (df[col_time]=="")] = None
        return timestamp

    def format_manual(df, timestamp, parameter):
        df_ret = df[["Bettenstellplatz", parameter,
                     "Bemerkungen", "Abweichung_Trageort"]].copy()
        df_ret = df_ret.rename({parameter: "Wert"}, axis=1)
        icol = df_ret.columns.get_loc("Wert")
        df_ret.insert(loc=icol, column="Vitalparameter", value=parameter)
        df_ret.insert(loc=0, column="Timestamp", value=timestamp)
        df_ret = df_ret[~df_ret["Wert"].isna()].copy()
        return df_ret

    def read_station_data(valid_dir):
        file_path = valid_dir/"HF-AF_25052021.csv"
        df = pd.read_csv(file_path,
                         converters={"Signatur": str.strip,
                                     "Bettenstellplatz": str.strip})
        df = df[df["Signatur"].isin(VALIDATION_DATA_SOURCES)]
        timestamp = form_timestamp(df=df, col_date="Datum", col_time="Zeit")
        df.insert(loc=0, column="Timestamp", value=timestamp)
        df = df.drop(["Datum", "Zeit"], axis=1)
        # Transform to long format.
        df = df.melt(id_vars=["Timestamp", "Bettenstellplatz", "Signatur"],
                     value_vars=["Herzfrequenz", "Atemfrequenz", "Temperatur"],
                     var_name="Vitalparameter", value_name="Wert")
        df = df[~df["Wert"].isna()].copy()
        df["Bemerkungen"] = ""
        df["Abweichung_Trageort"] = ""
        return df

    def read_manual_data(valid_dir):
        file_path = valid_dir/"Validierung_Daten_manuell_Mai2021_alle.csv"
        df = pd.read_csv(file_path,
                         converters={"Bettenstellplatz": no_whitespace,
                                     "Zeit_AF": fix_time,
                                     "Zeit_HF": fix_time,
                                     "Bemerkungen": str.strip,
                                     "Abweichung_Trageort": str.strip})
        # Atemfrequenz
        ts = form_timestamp(df=df, col_date="Datum", col_time="Zeit_AF")
        df_a = format_manual(df=df, timestamp=ts, parameter="Atemfrequenz")
        # Herzfrequenz
        ts = form_timestamp(df=df, col_date="Datum", col_time="Zeit_HF")
        df_h = format_manual(df=df, timestamp=ts, parameter="Herzfrequenz")
        # Temperatur (Zeit_Temp, use Zeit_HF is missing!)
        ts = form_timestamp(df=df, col_date="Datum", col_time="Zeit_HF")
        df_t = format_manual(df=df, timestamp=ts, parameter="Temperatur")
        df = pd.concat((df_a, df_h, df_t), axis=0)
        df["Signatur"] = "MANUELL"
        return df

    print("Reading Validation data...")
    valid_dir = data_dir/"original"/"validation"
    check_dir(valid_dir)
    df_station = read_station_data(valid_dir=valid_dir)
    df_manual = read_manual_data(valid_dir=valid_dir)
    df_valid = pd.concat((df_station, df_manual), axis=0)
    df_valid = df_valid.sort_values(["Bettenstellplatz", "Timestamp"])
    return df_valid


def read_baslerband_data(data_dir, n_max=None):
    def read_bb_file(path):
        # Sample path:
        # ../2021-05-25/2617_FL/basler_band_DB_B4_2C_E5_CC_45_activity_file.csv
        bed_id = path.parent.name.replace("_", "")
        if bed_id == "2668":
            bed_id = "2668E"
        device_id = path.stem
        device_id = device_id.replace("basler_band_", "")
        device_id = device_id.replace("_activity_file", "")
        df = pd.read_csv(path, index_col=[0], parse_dates=[0], sep=";")
        df.index.name = "Timestamp"
        # Filter by quality as specified
        df = df[df["wearing"]==4]
        df = df[["resp_filtered", "hrm_filtered",]]
        df = df.rename({"resp_filtered": "Atemfrequenz",
                        "hrm_filtered": "Herzfrequenz"}, axis=1)
        df["Bettenstellplatz"] = bed_id
        df["DeviceID"] = device_id
        df["Signatur"] = "BASLER_BAND"
        df = df.reset_index(drop=False)
        return df

    print("Reading Basler Band data...")
    bb_dir = data_dir/"original"/"basler_band"
    check_dir(bb_dir)
    files = bb_dir.glob("**/basler_band*activity_file.csv")
    files = sorted(files)
    dfs = []
    progress = create_progress_bar(size=len(files),
                                   label="Processing...")
    for i, path in enumerate(files):
        if i>=n_max:
            break
        progress.update(i)
        df = read_bb_file(path=path)
        dfs.append(df)
    progress.finish()
    df = pd.concat(dfs, axis=0)
    df = df.melt(id_vars=["Timestamp", "Bettenstellplatz", "Signatur", "DeviceID"],
                 value_vars=["Herzfrequenz", "Atemfrequenz"],
                 var_name="Vitalparameter", value_name="Wert")
    apply_replacement_lookup(df)
    df = df.sort_values(["Bettenstellplatz", "Timestamp"])
    return df


def read_core_data(data_dir, n_max=None):
    def read_core_file(path, columns):
        # Sample path:
        # ../2021-05-17/2617_FL/core_D6_BE_C5_06_B3_48_storage-cbta_d.csv
        bed_id = path.parent.name.replace("_", "")
        if bed_id == "2668":
            bed_id = "2668E"
        device_id = path.stem
        device_id = device_id.replace("core_", "")
        device_id = device_id.replace("_storage-cbta_d", "")
        df = pd.read_csv(path, index_col=[0], parse_dates=[0], sep=";")
        df.index.name = "Timestamp"
        df = df.rename(columns.to_dict(), axis=1)
        # Filter by quality as specified
        df = df[df["quality (core only)"]==4]
        df = df[["cbt [mC]",]]
        df = df.rename({"cbt [mC]": "Temperatur"}, axis=1)
        df["Temperatur"] /= 1000  # from °mC to °C
        df["Bettenstellplatz"] = bed_id
        df["DeviceID"] = device_id
        df["Signatur"] = "CORE"
        df = df.reset_index(drop=False)
        return df

    print("Reading Core data...")
    core_dir = data_dir/"original"/"core"
    check_dir(core_dir)
    columns = pd.read_csv(core_dir/"0_storage-cbta_d_columns.csv",
                          skipinitialspace=True,
                          index_col=[0], header=None, squeeze=True)
    columns.index = columns.index.astype(str)
    files = core_dir.glob("**/core_*storage-cbta_d.csv")
    files = sorted(files)
    progress = create_progress_bar(size=len(files),
                                   label="Processing...")
    dfs = []
    for i, path in enumerate(files):
        if i>=n_max:
            break
        progress.update(i)
        df = read_core_file(path=path, columns=columns)
        dfs.append(df)
    progress.finish()
    df = pd.concat(dfs, axis=0)
    df = df.melt(id_vars=["Timestamp", "Bettenstellplatz", "Signatur", "DeviceID"],
                 value_vars=["Temperatur"],
                 var_name="Vitalparameter", value_name="Wert")
    apply_replacement_lookup(df)
    df = df.sort_values(["Bettenstellplatz", "Timestamp"])
    return df


def read_data(data_dir, out_dir, force_read, n_files_max):
    df_bb = None
    df_core = None
    df_valid = None
    n_files_max = np.inf if n_files_max is None else n_files_max
    path_store = out_dir/"store.h5"
    if not force_read and path_store.is_file():
        store = pd.HDFStore(path_store, mode="r")
        if "valid" in store:
            print("Reading validation data lazily...")
            df_valid = store["valid"]
        if "bb" in store:
            print("Reading Basler Band data lazily...")
            df_bb = store["bb"]
        if "core" in store:
            print("Reading Core data lazily...")
            df_core = store["core"]
        store.close()
    if df_valid is None:
        df_valid = read_validation_data(data_dir=data_dir)
        df_valid.to_hdf(path_store, key="valid")
    if df_bb is None:
        df_bb = read_baslerband_data(data_dir=data_dir, n_max=n_files_max)
        df_bb.to_hdf(path_store, key="bb")
    if df_core is None:
        df_core = read_core_data(data_dir=data_dir, n_max=n_files_max)
        df_core.to_hdf(path_store, key="core")
    df_sensor = pd.concat([df_bb, df_core], axis=0)
    # Reset index so that it can be used for index operations.
    df_sensor = df_sensor.reset_index(drop=True)
    df_valid  = df_valid.reset_index(drop=True)
    return df_valid, df_sensor


def validate_data(df_sensor, df_valid, parameters, skip_zeros,
                  visualize, out_dir=None, interactive=False):
    if parameters is None:
        parameters = DEFAULT_PARAMETERS.copy()
    iloc = df_valid.columns.get_loc("Wert")
    df_valid.insert(loc=iloc+1, column="Messüberlappung",    value=None)
    df_valid.insert(loc=iloc+2, column="Sensor (mean)",      value=np.nan)
    df_valid.insert(loc=iloc+3, column="Sensor (std)",       value=np.nan)
    df_valid.insert(loc=iloc+4, column="Sensor (median)",    value=np.nan)
    df_valid.insert(loc=iloc+5, column="Sensor (samples)",   value=None)
    df_valid.insert(loc=iloc+6, column="Sensor (zeros)",     value=None)
    df_valid.insert(loc=iloc+7, column="Fehler (in range)",  value=None)

    def handle_key(event):
        if event.key in "eEqQ":
            nonlocal visualize
            visualize = False
    def handle_close(evt):
        nonlocal visualize
        visualize = False
    def plot_signal_hist(ax, data, parameter, bed_id, tsv):
        tsv_str = tsv.strftime("%H:%M (%d.%m.%y)")
        ax.clear()
        bw = BIN_WIDTHS[parameter]
        bin_left = np.round_(data.min(), 2)
        bin_right = data.max()+bw
        bins = np.arange(bin_left, bin_right, bw)
        if len(bins)<=2:
            bins = [bins[0]-bw] + list(bins) + [bins[-1]+bw]
        sns.histplot(x=data, kde=True, alpha=0.4, ax=ax, bins=bins)
        ax.set_title("%s: n=%d, t=%s" % (bed_id, len(data), tsv_str))
        ax.set_xlabel(parameter)
        ax.set_ylabel("Counts")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        #ax.set_xlim(VALUE_RANGES[parameter])
        ax.set_xticks(bins)
    def plot_signals(ax, data, ts, parameter, bed_id, tsv):
        tsv_str = tsv.strftime("%H:%M (%d.%m.%y)")
        ax.clear()
        sns.lineplot(x=ts, y=data, ax=ax, color=[0.6]*3)
        ax.plot([tsv, tsv], [data.min(), data.max()], color="red")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.set_title("%s: n=%d, t=%s" % (bed_id, len(data), tsv_str))
        ax.set_xlabel(parameter)
        ax.set_ylabel("Counts")

    fig1 = fig2 = None
    if visualize:
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig1.canvas.mpl_connect("key_release_event", handle_key)
        fig2.canvas.mpl_connect("key_release_event", handle_key)
        fig1.canvas.mpl_connect("close_event", handle_close)
        fig2.canvas.mpl_connect("close_event", handle_close)
        if interactive:
            fig1.show()
            fig2.show()

    for parameter in parameters:
        print("Aggregating data for parameter '%s'..." % parameter)
        dfs_all = df_sensor[df_sensor["Vitalparameter"]==parameter]
        dfv_all = df_valid[df_valid["Vitalparameter"]==parameter]
        gs = dfs_all.groupby("Bettenstellplatz")
        gv = dfv_all.groupby("Bettenstellplatz")
        bed_ids = set(gs.groups) & set(gv.groups)
        for bed_id in bed_ids:
            dfs = gs.get_group(bed_id)
            dfv = gv.get_group(bed_id)
            assert(dfs["Timestamp"].is_monotonic_increasing)
            assert(dfv["Timestamp"].is_monotonic_increasing)
            assert(not dfs["Timestamp"].isna().any())
            assert(not dfv["Timestamp"].isna().any())
            ts_start = dfv["Timestamp"]-pd.Timedelta(minutes=DELTA_TS)
            ts_stop  = dfv["Timestamp"]+pd.Timedelta(minutes=DELTA_TS)
            i_start  = dfs["Timestamp"].searchsorted(ts_start, side="left")
            i_stop   = dfs["Timestamp"].searchsorted(ts_stop, side="right")
            assert(len(i_start)==len(i_stop)==len(dfv))
            for j, (i0, i1, tsv) in enumerate(zip(i_start, i_stop, dfv["Timestamp"])):
                data = dfs.iloc[i0:i1]["Wert"]
                ts = dfs.iloc[i0:i1]["Timestamp"]
                if skip_zeros:
                    data = data[data!=0]
                valid = dfv.iloc[j]["Wert"]
                mean = data.mean()
                std = data.std()
                median = data.median()
                samples = len(data)
                zeros = (data==0).sum()
                is_in_range = abs(valid-mean) <= DELTAS[parameter]
                is_in_range = None if pd.isna(mean) else is_in_range
                assert(dfv.loc[dfv.index[j], "Wert"]==dfv.iloc[j]["Wert"])
                # Check if the conventional measurement overlaps with the
                # available sensor data. i0==i1==0 or i0==i1==len(dfs)
                # is true if the conventional measurement j is taken outside
                # the range [dfs.Timestamp.min(), dfs.Timestamp.max()].
                overlapping = not (i0==i1==0 or i0==i1==len(dfs))
                df_valid.loc[dfv.index[j], "Messüberlappung"] = overlapping
                df_valid.loc[dfv.index[j], "Sensor (mean)"] = mean
                df_valid.loc[dfv.index[j], "Sensor (std)"] = std
                df_valid.loc[dfv.index[j], "Sensor (median)"] = median
                df_valid.loc[dfv.index[j], "Sensor (samples)"] = samples
                df_valid.loc[dfv.index[j], "Sensor (zeros)"] = zeros
                df_valid.loc[dfv.index[j], "Fehler (in range)"] = is_in_range

                if visualize and len(data)>5:
                    plot_signal_hist(ax1, data, parameter, bed_id, tsv)
                    plot_signals(ax2, data, ts, parameter, bed_id, tsv)
                    if interactive:
                        fig1.canvas.draw()
                        fig2.canvas.draw()
                        while not plt.waitforbuttonpress() and visualize:
                            pass
                        fig1.canvas.flush_events()
                        fig2.canvas.flush_events()
                    if out_dir:
                        file_path = out_dir / "signals"
                        file_path /= "%s_%s_%02d.png" % (bed_id, parameter, j)
                        save_figure(path=file_path, fig=fig2)
                        file_path = out_dir / "hists"
                        file_path /= "%s_%s_%02d.png" % (bed_id, parameter, j)
                        save_figure(path=file_path, fig=fig1)
    if fig1:
        plt.close("all")
    return df_valid


def visualize_validation(df_valid, out_dir, skip_zeros):
    def visualize_all_validation_measurements(dfv, parameter, out_dir):
        validiert = ~dfv["Sensor (mean)"].isna()
        bw = BIN_WIDTHS_VALID[parameter]
        bin_left = np.round_(dfv["Wert"].min(), 2)
        bin_right = dfv["Wert"].max()+bw
        bins = np.arange(bin_left, bin_right, bw)
        fig, ax = plt.subplots()
        sns.histplot(x=dfv["Wert"], bins=bins, ax=ax, alpha=0.4,
                     label="Alle Messungen (n=%d)" % len(dfv), kde=True)
        sns.histplot(x=dfv.loc[validiert, "Wert"], color="black",
                     bins=bins, ax=ax, alpha=0.3, kde=True,
                     label="Validierte Messungen (n=%d)" % validiert.sum())
        ax.legend()
        ax.set_title("%s: Validierungsdaten (manuell)" % parameter)
        ax.set_xlabel(parameter)
        ax.set_ylabel("Counts")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        file_path = out_dir / ("measurements_%s.png" % parameter)
        save_figure(path=file_path, fig=fig, dpi=300)
        plt.close(fig)

    def visualize_errors(dfv, parameter, out_dir):
        fig, ax = plt.subplots()
        diff = dfv["Wert"] - dfv["Sensor (mean)"]
        # Drop nan values (measurements without sensor data).
        diff = diff[(~diff.isna())]
        # Crude outlier rejection.
        diff_crit = np.abs(np.asarray(VALUE_RANGES[parameter])).max()*10
        diff_mask = diff.abs()>diff_crit
        if diff_mask.any():
            df_tmp = dfv.loc[diff_mask[diff_mask].index]
            df_tmp = df_tmp.drop(["Bemerkungen", "Abweichung_Trageort",
                                  "Sensor (samples)", "Sensor (zeros)",
                                  "Sensor (std)", "Sensor (median)",
                                  "Messüberlappung",
                                  "Fehler (in range)"],
                                 axis=1)
            print()
            print("Warning: identified and removed outliers:")
            print(df_tmp)
            print()
        diff = diff[~diff_mask]
        sns.histplot(x=diff, ax=ax, alpha=0.4, kde=True)
        xlim = np.abs(ax.get_xlim()).max()
        xlim = [-xlim, xlim]
        ax.set_xlim(xlim)
        ax.set_xlabel("Error (valid-sensor.mean())")
        ax.set_ylabel("Counts")
        ax.set_title("Messfehler %s" % parameter)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        # Add patches indicating errors exceeding the limits.
        ylim = ax.get_ylim()
        # rect = patches.Rectangle((-DELTAS[parameter], ylim[0]),
        #                          width=2*DELTAS[parameter],
        #                          height=ylim[1]-ylim[0],
        #                          edgecolor=None, facecolor="green", alpha=0.1)
        # ax.add_patch(rect)
        ax.plot([-DELTAS[parameter], -DELTAS[parameter]], ylim, linestyle=":",
                color="red", alpha=0.7)
        ax.plot([DELTAS[parameter], DELTAS[parameter]], ylim, linestyle=":",
                color="red", alpha=0.7)
        ax.set_ylim(ylim)
        file_path = out_dir / ("errors_%s.png" % parameter)
        save_figure(path=file_path, fig=fig, dpi=300)


    sub_dir_name = "without_zeros" if skip_zeros else "with_zeros"
    out_dir = out_dir / "validation" / sub_dir_name
    for parameter, dfv in df_valid.groupby("Vitalparameter"):
        visualize_all_validation_measurements(dfv=dfv,
                                              parameter=parameter,
                                              out_dir=out_dir)
        visualize_errors(dfv=dfv, parameter=parameter, out_dir=out_dir)

        # Performance metrics of validation:
        # 1) availability: number of measurements for which validation was
        #    possible divided by the total number of measurements
        # 2) inlier_rate: number values within boundaries divided by the
        #    number of measurements for which validation was possible
        available = ~dfv["Sensor (mean)"].isna()
        diff = dfv["Wert"] - dfv["Sensor (mean)"]
        availability = available.sum()/len(dfv)
        inlier_rate = (diff.abs()<=DELTAS[parameter]).sum()/available.sum()
        print()
        print("Summary for vital parameter: %s" % parameter)
        print("    availability: %.3f" % availability)
        print("    inlier_rate:  %.3f" % inlier_rate)


def run(args):
    data_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    force_read = args.force_read
    n_files_max = args.n_files_max
    parameters = args.parameters
    skip_zeros = args.skip_zeros
    detailed = args.detailed
    ensure_dir(out_dir)
    file_name = ("validation_zeros_filtered.csv" if skip_zeros else
                 "validation_zeros_not_filtered.csv")
    file_path = out_dir/file_name

    #sns.set_style("ticks")
    #sns.set_palette("pastel")

    if not file_path.is_file() or force_read:
        df_valid, df_sensor = read_data(data_dir=data_dir,
                                        out_dir=out_dir,
                                        force_read=force_read,
                                        n_files_max=n_files_max)
        df_valid = validate_data(df_sensor=df_sensor,
                                 df_valid=df_valid,
                                 parameters=parameters,
                                 skip_zeros=skip_zeros,
                                 visualize=detailed,
                                 out_dir=out_dir)
        df_valid.to_csv(file_path, index=False)
    else:
        # Skip computation of validation data if already present.
        print("Reading validation results for visualization...")
        print("To rerun the validation delete the results file:")
        print(file_path)
        df_valid = pd.read_csv(file_path, parse_dates=["Timestamp"])
    visualize_validation(df_valid=df_valid,
                         out_dir=out_dir,
                         skip_zeros=skip_zeros)


def parse_args():
    description = "Tailor-made utility to evaluate data from COVID station."
    formatter = argparse.RawTextHelpFormatter
    parser = argparse.ArgumentParser(add_help=False,
                                     formatter_class=formatter,
                                     description=description)
    parser.add_argument("-h", "--help", action="help",
                        help="Show this help text")
    parser.add_argument("-i", "--in-dir", required=True,
                        help="Input directory")
    parser.add_argument("-o", "--out-dir", default="./results/",
                        help="Output directory")
    parser.add_argument("-f", "--force-read", action="store_true",
                        help="Force re-reading of stored data.")
    parser.add_argument("--parameters", default=None, nargs="+",
                        choices=["Atemfrequenz", "Herzfrequenz", "Temperatur"],
                        help="Select the vital sign parameters to validate.")
    parser.add_argument("--n-files-max", default=None, type=int,
                        help="Limit the number sensor files to be read.")
    parser.add_argument("--skip-zeros", action="store_true",
                        help=("Skip zero measurements, assuming those are "
                              "invalid"))
    parser.add_argument("--detailed", action="store_true",
                        help="Plot additional details (slow).")
    parser.set_defaults(func=run)
    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
