import os
import pandas as pd
from pathlib import Path
from natsort import natsort


def get_label_filename(day, pat_id):
    id_prefix = pat_id + "-" + str(day) + ".xlsx"
    return id_prefix

def load_labels(filepath, tz_to_zurich=True):
    filepath = Path(filepath)
    if not filepath.is_file():
        print("Warning: file does not exist (%s)" % filepath.name)
        return pd.DataFrame()
    df = pd.read_excel(filepath, engine="openpyxl")
    df.drop(df.tail(1).index, inplace=True)
    df = df.iloc[:, 0].str.split(",", expand=True)
    header = df.iloc[0]
    df = df[1:]
    df.columns = header
    df.columns.name = None

    # Fix typos in the manually created files
    df["Task"] = df["Task"].str.lower()
    df["Task"] = df["Task"].replace({"t": "temp",
                                     "temo": "temp",
                                     #"fault" : "default",  # ????????
                                     "df": "default",
                                     "def": "default",
                                     "defaukt": "default",
                                     })
    # Format date/time information
    start = pd.to_datetime(df["Date"].astype(str) + " " +
                           df["Start"].astype(str))
    if tz_to_zurich:
        start = start.dt.tz_localize("Europe/Zurich")
    duration = pd.to_timedelta(df["Time"])
    stop = start + duration
    df["StartDate"] = start
    df["Duration"] = duration
    df["EndDate"] = stop
    df = df.drop(["Start", "Time", "Date"], axis=1)
    # Reverse order
    df = df[::-1]

    # Drop columns with axis labeled [None]
    if None in df.columns:
        # A col name None is impractical
        df = df.rename({None:"Unnamed"}, axis=1)
        # Assert we don't lose any information
        assert not df["Unnamed"].astype(bool).any()
        df = df.drop("Unnamed", axis=1)
    return df


def merge_labels(df, df_labels):
    """
    Modifies df in-place.
    """
    if not df.empty and not df_labels.empty:
        # Specify dtype to avoid warnings regarding mixed
        # dtypes when saving into a HDF store.
        df["DeMortonLabel"] = pd.Series(dtype=str)  # {str} + {None}
        df["DeMortonDay"] = pd.Series(dtype=float)  # {int} + {None}
        df["DeMorton"] = pd.Series(dtype=float)     # {True, False, None}
        df_labels.apply(lambda row: add_label(row, df), axis=1)
    return df


def add_label(label_row, df):
    start = label_row["StartDate"]
    end = label_row["EndDate"]
    sel = (df.timestamp >= start) & (df.timestamp <= end)

    df.loc[sel, "DeMorton"] = 1
    if "Task" in label_row or True:
        # Task is always available!
        label = label_row["Task"]
        df.loc[sel, "DeMortonLabel"] = label
    if "Day" in label_row:
        # Day is added optionally by calling script.
        day = label_row["Day"]
        df.loc[sel, "DeMortonDay"] = day

