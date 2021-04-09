import os
import pandas as pd
from pathlib import Path
from natsort import natsort

from .patient_data_loader import PatientDataLoader


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
    df["Task"] = df["Task"].str.lower()
    df["start_date"] = pd.to_datetime(df.Date.astype(str) + " "
                                      + df.Start.astype(str))
    df["start_date"] = df["start_date"].dt.tz_localize("Europe/Zurich")
    df["duration"] = pd.to_timedelta(df["Time"])
    df["end_date"] = df["start_date"] + df["duration"]
    return df

def load_labels_for_patient(labels_dir, pat_id):
    labels_dir = Path(labels_dir)
    df1 = load_labels(labels_dir / get_label_filename(day=1, pat_id=pat_id))
    df2 = load_labels(labels_dir / get_label_filename(day=2, pat_id=pat_id))
    df3 = load_labels(labels_dir / get_label_filename(day=3, pat_id=pat_id))
    df = pd.concat([df1, df2, df3], axis=0)
    return df

def merge_labels(df, df_labels):
    """
    Modifies df in-place.
    """
    if not df.empty and not df_labels.empty:
        df["DeMortonLabel"] = None
        df["DeMorton"] = None
        df_labels.apply(lambda row: add_label(row, df), axis=1)
    return df


def add_label(label_row, df):
    start = label_row["start_date"]
    end = label_row["end_date"]
    label = label_row["Task"]

    sel = (df.timestamp >= start) & (df.timestamp <= end)
    df.loc[sel, "DeMortonLabel"] = label
    df.loc[sel, "DeMorton"] = 1



class ImoveLabelLoader:
    """
    Legacy
    """
    loader = PatientDataLoader()

    def load_labels(self, dir_name, filename, tz_to_zurich=True):
        print("loading xlsx file " + filename + " ...")

        path = os.path.join(dir_name, filename)
        if not os.path.exists(path):
            print("Warning: file does not exist (%s)" % path)
            return pd.DataFrame()

        df = load_labels(filepath=path, tz_to_zurich=tz_to_zurich)
        return df

    def merge_data_and_labels(self, data_dir, label_dir, out_dir, start_range, end_range, in_file_suffix):

        files_sorted = natsort.natsorted(os.listdir(data_dir))

        for count in range(start_range, end_range+1):
            pat_id = str(count).zfill(3)

            found = any(pat_id in fn for fn in files_sorted)
            if not (found):
                print("Warning: file not found with id: ", pat_id)

            else:
                print("processing id: ", pat_id, " ...")

                label_dir = Path(label_dir)
                df1 = load_labels(label_dir / get_label_filename(day=1, pat_id=pat_id))
                df2 = load_labels(label_dir / get_label_filename(day=2, pat_id=pat_id))
                df3 = load_labels(label_dir / get_label_filename(day=3, pat_id=pat_id))

                filename = pat_id + "L" + in_file_suffix + ".csv"
                self.create_labels(data_dir, out_dir, df1, df2, df3, filename)
                filename = pat_id + "R" + in_file_suffix + ".csv"
                self.create_labels(data_dir, out_dir, df1, df2, df3, filename)

        print("num files: ", len(files_sorted))

    def create_labels(self, data_dir, out_dir, df1, df2, df3, filename):
        df = self.loader.load_everion_patient_data(data_dir, filename, ";", True)
        if not df.empty:
            df["DeMortonLabel"] = ""
            df["DeMorton"] = ""

            if not df1.empty:
                df1.apply(lambda row: add_label(row, df), axis=1)
            if not df2.empty:
                df2.apply(lambda row: add_label(row, df), axis=1)
            if not df3.empty:
                df3.apply(lambda row: add_label(row, df), axis=1)

            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("UTC")
            df.to_csv(os.path.join(out_dir, filename), ";")

