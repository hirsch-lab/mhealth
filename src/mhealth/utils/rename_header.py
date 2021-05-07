import pandas as pd
from pathlib import Path

from . import everion_keys


def rename_headers_inplace(df, keys):
    """
    In-place renaming and formatting.
    """
    df.rename(columns=keys, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("UTC")
    return


class RenameHeader:
    """
    Legacy
    """
    def renaming(self, df, keys, patient_id, out_dir):
        rename_headers_inplace(df=df, keys=keys)
        df.to_csv(Path(out_dir, "Renamed_Header_" + patient_id + ".csv"))

    def change_header(self, in_dir, start_idx, end_idx):
        in_dir = Path(in_dir)
        header_dir_name = Path(in_dir).name + "_header"
        out_dir = in_dir.parent / header_dir_name
        if not out_dir.is_dir():
            out_dir.mkdri(parents=True)

        for filepath in sorted(in_dir.glob("*.csv")):
            df = pd.DataFrame(pd.read_csv(filepath, sep=";"))
            patient_id = filepath.name[start_idx:end_idx]
            self.renaming(df=df, keys=everion_keys.TAG_NAMES_MIXED_VITAL_RAW,
                          patient_id=patient_id, out_dir=out_dir)





