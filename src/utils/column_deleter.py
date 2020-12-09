import glob
import os
from pathlib import Path
import pandas as pd

from patient.patient_data_loader import PatientDataLoader


class ColumnDeleter:
    loader = PatientDataLoader()

    def delete_columns(self, dir_name, out_dir, start_del, end_del, delimiter):

        for filename in os.listdir(dir_name):
            if not (filename.endswith('csv')):
                continue
            df = self.loader.load_everion_patient_data(dir_name, filename, delimiter, False)

            df_col = df.drop(df.columns[start_del:end_del], axis='columns')

            df_col.to_csv(os.path.join(out_dir, filename))
