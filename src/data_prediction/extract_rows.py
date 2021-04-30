import glob
import os
from pathlib import Path
import pandas as pd

from patient.patient_data_loader import PatientDataLoader


class DataSelection:

    loader = PatientDataLoader()

    def extract_rows(self, df, patient_id, out_dir_name, end_del):
        df_row = df[:end_del]

        df_row.to_csv(Path(out_dir_name, 'Data_Selected_' + patient_id + '.csv'))

    def compress_data(self, dir_name, start_idx, end_idx, end_del):
        short_dir = Path(dir_name).name + '_selected_34700'
        out_dir_name = os.path.join(os.path.join(dir_name, os.pardir), short_dir)
        if not os.path.exists(out_dir_name):
            os.mkdir(out_dir_name)

        for filename in glob.glob(dir_name + '*.csv'):
            df = pd.DataFrame(pd.read_csv(filename, sep=';', error_bad_lines=False))
            patient_id = filename[start_idx:end_idx]

            self.extract_rows(df, patient_id, out_dir_name, end_del)
