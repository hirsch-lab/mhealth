import os
import pandas as pd

from ..patient.patient_data_loader import PatientDataLoader


class ColumnDeleter:
    loader = PatientDataLoader()

    def delete_columns(self, dir_name, out_dir, start_del, end_del, delimiter):

        for filename in os.listdir(dir_name):
            if not (filename.endswith('csv')):
                continue
            df = self.loader.load_everion_patient_data(dir_name, filename, delimiter, False)

            df_col = df.drop(df.columns[start_del:end_del], axis='columns')

            df_col.to_csv(os.path.join(out_dir, filename))

    def delete_columns_and_rename_header(self, dir_name, out_dir, start_del, end_del, delimiter):

        for filename in os.listdir(dir_name):
            if not (filename.endswith('csv')):
                continue
            df = self.loader.load_everion_patient_data(dir_name, filename, delimiter, False, True)

            if df.empty:
                continue

            df = df.drop(df.columns[start_del:end_del], axis='columns')

            df.columns = [
            'Dark',
            'Green',
            'Red',
            'IR',
            'AX',
            'AY',
            'AZ',
            'greencurr',
            'redcurr',
            'IRcurr',
            'ADCoeffs',
            'timestamp']

            side = str(filename[29:30])
            pid = str(filename[6:9])
            filename = pid + side.upper() + '_storage-vital_raw.csv'

            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert('UTC')
            df.to_csv(os.path.join(out_dir, filename), ';')
            print(filename + ' done')
