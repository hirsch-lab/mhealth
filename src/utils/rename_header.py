import os
import glob
import pandas as pd
from pathlib import Path

#from patient_data_loader import PatientDataLoader


class RenameHeader:

    keys = {'2.1':  'HR',
            '4.1':	'HRQ',
            '6.1':	'SPo2',
            '8.1':	'SPO2Q',
            '10.1':	'BloodPressure',
            '14.1':	'BloodPerfusion',
            '18.1':	'Activity',
            '20.1':	'Classification',
            '22.1':	'QualityClassification',
            '24.1':	'steps',
            '26.1':	'Energy',
            '28.1':	'RespRate',
            '29.1':	'HRV',
            '42.1':	'phase',
            '44.1':	'phase',
            '46.1':	'localtemp',
            '48.1':	'objtemp',
            '50.1':	'baromtemp',
            '52.1':	'pressure',
            }

    # loader = PatientDataLoader()

    def renaming(self, df, keys, patient_id, out_dir):
        # df = self.loader.load_everion_patient_data(in_dir, filename)
        # if df.empty:
        #     return

        df.rename(columns=keys, inplace=True)

        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert('UTC')
        df.to_csv(Path(out_dir, 'Renamed_Header_' + patient_id + '.csv'))



    def change_header(self, dir_name, start_idx, end_idx):
        header_dir = Path(dir_name).name + '_header'
        out_dir = os.path.join(os.path.join(dir_name, os.pardir), header_dir)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        for filename in glob.glob(dir_name + '*.csv'):
            df = pd.DataFrame(pd.read_csv(filename, sep=';'))
            patient_id = filename[start_idx:end_idx]

            # patient_dir_name = os.path.join(out_dir, patient_id)
            # if not os.path.exists(patient_dir_name):
            #     os.mkdir(patient_dir_name)

            self.renaming(df, self.keys, patient_id, out_dir) #patient_dir_name)




