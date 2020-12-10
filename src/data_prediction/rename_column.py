import os
import glob
import pandas as pd
from pathlib import Path

from patient.patient_data_loader import PatientDataLoader


class ReplaceValues:

    activity_class = {'0':'NaN',
                    '1':'0',
                    '2':'1',
                    '3':'1',
                    '4':'1',
                    '5':'1',
                    '6':'1',
                    '7':'1',
                    '8':'1',
                    '9':'1',
                    '10':'1',
                    '11':'1',
                    '12':'1',
                    '13':'1',
                    '14':'1',
                    '15':'1',
                    '16':'0',
                    '17':'1',
                    '18':'0',
                    '19':'0',
                    '20':'0',
                    }

    loader = PatientDataLoader()

    def replace_values(self, df, activity_class, patient_id, out_dir):
        ''' replace calues in Classification column using activity_class lookup '''

        for key, value in activity_class.items():
            df['Classification'] = df['Classification'].replace(int(key), value)

        df.to_csv(Path(out_dir, 'Activity_class_' + patient_id + '.csv'))

    def change_values(self, dir_name, start_idx, end_idx):
        header_dir = Path(dir_name).name + '_classification'
        out_dir = os.path.join(os.path.join(dir_name, os.pardir), header_dir)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        for filename in glob.glob(dir_name + '*.csv'):
            #df = pd.DataFrame(pd.read_csv(filename, sep=';'))
            df = self.loader.load_everion_patient_data(dir_name, filename, ';')
            patient_id = filename[start_idx:end_idx]

            # patient_dir_name = os.path.join(out_dir, patient_id)
            # if not os.path.exists(patient_dir_name):
            #     os.mkdir(patient_dir_name)

            #self.renaming(df, self.keys, patient_id, out_dir) #patient_dir_name)
            self.replace_values(df, self.activity_class, patient_id, out_dir)



