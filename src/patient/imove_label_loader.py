import os
import pandas as pd
from natsort import natsort

from patient.patient_data_loader import PatientDataLoader


class ImoveLabelLoader:
    loader = PatientDataLoader()

    def load_labels(self, dir_name, filename, tz_to_zurich=True):
        print("loading xlsx file " + filename + " ...")

        path = os.path.join(dir_name, filename)
        if os.path.getsize(path) <= 0:
            print("file is empty")
            return pd.DataFrame()

        df = pd.read_excel(path, engine='openpyxl')
        df.drop(df.tail(1).index, inplace=True)
        df = df.iloc[:, 0].str.split(',', expand=True)
        header = df.iloc[0]
        df = df[1:]
        df.columns = header
        df['start_date'] = pd.to_datetime(df.Date.astype(str)+' '+df.Start.astype(str))
        df['duration'] = pd.to_timedelta(df['Time'])
        df['end_date'] = df['start_date'] + df['duration']

        # TODO: convert to same time zone as input data
        #if tz_to_zurich:
        #    df['Date'] = pd.to_datetime(df['Date']).dt.tz_convert('Europe/Zurich')
        #else:
        #    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert('UTC')

        return df


    def merge_data_and_labels(self,data_dir, label_dir, id_range, in_file_suffix):

        files_sorted = natsort.natsorted(os.listdir(data_dir))

        for count in range(id_range):
            id = str(count + 1).zfill(3)

            found = False
            for i, filename in enumerate(files_sorted):
                if filename.__contains__(id):
                    found = True

            if not (found):
                print("file not found with id: ", id)

            else:
                print("processing id: ", id, " ...")

                df1 = self.load_labels(label_dir, self.get_label_filename(1, id))
                df2 = self.load_labels(label_dir, self.get_label_filename(2, id))
                df3 = self.load_labels(label_dir, self.get_label_filename(3, id))

                filename_l = id + 'L' + in_file_suffix + '.csv'
                df_l = self.loader.load_everion_patient_data(data_dir, filename_l, ';')

                df_l = df_l.set_index(['timestamp'])
                df_l['mobility_index'] = ''

                df1.apply(lambda row: self.add_label(row, df_l), axis=1)

                tmp = 1

        print("num files: ", len(files_sorted))

    def add_label(self, label_row, df):
        start_time = label_row['start_date']
        end_time = label_row['end_date']
        label = label_row['Task']
        #mask = df['mobility_index'].between_time(start_time, end_time)
        #mask = (df > start_time) & (df <= end_time)

        df[mask] = label

        tmp = 0



    def get_label_filename(self, day, id):
        id_prefix = id + '-' + str(day) + '.xlsx'
        return id_prefix

