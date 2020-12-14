import os
import pandas as pd
from natsort import natsort

from patient.patient_data_loader import PatientDataLoader


class ImoveLabelLoader:
    loader = PatientDataLoader()

    def load_labels(self, dir_name, filename, tz_to_zurich=True):
        print("loading xlsx file " + filename + " ...")

        path = os.path.join(dir_name, filename)
        if not os.path.exists(path):
            print('file does not exist ' + path)
            return pd.DataFrame()

        df = pd.read_excel(path, engine='openpyxl')
        df.drop(df.tail(1).index, inplace=True)
        df = df.iloc[:, 0].str.split(',', expand=True)
        header = df.iloc[0]
        df = df[1:]
        df.columns = header
        df['start_date'] = pd.to_datetime(df.Date.astype(str)+' '+df.Start.astype(str)).dt.tz_localize('Europe/Zurich')
        df['duration'] = pd.to_timedelta(df['Time'])
        df['end_date'] = df['start_date'] + df['duration']

        return df


    def merge_data_and_labels(self,data_dir, label_dir, out_dir,  id_range, in_file_suffix):

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

                filename = id + 'L' + in_file_suffix + '.csv'
                self.create_labels(data_dir, out_dir, df1, df2, df3, filename)
                filename = id + 'R' + in_file_suffix + '.csv'
                self.create_labels(data_dir, out_dir, df1, df2, df3, filename)

        print("num files: ", len(files_sorted))

    def create_labels(self, data_dir, out_dir, df1, df2, df3, filename):
        df = self.loader.load_everion_patient_data(data_dir, filename, ';')
        if not df.empty:
            df = df.set_index(['timestamp'])
            df.sort_index()

            df['mobility_index'] = ''
            df['de_morton'] = 0

            if not df1.empty:
                df1.apply(lambda row: self.add_label(row, df), axis=1)
            if not df2.empty:
                df2.apply(lambda row: self.add_label(row, df), axis=1)
            if not df3.empty:
                df3.apply(lambda row: self.add_label(row, df), axis=1)

            df.to_csv(os.path.join(out_dir, filename))

    def add_label(self, label_row, df):

        tmp = label_row['start_date'] in df.index
        tmp2 = label_row['end_date'] in df.index
        # TODO: continue here
        if label_row['start_date'] in df.index and label_row['end_date'] in df.index:
            df['mobility_index'].loc[label_row['start_date']:label_row['end_date']] = str(label_row['Task'])
            df['de_morton'].loc[label_row['start_date']:label_row['end_date']] = 1
        else:
            print(
                'dates out of data range. start_time=' + str(label_row['start_date']) + ', end_time=' + str(
                    label_row['end_date']) + ', label=' +
                str(label_row['Task']))




    def get_label_filename(self, day, id):
        id_prefix = id + '-' + str(day) + '.xlsx'
        return id_prefix

