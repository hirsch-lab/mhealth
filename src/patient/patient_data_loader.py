import csv
import os
import pandas as pd


class PatientDataLoader:

    def load_everion_patient_data(self, dir_name, filename, csv_delimiter, tz_to_zurich=True, drop_first_row=False):
        print("loading everion data from file " + filename + " ...")

        csv_in_file = os.path.join(dir_name, filename)
        if not os.path.exists(csv_in_file) or os.path.getsize(csv_in_file) <= 0:
            print("csv file is empty")
            return pd.DataFrame()

        df = pd.read_csv(csv_in_file, sep=csv_delimiter)
        if drop_first_row:
            df.drop(df.head(1).index, inplace=True)

        if tz_to_zurich:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert('Europe/Zurich')
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert('UTC')

        return df

    @staticmethod
    def load_extra_patient_data(filename):
        new_data_dict = {}
        with open(filename, 'r', encoding='utf-8-sig') as data_file:
            data = csv.DictReader(data_file, delimiter=";")
            for row in data:
                item = new_data_dict.get(row["pid"], dict())
                item["gender_code"] = int(row["gender_code"])
                item['age'] = int(row["age"])

                new_data_dict[row["pid"]] = item

        return new_data_dict
