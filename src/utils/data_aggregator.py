import os
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
import pandas as pd

from patient.patient_data_loader import PatientDataLoader


class DataAggregator:
    loader = PatientDataLoader()

    def mean_data(self, df, patient_id, out_dir):
        df_agg = pd.DataFrame({'mean': df.agg("mean", axis="rows")})

        df_agg.to_csv(os.path.join(out_dir, 'Data_Aggregation_' + patient_id + '.csv'))


    def aggregate_data(self, dir_name, out_dir, start_idx, end_idx):
        for filename in os.listdir(dir_name):
            if filename.endswith('csv'):
                df = self.loader.load_everion_patient_data(dir_name, filename, ';')
                patient_id = filename[start_idx:end_idx]

                self.mean_data(df, patient_id, out_dir)

    def aggregate_data_minutes(self, df, properties):
        df['minute'] = df['timestamp'].dt.minute
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['year'] = df['timestamp'].dt.year
        df.sort_index(axis=1, ascending=True, inplace=True)
        df_m = pd.DataFrame({'minute': df.minute, 'hour': df.hour,'day': df.day, 'month': df.month, 'year': df.year})
        self.prepare_map_keys(df, df_m, properties)
        df_m = df_m.groupby(['year', 'month', 'day', 'hour', 'minute']).mean()
        df_m2 = self.pad_empty_minutes(df_m)
        df_m2.sort_index(axis=0, ascending=True, inplace=True)
        return df_m2


    def aggregate_data_hourly(self, df, properties):
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['year'] = df['timestamp'].dt.year
        df.sort_index(axis=1, ascending=True, inplace=True)
        df_ = pd.DataFrame({'hour': df.hour, 'day': df.day, 'month': df.month, 'year': df.year})
        self.prepare_map_keys(df, df_, properties)
        df_h = df_.groupby(['year', 'month', 'day', 'hour']).mean()
        df_h2 = self.pad_empty_hours(df_h)
        df_h2.sort_index(axis=0, ascending=True, inplace=True)
        return df_h2

    def pad_empty_minutes(self, df):
        counter = 0
        previous_date = datetime.now()
        for index, row in df.iterrows():
            current_date = self.date_from_multiindex_m(index)
            if counter <= 0:
                previous_date = self.date_from_multiindex_m(index)
                counter += 1
                continue

            min_diff = pd.Timedelta(current_date - previous_date).seconds / 60.0
            min_diff += pd.Timedelta(current_date - previous_date).days * 24 * 60
            new_row = pd.DataFrame.copy(row)
            if (min_diff > 1) | ((min_diff <= -1) & (min_diff > -59)):
                for i in range(int(min_diff) - 1):
                    new_timestamp = self.date_from_multiindex_m(index)
                    new_timestamp -= timedelta(minutes=1)
                    new_row[:] = np.nan
                    new_index = (new_timestamp.year, new_timestamp.month, new_timestamp.day, new_timestamp.hour,
                                 new_timestamp.minute)
                    new_row.name = new_index
                    df = df.append(new_row)
                    index = new_index

            previous_date = current_date
            counter += 1
        return df

    def pad_empty_hours(self, df):
        counter = 0
        previous_date = datetime.now()
        for index, row in df.iterrows():
            current_date = self.date_from_multiindex(index)
            if counter <= 0:
                previous_date = self.date_from_multiindex(index)
                counter += 1
                continue

            hours_diff = pd.Timedelta(current_date - previous_date).seconds / 3600.0
            hours_diff += pd.Timedelta(current_date - previous_date).days * 24
            new_row = pd.DataFrame.copy(row)
            if (hours_diff > 1) | ((hours_diff <= -1) & (hours_diff > -23)):
                for i in range(int(hours_diff) - 1):
                    new_timestamp = self.date_from_multiindex(index)
                    new_timestamp -= timedelta(hours=1)
                    new_row[:] = np.nan
                    new_index = (new_timestamp.year, new_timestamp.month, new_timestamp.day, new_timestamp.hour)
                    new_row.name = new_index
                    df = df.append(new_row)
                    index = new_index

            previous_date = current_date
            counter += 1
        return df

    def date_from_multiindex(self, index):
        return datetime(index[0], index[1], index[2], index[3])

    def date_from_multiindex_m(self, index):
        return datetime(index[0], index[1], index[2], index[3], index[4])

    def prepare_map_keys(self, df, df_h, properties):
        for key in properties.keys:
            df[key] = df[key].mean() + df[key] - df[key].mean()

            if properties.normalization == Normalization.MAX_NORM:
                df[key] = df[key] / df[key].max()

            if properties.normalization == Normalization.MIN_MAX_NORM:
                df[key] = (df[key] - df[key].min()) / (df[key].max() - df[key].min())

            df_h[properties.short_keys[key]] = df[key]

        df_h.sort_index(axis=1, ascending=True, inplace=True)



class Normalization(Enum):
    NONE = 1
    MAX_NORM = 2
    MIN_MAX_NORM = 3
