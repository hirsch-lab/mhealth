import os

import numpy as np

from patient.patient_data_loader import PatientDataLoader


class QualityFilter:
    loader = PatientDataLoader()

    def filter_bad_quality_vital(self, in_dir, out_dir, in_file_name, min_quality):
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        csv_out_file = os.path.join(out_dir, in_file_name)
        if os.path.exists(csv_out_file):
            os.remove(csv_out_file)

        df = self.loader.load_everion_patient_data(in_dir, in_file_name, ';', False)
        if df.empty:
            return

        self.set_bad_quality_nan(df, min_quality, 'core_temperature')
        self.set_bad_quality_nan(df, min_quality, 'oxygen_saturation')
        self.set_bad_quality_nan(df, min_quality, 'activity_classification')
        self.set_bad_quality_nan(df, min_quality, 'energy')
        self.set_bad_quality_nan(df, min_quality, 'heart_rate_variability')
        self.set_bad_quality_nan(df, min_quality, 'respiration_rate')

        self.set_bad_quality_nan_range(df, min_quality, 'heart_rate', range(0,18))
        self.set_bad_quality_nan_range(df, min_quality, 'heart_rate', range(20,22))
        self.set_bad_quality_nan_range(df, min_quality, 'heart_rate', range(27,29))

        df.to_csv(csv_out_file, sep=';')

    def filter_bad_quality_mixed_vital_raw(self, in_dir, out_dir, in_file_name, min_quality):
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        csv_out_file = os.path.join(out_dir, in_file_name)
        if os.path.exists(csv_out_file):
            os.remove(csv_out_file)

        df = self.loader.load_everion_patient_data(in_dir, in_file_name, ';')
        if df.empty:
            return

        self.filter_quality(df, min_quality, 'Classification', 'QualityClassification')
        self.filter_quality(df, min_quality, 'SPo2', 'SPO2Q')

        self.filter_quality_range(df, min_quality, range(0,19), 'HRQ')
        df.to_csv(csv_out_file, sep=';')

    def set_bad_quality_nan_range(self, df, min_quality, signal_name, range):
        signal_quality_name = signal_name + '_quality'
        self.filter_quality_range(df, min_quality, range, signal_quality_name)

    def filter_quality_range(self, df, min_quality, range, signal_quality_name):
        df.iloc[df[signal_quality_name] < min_quality, range] = np.nan
        df.iloc[df[signal_quality_name] > 100, range] = np.nan

    def set_bad_quality_nan(self, df, min_quality, signal_name):
        signal_quality_name = signal_name + '_quality'
        self.filter_quality(df, min_quality, signal_name, signal_quality_name)

    def filter_quality(self, df, min_quality, signal_name, signal_quality_name):
        df.loc[df[signal_quality_name] < min_quality, signal_name] = np.nan
        df.loc[df[signal_quality_name] > 100, signal_name] = np.nan
