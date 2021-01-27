import os

import numpy as np
import pandas as pd

from patient.patient_data_loader import PatientDataLoader


# All operations modify input df in-place!

def set_bad_quality_nan_range(df, min_quality, signal_name, range):
    signal_quality_name = signal_name + '_quality'
    filter_quality_range(df, min_quality, range, signal_quality_name)

def filter_quality_range(df, min_quality, range, signal_quality_name):
    assert False, "Deprecated. Use filter_quality_except() instead."
    df.iloc[df[signal_quality_name] < min_quality, range] = np.nan
    df.iloc[df[signal_quality_name] > 100, range] = np.nan

def filter_quality_except(df, min_quality, except_cols, signal_quality_name):
    col_mask = ~df.columns.isin(except_cols)
    df.loc[df[signal_quality_name] < min_quality, col_mask] = np.nan
    df.loc[df[signal_quality_name] > 100, col_mask] = np.nan

def set_bad_quality_nan(df, min_quality, signal_name):
    signal_quality_name = signal_name + '_quality'
    filter_quality(df, min_quality, signal_name, signal_quality_name)

def filter_quality(df, min_quality, signal_name, signal_quality_name):
    df.loc[df[signal_quality_name] < min_quality, signal_name] = np.nan
    df.loc[df[signal_quality_name] > 100, signal_name] = np.nan

def filter_bad_quality_vital(df, min_quality):
    set_bad_quality_nan(df, min_quality, 'core_temperature')
    set_bad_quality_nan(df, min_quality, 'oxygen_saturation')
    set_bad_quality_nan(df, min_quality, 'activity_classification')
    set_bad_quality_nan(df, min_quality, 'energy')
    set_bad_quality_nan(df, min_quality, 'heart_rate_variability')
    set_bad_quality_nan(df, min_quality, 'respiration_rate')

    # quality signals omitted to remember original quality
    set_bad_quality_nan_range(df, min_quality, 'heart_rate', range(0,18))
    set_bad_quality_nan_range(df, min_quality, 'heart_rate', range(20,22))
    set_bad_quality_nan_range(df, min_quality, 'heart_rate', range(27,29))

def filter_bad_quality_mixed_vital_raw(df, min_quality):
    """
    Filter data selected by everion_keys.MIX
    """
    filter_quality(df, min_quality, 'Classification', 'QualityClassification')
    filter_quality(df, min_quality, 'SPo2', 'SPO2Q')

    # Quality signals omitted to remember original quality.
    quality_cols = ["HRQ", "SPo2Q", "QualityClassification", "timestamp"]
    filter_quality_except(df, min_quality, quality_cols, 'HRQ')


class QualityFilter:
    """
    Legacy
    """
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

        # In-place.
        filter_bad_quality_vital(df, min_quality)

        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert('UTC')
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

        # In-place.
        filter_bad_quality_mixed_vital_raw(df, min_quality)

        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert('UTC')
        df.to_csv(csv_out_file, sep=';')

