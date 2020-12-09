import os

import matplotlib.dates as d
import matplotlib.pyplot as plt
import seaborn as sns

from patient.patient_data_loader import PatientDataLoader
from utils.data_aggregator import DataAggregator
from utils.everion_keys import EverionKeys
from utils.plotter_helper import PlotterHelper

sns.set()


class Plotter:
    loader = PatientDataLoader()
    aggregator = DataAggregator()

    def plot_patient(self, in_dir, out_dir, in_file_name):
        patient_id = in_file_name[:3]
        out_dir_subset = os.path.join(out_dir, 'subset')
        out_dir_quality = os.path.join(out_dir, 'quality')
        out_dir_all = os.path.join(out_dir, 'all')
        out_dir_qv = os.path.join(out_dir, 'quality_values')

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        if not os.path.exists(out_dir_all):
            os.mkdir(out_dir_all)
        if not os.path.exists(out_dir_quality):
            os.mkdir(out_dir_quality)
        if not os.path.exists(out_dir_subset):
            os.mkdir(out_dir_subset)
        if not os.path.exists(out_dir_qv):
            os.mkdir(out_dir_qv)

        df = self.loader.load_everion_patient_data(in_dir, in_file_name, ';')

        if df.empty:
            return

        df['barometer_pressure'] = df['barometer_pressure'] / 100

        self.generate_subset_plots(df, out_dir_subset, patient_id)
        self.generate_all_plots(df, out_dir_all, patient_id)
        self.generate_quality_plots(df, out_dir_quality, patient_id)
        self.generate_quality_value_plots(df, out_dir_qv, patient_id)

    def plot_patient_mixed_vital_raw(self, in_dir, out_dir, in_file_name, keys, start_idx, end_idx):
        patient_id = in_file_name[start_idx:end_idx]

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        df = self.loader.load_everion_patient_data(in_dir, in_file_name, ';')

        if df.empty:
            return

        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month

        self.plot_keys(df, patient_id, os.path.join(out_dir, patient_id + '.png'), keys)


    def generate_subset_plots(self, df, out_dir_subset, patient_id):
        keys = {'heart_rate', 'heart_rate_variability', 'oxygen_saturation', 'core_temperature', 'respiration_rate'}
        self.plot_keys(df, patient_id, os.path.join(out_dir_subset, patient_id + '_subset' + '.png'), keys)

    def generate_all_plots(self, df, out_dir_all, patient_id):
        self.plot_keys(df, patient_id, os.path.join(out_dir_all, patient_id + '_all' + '.png'), EverionKeys.all_vital)

    def generate_quality_plots(self, df, out_dir_quality, patient_id):
        keys = {'core_temperature_quality', 'respiration_rate_quality', 'heart_rate_variability_quality',
                'energy_quality', 'activity_classification_quality', 'oxygen_saturation_quality', 'heart_rate_quality'}
        self.plot_keys(df, patient_id, os.path.join(out_dir_quality, patient_id + '_quality' + '.png'), keys)

    def generate_quality_value_plots(self, df, out_dir_qv, patient_id):
        keys = {'core_temperature', 'core_temperature_quality'}
        self.plot_keys(df, patient_id, os.path.join(out_dir_qv, patient_id + '_t_qv' + '.png'), keys)
        keys = {'respiration_rate', 'respiration_rate_quality'}
        self.plot_keys(df, patient_id, os.path.join(out_dir_qv, patient_id + '_rr_qv' + '.png'), keys)
        keys = {'heart_rate_variability', 'heart_rate_variability_quality'}
        self.plot_keys(df, patient_id, os.path.join(out_dir_qv, patient_id + '_hrv_qv' + '.png'), keys)
        keys = {'heart_rate', 'heart_rate_quality'}
        self.plot_keys(df, patient_id, os.path.join(out_dir_qv, patient_id + '_hr_qv' + '.png'), keys)
        keys = {'oxygen_saturation', 'oxygen_saturation_quality'}
        self.plot_keys(df, patient_id, os.path.join(out_dir_qv, patient_id + '_spo2_qv' + '.png'), keys)

    def plot_keys(self, df, patient_id, out_filename, keys):
        fig, ax = plt.subplots(figsize=[20, 6])

        mdates = d.date2num(df['timestamp'])

        for key in keys:
            plt.plot_date(mdates, df[key], tz=None, xdate=True, linewidth=0.5, linestyle='solid', marker='')

        formatter = d.DateFormatter('%m/%d/%y %H:%M:%S')
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_tick_params(rotation=30, labelsize=10)

        plt.legend(bbox_to_anchor=(0.95, 1.15), loc='upper left', labels = keys)

        plt.xlabel('Time [sec]')
        plt.title('Patient: ' + patient_id)
        fig.savefig(out_filename, bbox_inches='tight')

        plt.close(fig)


    def plot_hourly_lines(self, properties):
        for filename in os.listdir(properties.in_dir):
            if not (filename.endswith('csv')):
                continue

            patient_id = filename[properties.start_idx:properties.end_idx]
            print("processing file " + filename + " with pid=" + patient_id + " ...")

            df = self.loader.load_everion_patient_data(properties.in_dir, filename, ';')
            if not df.empty:
                df_hm = self.aggregator.aggregate_data_hourly(df, properties)
                out_file_path = os.path.join(properties.out_dir, 'Hourly_lines_' + patient_id + '.png')
                PlotterHelper.save_custom_plots(out_file_path, df, df_hm, patient_id, self.custom_line_plot,
                                        self.custom_plot_fct_empty, 0, 15, 3)


    def custom_line_plot(self, ax, custom_plot_fct, df_hm, font_size, x, x_daily_lines, x_ticks):
        color_dict = {'HR': 'limegreen', 'HRV': 'silver', 'RR': 'yellow', 'SPO2': 'deepskyblue', 'Temp': 'white'}
        ax0 = df_hm.plot(xticks=x_ticks, figsize=(15,3), color=[color_dict.get(x, '#333333') for x in df_hm.columns])
        ax0.set_facecolor('dimgray')
        plt.grid(color='silver', linestyle='--', linewidth=0.7)

    def custom_plot_fct_empty(self):
        return

    def plot_hourly_lines_subplots(self, properties):
        for filename in os.listdir(properties.in_dir):
            if not (filename.endswith('csv')):
                continue

            patient_id = filename[properties.start_idx:properties.end_idx]
            print("processing file " + filename + " with pid=" + patient_id + " ...")

            df = self.loader.load_everion_patient_data(properties.in_dir, filename, ';')
            if not df.empty:
                df_hm = self.aggregator.aggregate_data_hourly(df, properties)
                out_file_path = os.path.join(properties.out_dir, 'Hourly_lines2_' + patient_id + '.png')

                PlotterHelper.save_custom_plots(out_file_path, df, df_hm, patient_id, PlotterHelper.custom_subplots,
                                        self.custom_plot_fct, 0, 15, 3)


    def custom_plot_fct(self, ax, font_size, key, signal, x):
        ax.plot(x, signal.transpose())
        ax.set_ylabel(key, fontsize=font_size)
        ax.set_yticks([])

