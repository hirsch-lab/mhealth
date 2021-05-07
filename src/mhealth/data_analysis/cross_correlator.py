import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..utils.data_aggregator import DataAggregator
from ..patient.patient_data_loader import PatientDataLoader
from ..utils.plotter_helper import PlotterHelper

sns.set()

class CrossCorrelator:
    loader = PatientDataLoader()
    aggregator = DataAggregator()

    def plot_daily_correlations(self, properties):
        for filename in os.listdir(properties.in_dir):
            if not (filename.endswith('csv')):
                continue

            patient_id = filename[properties.start_idx:properties.end_idx]
            print("processing file " + filename + " with pid=" + patient_id + " ...")

            df = self.loader.load_everion_patient_data(properties.in_dir, filename, ';')
            if not df.empty:
                df_h = self.aggregator.aggregate_data_hourly(df, properties)

                for (key, signal) in df_h.iteritems():
                    if signal.isna().all():
                        continue
                    self.plot_signal(df, df_h, key, patient_id, properties)



    def plot_signal(self, df, df_h, key, patient_id, properties):
        df_d = df_h.reset_index().pivot_table(index='hour', columns='day', values=key)
        df_corr = df_d.corr()
        out_file_path = os.path.join(properties.out_dir, 'Cross_' + patient_id + '_' + key + '.png')
        self.save_custom_plots(out_file_path, df, df_corr, patient_id, key)

    def plot_hourly_correlations(self, properties):
        for filename in os.listdir(properties.in_dir):
            if not (filename.endswith('csv')):
                continue

            patient_id = filename[properties.start_idx:properties.end_idx]
            print("processing file " + filename + " with pid=" + patient_id + " ...")

            df = self.loader.load_everion_patient_data(properties.in_dir, filename, ';')
            if not df.empty:
                df_hm = self.aggregator.aggregate_data_hourly(df, properties)
                df_corr = df_hm.corr()
                out_file_path = os.path.join(properties.out_dir, 'Cross_' + patient_id + '.png')
                self.save_custom_plots_hourly(out_file_path, df, df_corr, patient_id)

    def save_custom_plots(self, out_file_path, df, df_hm, patient_id, key):
        fig = plt.figure()
        font_size = 10
        mask = np.zeros_like(df_hm)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(df_hm, cmap='Spectral_r', annot=True, mask=mask, vmin=-1, vmax=1)
        plt.title(PlotterHelper.get_plot_title(patient_id, df['timestamp'].min(),
                                                  df['timestamp'].max()) + ' ' + key,
                     fontsize=font_size)

        plt.savefig(out_file_path, bbox_inches='tight')
        plt.close(fig)

    def save_custom_plots_hourly(self, out_file_path, df, df_hm, patient_id):
        fig = plt.figure()
        font_size = 10
        mask = np.zeros_like(df_hm)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(df_hm, cmap='Spectral_r', annot=True, mask=mask, vmin=-1, vmax=1)
        plt.title(PlotterHelper.get_plot_title(patient_id, df['timestamp'].min(), df['timestamp'].max()),
                     fontsize=font_size)

        plt.savefig(out_file_path, bbox_inches='tight')
        plt.close(fig)
