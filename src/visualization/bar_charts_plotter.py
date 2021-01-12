import os

import numpy as np
import pandas as pd
import seaborn as sns

from utils.data_aggregator import DataAggregator
from patient.patient_data_loader import PatientDataLoader
from utils.plotter_helper import PlotterHelper
from utils.signal_properties import SignalProperties

import matplotlib.pyplot as plt

sns.set()

class BarChartsPlotter:
    loader = PatientDataLoader()
    aggregator = DataAggregator()

    def plot_bars_multiscale(self, properties):
        for filename in os.listdir(properties.in_dir):
            if not (filename.endswith('csv')):
                continue

            patient_id = filename[properties.start_idx:properties.end_idx]
            print("processing file " + filename + " with pid=" + patient_id + " ...")

            df_s = self.loader.load_everion_patient_data(properties.in_dir, filename, ';')
            if not df_s.empty:
                df_m = self.aggregator.aggregate_data_minutes(df_s, properties)
                df_h = self.aggregator.aggregate_data_hourly(df_s, properties)
                self.save_plots_multiscale(df_s, df_m, df_h, properties, patient_id)


    def save_plots_multiscale(self, df_s, df_m, df_h, properties, patient_id):
        out_file_path = os.path.join(properties.out_dir, 'Bars_multiscale_' + patient_id + '_mean.png')
        self.save_multiscale_plots(out_file_path, df_s, df_m, df_h, patient_id, self.custom_subplots_multiscale,
                                   self.custom_plot_fct, 0.3, 15, 5)

    def save_multiscale_plots(self, out_file_path, df_s, df_m, df_h, patient_id, custom_subplots, custom_plot_fct, hspace,
                              fig_width,
                              fig_height):

        fig, axn = plt.subplots(2, 1, sharex=False, sharey=False, figsize=(fig_width, fig_height))
        plt.subplots_adjust(hspace=hspace)
        font_size = 10

        index = df_h.reset_index(level=['year', 'month', 'day']).index
        if len(index) < 1:
            return

        step = PlotterHelper.get_step_size(len(index))
        offset = step - (index[0] % step)
        x_tick_labels = PlotterHelper.get_hourly_x_tick_labels(df_h.index, offset, step)
        x_ticks = np.arange(offset, len(index), step)

        if len(x_ticks) <= 0:
         x_ticks = np.arange(0.5, len(index), 1)
        x = np.arange(0, len(index), 1)

        offset_lines = 24 - (index[0] % 24)
        x_daily_lines = np.arange(offset_lines, len(index), 24)

        custom_subplots(axn, df_m, df_h, font_size, x, x_daily_lines, x_ticks, x_tick_labels)

        fig.suptitle(PlotterHelper.get_plot_title(patient_id, df_s['timestamp'].dt.tz_convert(None).min(),
                                                  df_s['timestamp'].dt.tz_convert(None).max()),
                     fontsize=font_size)

        plt.savefig(out_file_path, bbox_inches='tight')
        plt.close(fig)




    def plot_bars_hourly(self, properties):
        for filename in os.listdir(properties.in_dir):
            if not (filename.endswith('csv')):
                continue

            patient_id = filename[properties.start_idx:properties.end_idx]
            print("processing file " + filename + " with pid=" + patient_id + " ...")

            df = self.loader.load_everion_patient_data(properties.in_dir, filename, ';')
            if not df.empty:
                df_hm = self.aggregator.aggregate_data_hourly(df, properties)
                self.save_hourly_plots(df, df_hm, properties, patient_id)

    def save_hourly_plots(self, df, df_hm, properties, patient_id):
        out_file_path = os.path.join(properties.out_dir, 'Bars_' + patient_id + '_mean.png')
        PlotterHelper.save_custom_plots(out_file_path, df, df_hm, patient_id, self.custom_subplots,
                                        self.custom_plot_fct, 0.3, 15, 5)


    def custom_subplots(self, axn, custom_plot, df_hm, font_size, x, x_daily_lines, x_ticks):
        centers = self.get_centers(df_hm)
        df_hm -= centers
        df_ma = pd.DataFrame.copy(df_hm)
        df_ma = df_ma.rolling(window=4).mean()
        counter = 0
        for (key, signal) in df_hm.iteritems():
            if signal.isna().all():
                continue
            ax = axn.flat[counter]
            custom_plot(ax, centers, key, signal, df_ma.iloc[:, counter], x)

            ax.set_ylabel(key, fontsize=font_size)
            ax.set_xticks(x_ticks)

            ax.vlines(x_ticks, *ax.get_ylim(), color='k', linestyles='--', linewidth=0.25)
            ax.vlines(x_daily_lines, *ax.get_ylim(), color='k', linestyles='--', linewidth=0.75)

            counter += 1

    def custom_plot_fct(self, ax, centers, key, signal, signal_2, x):
        cbar = PlotterHelper.get_bar_color(signal, SignalProperties.colors_d[3], SignalProperties.colors_d[0])
        ax.bar(x, signal.transpose(), color=cbar, edgecolor='None', linewidth=0.5)
        ax.plot(x, signal_2, color='red')
        y_ticks = ax.get_yticks()
        y_ticks_labels = y_ticks + centers[key]
        y_ticks_labels = np.around(y_ticks_labels, decimals=1)
        # Exact positions of rounded labels
        y_ticks = y_ticks_labels - centers[key]
        # Set tick positions and labels together (issues a warning otherwise).
        # https://github.com/matplotlib/matplotlib/issues/18848
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks_labels)


    def custom_subplots_multiscale(self, axn, df_m, df_h, font_size, x, x_daily_lines, x_ticks, x_tick_labels):
        centers = self.get_centers(df_h)
        df_h -= centers
        key = 'RR'
        ax = axn.flat[0]
        cbar = PlotterHelper.get_bar_color(df_h, SignalProperties.colors_d[3], SignalProperties.colors_d[0])
        ax.bar(x, df_h[key].transpose(), edgecolor='None', linewidth=0.5)
        # See custom_plot_fct() for details.
        y_ticks = ax.get_yticks()
        y_ticks_labels = y_ticks + centers[key]
        y_ticks_labels = np.around(y_ticks_labels, decimals=1)
        y_ticks = y_ticks_labels - centers[key]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks_labels)
        ax.set_ylabel(key, fontsize=font_size)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)
        ax.vlines(x_ticks, *ax.get_ylim(), color='k', linestyles='--', linewidth=0.25)
        ax.vlines(x_daily_lines, *ax.get_ylim(), color='k', linestyles='--', linewidth=0.75)

        index = df_m.reset_index(level=['year', 'month', 'day','hour']).index
        x2 = np.arange(0, len(index), 1)
        ax2 = axn.flat[1]
        df_m -= centers
        ax2.bar(x2, df_m[key].transpose(), edgecolor='None', linewidth=0.5)
        ax2.set_ylabel(key, fontsize=font_size)
        #TODO: set labels properly
        #y_ticks2 = ax2.get_yticks()
        #y_ticks2 += centers[key]
        #ax2.set_yticklabels(np.around(y_ticks2, decimals=1))
        #ax2.set_xticks([])
        #ax2.set_xticklabels([])




    def get_centers(self, df_hm):
        centers = df_hm.mean()
        return centers
        # for (key, signal) in df_hm.iteritems():
        #     expected_min = SignalProperties.vitals[key]['expected_min']
        #     expected_max = SignalProperties.vitals[key]['expected_max']
        #     if not np.isnan(expected_min) and not np.isnan(expected_max):
        #         centers[key] = (expected_min+ expected_max)/2
        # return centers

