# Heat map
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ..utils.plotter_helper import PlotterHelper
from ..utils.data_aggregator import DataAggregator
from ..utils.signal_properties import SignalProperties
from ..patient.patient_data_loader import PatientDataLoader

sns.set()


class HeatmapPlotter:
    loader = PatientDataLoader()
    aggregator = DataAggregator()

    def heatmap_days_hours(self, df, properties, patient_id):
        df_hm = self.aggregator.aggregate_data_hourly(df, properties)

        out_file_path = os.path.join(properties.out_dir,
                                     'Heatmap_' + patient_id + '_' + str(properties.normalization) + '.png')
        PlotterHelper.save_custom_plots(out_file_path, df, df_hm, patient_id, PlotterHelper.custom_subplots,
                                        self.custom_plot_fct, 0.3, 15, 3)


    def custom_plot_fct(self, ax, font_size, key, signal, x):
        min_scale = PlotterHelper.get_min_scale(key, signal)
        max_scale = PlotterHelper.get_max_scale(key, signal)
        signal = signal.values[:, np.newaxis]
        cbar_min_scale = int(min_scale)
        cbar_max_scale = int(max_scale + 0.5)
        cbar_step = int((cbar_max_scale - cbar_min_scale) / 2)
        if cbar_step <= 0:
            cbar_step = 1
        cbar_ticks = np.arange(cbar_min_scale, cbar_max_scale + 1, cbar_step)
        ax_sns = sns.heatmap(signal.transpose(), vmin=cbar_min_scale, vmax=cbar_max_scale,
                             cmap=SignalProperties.vital[key]['color_map'],
                             cbar_kws={'ticks': cbar_ticks, 'location' : 'left', 'use_gridspec' : False, 'pad':0.04},
                             ax=ax)
        cbar = ax_sns.collections[0].colorbar
        cbar.ax.tick_params(labelsize=font_size - 2)
        ax.set_yticks([])



    def save_heatmap(self, df_hm, properties, patient_id):
        out_file_path = os.path.join(properties.out_dir,
                                     'Heatmap_' + patient_id + '_' + str(properties.normalization) + '.png')
        plt.figure()
        cmap = sns.color_palette(properties.colormap, 20)
        drange = np.arange(properties.min_scale, properties.max_scale + 1, properties.max_scale / 20)
        sns.heatmap(df_hm.transpose(), vmin=properties.min_scale, vmax=properties.max_scale, cmap=cmap,
                    cbar_kws=dict(ticks=drange))

        plt.title('Patient: ' + patient_id)
        plt.savefig(out_file_path, bbox_inches='tight')
        plt.close()

    def plot_heatmaps(self, properties):
        for filename in os.listdir(properties.in_dir):
            if not (filename.endswith('csv')):
                continue

            patient_id = filename[properties.start_idx:properties.end_idx]
            print("processing file " + filename + " with pid=" + patient_id + " ...")

            df = self.loader.load_everion_patient_data(properties.in_dir, filename, ';')
            if not df.empty:
                self.heatmap_days_hours(df, properties, patient_id)
