import os
import pandas as pd
import matplotlib.pyplot as plt

from ..patient.patient_data_loader import PatientDataLoader


class SignalPlotter:
    loader = PatientDataLoader()

    def plot_signal(self, in_dir, out_dir, signal_name):
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        df_signals = pd.DataFrame(columns=['001', '002', '007'])
        fig, ax = plt.subplots(figsize=[20, 6])
        #TODO: do not load files every single time, keep in cache and read from there

        for filename in os.listdir(in_dir):
            print('processing ', filename, ' ...')
            if not (filename.endswith('csv')):
                continue

            patient_id = filename[:3]

            df = self.loader.load_everion_patient_data(in_dir, filename,';')
            if df.empty:
                continue

            df_signals[patient_id] = df[signal_name]
            df[signal_name].plot(linewidth=1, ax=ax, label=patient_id)

        plt.title('Signal: ' + signal_name)
        plt.xlabel('Time [sec]')
        plt.legend(bbox_to_anchor=(0.95, 1.15), loc='upper left')
        fig.savefig(os.path.join(out_dir, signal_name + '.png'), bbox_inches='tight')


