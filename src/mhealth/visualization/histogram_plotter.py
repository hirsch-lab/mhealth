import os
import seaborn as sns
import matplotlib.pyplot as plt

from ..patient.patient_data_loader import PatientDataLoader

sns.set()

class HistogramPlotter:
    loader = PatientDataLoader()

    def plot_histogram(self, out_dir, patient_id, df, keys):
        for key in keys:
            if df[key].isna().all():
                continue

            fig = plt.figure(figsize=(12, 6))
            hrv = fig.add_subplot(121)

            hrv.hist(df[key], bins=80, color='blue')
            hrv.set_title('Patient ' + patient_id + ': ' + key)

            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

            patient_dir_name = os.path.join(out_dir, patient_id)
            if not os.path.exists(patient_dir_name):
                os.mkdir(patient_dir_name)

            fig.savefig(os.path.join(patient_dir_name, 'Histogram_' + patient_id + '_' + key + '.png'), bbox_inches='tight')
            plt.close(fig)

    def plot_all_histograms(self, in_dir, out_dir, start_idx, end_idx, keys):
        for filename in os.listdir(in_dir):
            if not (filename.endswith('csv')):
                continue

            patient_id = filename[start_idx:end_idx]
            print('Processing patient with pid=', patient_id, ', from filename=', filename)

            df = self.loader.load_everion_patient_data(in_dir, filename, ';')
            if df.empty:
                continue

            self.plot_histogram(out_dir, patient_id, df, keys)
