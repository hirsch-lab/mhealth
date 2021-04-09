import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ..patient.patient_data_loader import PatientDataLoader

sns.set()


class GenderAgeVisualizer:

    def get_patient_row(self, df, patient_id, new_data_dict):
        df_nested = pd.DataFrame(new_data_dict)
        pid = patient_id
        gender_code = df_nested[pid]['gender_code']
        gender = 'female'
        if gender_code == 1:
            gender = 'male'
        age = df_nested[pid]['age']

        df = df.mean(numeric_only=True)

        df_row = pd.DataFrame({'RR': df.RR, 'SpO2': df.SpO2, 'HR': df.HR,
                               'HRV': df.HRV, 'Temp': df.Temp, 'Age': age,
                               'Gender': gender, 'PID': pid}, index=[0])

        return df_row

    def plot_age_means(self, df, out_dir):
        males = df[df['Gender'] == 'male'].sort_values(by=['Age'])
        females = df[df['Gender'] == 'female'].sort_values(by=['Age'])

        window_size=3
        appendix = '_ma'
        for i, col in enumerate(df.columns[:-3]):
            males[col + appendix] = males[col].rolling(window=window_size).mean()
            females[col + appendix] = females[col].rolling(window=window_size).mean()

        fig, ax = plt.subplots(5, 1, sharex=True, sharey=False, figsize=(20, 10))

        for i, col in enumerate(df.columns[:-3]):
            sns.scatterplot(x="Age", y=col, ax=ax[i], data= df, hue = 'Gender')
            sns.lineplot(ax=ax[i], data=males, x='Age', y=col + appendix)
            sns.lineplot(ax=ax[i], data=females, x='Age', y=col + appendix)

        fig.suptitle('Signal means across ages')
        fig.tight_layout()
        fig.subplots_adjust(top=0.95)

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        fig.savefig(os.path.join(out_dir, 'Age_means.png'), bbox_inches='tight')
        plt.clf()
        plt.close(fig)


    def plot_data(self, in_dir, out_dir, start_idx, end_idx, lookup_table, keys, short_keys):
        column_names = ['RR', 'SpO2', 'HR', 'HRV', 'Temp', 'Age', 'Gender', 'PID']
        combined_df = pd.DataFrame(columns=column_names)
        loader = PatientDataLoader()
        for filename in os.listdir(in_dir):
            if not (filename.endswith('csv')):
                continue

            patient_id = filename[start_idx:end_idx]
            print("processing file " + filename + " with pid=" + patient_id + " ...")

            df = loader.load_everion_patient_data(in_dir, filename, ';')
            if df.empty:
                continue

            for key in keys:
                df[short_keys[key]] = df[key]

            df_row = self.get_patient_row(df, patient_id, lookup_table)
            combined_df = combined_df.append(df_row)

        self.plot_age_means(combined_df, out_dir)

