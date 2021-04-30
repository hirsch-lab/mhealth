import os
import datetime
from math import sqrt
from scipy import stats

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
import pingouin as pg

import seaborn as sns

from patient.patient_data_loader import PatientDataLoader

# dir_path = Path(__file__).parent
# src_path = (dir_path / ".." / ".." / "src").resolve()
# sys.path.insert(0, str(src_path))


class DataComparator:
    loader = PatientDataLoader()

    def load_files(self, dir_name, dir_nurse, start_idx, end_idx):
        # Conversion function: converter(14) returns "014"
        def converter(x): return str(x).zfill(3)
        vitals_masimo = pd.read_csv(dir_nurse, ";",
                                   parse_dates=[["Date", "Time"]],
                                   converters={"Record_id": converter})
        vitals_masimo["Date_Time"] = vitals_masimo["Date_Time"].dt.tz_localize("Europe/Zurich")

        # Alternatively, convert explicitly:
        #vitals_nurse["Record_id"] = vitals_nurse["Record_id"].apply(converter)

        dir_name = Path(dir_name)
        mvse_dir = dir_name.name + "_mvse_with_comp_Ttest"
        out_dir_name = os.path.join(os.path.join(dir_name, os.pardir), mvse_dir)
        if not os.path.exists(out_dir_name):
            os.mkdir(out_dir_name)

        all_everion_means = []
        all_masimo_vitals = []
        for filename in os.listdir(dir_name):
            if filename.endswith("csv"):
                # Check with Susanne: are Everion data measured in UTC?
                df = self.loader.load_everion_patient_data(dir_name, filename, ";",
                                                           tz_to_zurich=True)

                # Alternative patient_id extraction: use a regular expression:
                # ret = re.match(".*([0-9]{3}).*", filename)
                # assert ret is not None, "Filename does not match pattern."
                # patient_id = ret.group(1)
                patient_id = filename[start_idx:end_idx]

                if not os.path.exists(out_dir_name):
                    os.mkdir(out_dir_name)
                out_file = os.path.join(out_dir_name, "%s_means.csv" % patient_id)

                df = df.set_index("timestamp")
                df = self.extract_cols(df)


                means = self.exam_times(df, vitals_masimo, patient_id)
                all_everion_means.append(means)
                means.to_csv(out_file)
                #self.comparison(means, vitals_masimo, patient_id, out_dir_name)

                masimo_data = vitals_masimo.loc[vitals_masimo["Record_id"] == patient_id]

                masimo_data = masimo_data.set_index("Date_Time")
                masimo_data = masimo_data[["HR", "SPo2", "objtemp", "BloodPressure", "RespRate"]]

                #self.plot_comparison(means, masimo_data, df, out_dir_name, patient_id)

                all_masimo_vitals.append(masimo_data)

        all_everion_means = pd.concat(all_everion_means)

        all_masimo_vitals = pd.concat(all_masimo_vitals)

        self.distribution(all_everion_means,all_masimo_vitals)
        #self.compare_all_patients(all_everion_means, all_masimo_vitals, out_dir_name)

    def distribution(self, data1, data2):
        everion_data = data1[["HR", "SPo2", "objtemp", "BloodPressure", "RespRate"]]
        masimo_data = data2[["HR", "SPo2", "objtemp", "BloodPressure", "RespRate"]]
        print(everion_data)
        print(masimo_data)

        diff = everion_data-masimo_data

        result = pd.DataFrame(index=diff.columns, columns=["Shapiro_Wilk", "Diff_Test", "Test_Type", "Sample_Size"])
        for vital in diff.columns:
            x = diff[vital]
            x = x.dropna()
            # Option1: x = x.values
            # Option2: x = np.asarrray(x)
            # Option3: x = x # Pandas object is like numpy array
            result.loc[vital, "Sample_Size"] = len(x)
            if len(x) < 7:
                result.loc[vital, "Shapiro_Wilk"] = None
            else:
                stat, p = stats.shapiro(x)
                result.loc[vital, "Shapiro_Wilk"] = p

        for vital in result.index:
            if pd.isna(result.loc[vital, "Shapiro_Wilk"]):
                result.loc[vital, "Diff_Test"] = None
                result.loc[vital, "Test_Type"] = None

            elif result.loc[vital, "Shapiro_Wilk"] <= 0.05:
                stat, p = self.wilcoxon_test(everion_data[vital], masimo_data[vital])
                result.loc[vital, "Diff_Test"] = p
                result.loc[vital, "Test_Type"] = "Wilcoxon"

            else:
                # stat, p = self.mannwhit_test(everion_data[vital], masimo_data[vital])
                # result.loc[vital, "Diff_Test"] = p
                # result.loc[vital, "Test_Type"] = "MannWhitU"

                stat, p = self.ttest(everion_data[vital], masimo_data[vital])
                result.loc[vital, "Diff_Test"] = p
                result.loc[vital, "Test_Type"] = "Ttest"


        print(result)

        result.to_csv('/Users/reys/Desktop/ACLS_Master/MasterThesis/DataMining_covid/UKBB/'
                      'data_short_header_mvse_with_comp_Ttest//ttest_results.csv')

    def ttest(self, data1, data2):
        ttest = stats.ttest_rel(data1, data2, nan_policy="omit")
        print(ttest)
        return ttest

    def mannwhit_test(self, data1, data2):
        mannwhittest = stats.mannwhitneyu(data1, data2)
        print(mannwhittest)
        return mannwhittest

    def wilcoxon_test(self, data1, data2):
        wilcoxon = stats.wilcoxon(data1, data2)
        print(wilcoxon)
        return wilcoxon

    def exam_times(self, df, vitals_masimo, patient_id):
        exam_times = vitals_masimo.loc[vitals_masimo["Record_id"] == patient_id, "Date_Time"]
        means = []
        for exam_time in exam_times:
            time_delta = datetime.timedelta(minutes=30)
            start_time = exam_time - time_delta
            stop_time = exam_time + time_delta
            mask = (df.index >= start_time) & (df.index <= stop_time)
            df_sub = df[mask]
            mean = self.mean_signals(df_sub)

            mean.name = exam_time
            means.append(mean)
        means = pd.concat(means, axis=1).T
        return means

    def extract_cols(self, df):
        df = df[["HR", "SPo2", "objtemp", "BloodPressure", "RespRate"]]
        return df

    def mean_signals(self, df_sub):
        # Now do what you have to do with this data (compute average, etc...)
        return df_sub.mean(axis=0)

    def md_sd(self, data1, data2):
        data1 = data1
        data2 = data2
        mean = (data1 + data2)/2

        diff = data1 - data2  # Difference between data1 and data2
        md = diff.mean(axis=0)  # Mean of the difference
        sd = diff.std(axis=0)  # Standard deviation of the difference
        return mean, diff, md, sd

    def bland_altman_plot(self, mean, diff, md, sd, data1, patient_id, out_dir_name, *args, **kwargs):
        for col in data1.columns:
            plt.figure()

            plt.scatter(mean[col], diff[col], *args, **kwargs)

            s = sd[col]
            m = md[col]
            xlim = plt.xlim()
            h_mean, = plt.plot(xlim, [m, m], "b", zorder=100)
            h_cip, = plt.plot(xlim, [m +1.96 * s, m +1.96 * s], ":r", zorder=100)
            h_cim, = plt.plot(xlim, [m -1.96 * s, m -1.96 * s], ":r", zorder=100)
            plt.xlabel('Mean')
            plt.ylabel('Difference')
            plt.title(col)

            my_file = (col + patient_id + '.jpeg')
            plt.savefig(os.path.join(out_dir_name, my_file))
            plt.close()

    def plot_all_patients(self, mean, diff, md, sd, data1, out_dir_name, *args, **kwargs):
        for col in data1.columns:
            plt.figure()

            plt.scatter(mean[col], diff[col], *args, **kwargs)

            s = sd[col]
            m = md[col]
            xlim = plt.xlim()
            h_mean, = plt.plot(xlim, [m, m], "b", zorder=100)
            h_cip, = plt.plot(xlim, [m +1.96 * s, m +1.96 * s], ":r", zorder=100)
            h_cim, = plt.plot(xlim, [m -1.96 * s, m -1.96 * s], ":r", zorder=100)
            plt.xlabel('Mean')
            plt.ylabel('Difference')
            plt.title(col)

            my_file = (col + '.jpeg')
            plt.savefig(os.path.join(out_dir_name, my_file))
            plt.close()

    def plot_comparison(self, everion_data, masimo_data, df, out_dir_name, patient_id):

        fig, ax = plt.subplots(5, 1, figsize=(12, 12))

        time_delta = datetime.timedelta(days=1)

        xmin1 = everion_data.index.min()
        xmin2 = masimo_data.index.min()
        xmin3 = df.index.min()

        xmax1 = everion_data.index.max()
        xmax2 = masimo_data.index.max()
        xmax3 = df.index.max()

        xmin = min(xmin1, xmin2, xmin3) - time_delta
        xmax = max(xmax1, xmax2, xmax3) + time_delta

        for i, col in enumerate(everion_data.columns):
            plt.figure()

            sns.lineplot(x=df.index, y=col, ax=ax[i], data=df, color="orange", alpha=0.5)
            sns.lineplot(x=everion_data.index, y=col, ax=ax[i], data=everion_data)
            sns.scatterplot(ax=ax[i], data=masimo_data, x=masimo_data.index, y=col, color="red", ci=None)

            ax[i].set_xlim([xmin, xmax])
            fig.suptitle("Available data of patient " + patient_id + ":" + " Everion (blue) vs. Masimo (red)")
            fig.tight_layout()
            fig.subplots_adjust(top=0.95)

            #plt.show()
            my_file = ("comparison_" + patient_id + '.jpeg')
            fig.savefig(os.path.join(out_dir_name, my_file))
            plt.clf()
            plt.close(fig)


    def comparison(self, everion_data, vitals_masimo, patient_id, out_dir_name):
        #compare everion means of each patient with masimo data and plot
        masimo_file = vitals_masimo.loc[vitals_masimo["Record_id"] == patient_id]
        masimo_file = masimo_file.set_index("Date_Time")
        masimo_file = masimo_file[["HR", "SPo2", "objtemp", "BloodPressure", "RespRate"]]

        mean, diff, md, sd = self.md_sd(everion_data, masimo_file)
        self.bland_altman_plot(mean, diff, md, sd, everion_data, patient_id, out_dir_name)

    def compare_all_patients(self, all_everion_means, all_masimo_vitals, out_dir_name):
        # compare everion means with masimo data and plot
        mean, diff, md, sd = self.md_sd(all_everion_means, all_masimo_vitals)

        # self.student_test(all_everion_means, all_masimo_vitals)
        self.plot_all_patients(mean, diff, md, sd, all_everion_means, out_dir_name)

    def student_test(self, data1, data2):
        data1 = data1
        data2 = data2

        mean_everion = data1.mean(axis=0)
        mean_masimo = data2.mean(axis=0)

        cols = data1.columns

        for col in cols:
            x1 = data1[col]
            x2 = data2[col]

            x1m = mean_everion[col]
            x2m = mean_masimo[col]

            n1 = len(x1)
            n2 = len(x2)

            s2 = (sum((x1 - x1m)**2) + sum((x2 - x2m)**2)) / (n1 + n2 -2)
            t = (x1m - x2m) / sqrt(s2 *((1/n1)+(1/n2)))
            print(t)


if __name__ == "__main__":
    dir_name = '/Users/reys/Desktop/ACLS_Master/MasterThesis/DataMining_covid/UKBB/data_short_header/'
    dir_nurse = '/Users/reys/Desktop/ACLS_Master/MasterThesis/Ines_Daten/22.12.20/Vitals_Phoenix_included_patients_update_Dec_2020.csv'
    comparison = DataComparator()
    comparison.load_files(dir_name=dir_name, dir_nurse=dir_nurse,
                          start_idx=0, end_idx=3)
