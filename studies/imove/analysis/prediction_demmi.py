import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from mhealth.utils.file_helper import ensure_dir


#################################################################################################################

class DeMorton():

    def load_files(self, dir_name):

        dir_name = Path(dir_name)

        # file_path = os.path.join(dir_name, 'vital.csv')
        # morton_vital = pd.read_csv(file_path, sep=";", parse_dates=["timestamp"])

        # file_path = os.path.join(dir_name, 'raw.csv')
        # morton_raw = pd.read_csv(file_path, sep=";", parse_dates=["timestamp"])

        print("Reading vital data...")
        file_path = dir_name / "demorton-vital.csv"
        morton_vital = pd.read_csv(file_path, sep=",",
                                   parse_dates=["timestamp"],
                                   dtype={"DeMortonLabel": str})

        morton_vital = morton_vital.rename({"DeMortonDay": "Day",
                                            "DeMortonLabel": "Task"}, axis=1)
        morton_vital = morton_vital.set_index("timestamp")

        print("Reading raw data...")
        file_path = dir_name / "demorton-raw.csv"
        morton_raw = pd.read_csv(file_path, sep=",",
                                 parse_dates=["timestamp"],
                                 dtype={"DeMortonLabel": str})
        morton_raw = morton_raw.rename({"DeMortonDay": "Day",
                                        "DeMortonLabel": "Task"}, axis=1)
        morton_raw = morton_raw.set_index("timestamp")

        #morton_raw = None

        return morton_vital, morton_raw

#################################################################################################################

    def energy_acc(self, x):
        x = x - x.mean()
        # if False:
        #     # Low-pass filter. Adjust window size
        #     x = x.rolling(window=5, center=True).mean()
        t = x.index.to_series()
        mx = (x + x.shift()) / 2
        dt = (t - t.shift()).dt.total_seconds()
        energy = np.sqrt((mx * mx * dt).sum())
        return energy

    def min_max_mean_acc(self, df):
        ret = {}
        ret["Acc-max"] = df.groupby(["Day", "Side"])["A"].max().mean()
        ret["Acc-min"] = df.groupby(["Day", "Side"])["A"].min().mean()
        ret["Acc-min-quantile"] = df.groupby(["Day", "Side"])["A"].quantile(q=0.05).mean()
        ret["Acc-max-quantile"] = df.groupby(["Day", "Side"])["A"].quantile(q=0.95).mean()
        ret["Acc-sum"] = df.groupby(["Day", "Side"])["A"].sum().mean()
        ret["Acc-var"] = df.groupby(["Day", "Side"])["A"].var().mean()
        ret["Acc-energy"] = df.groupby(["Side"])["A"].apply(morton_corr.energy_acc).mean()

        return pd.Series(ret)

#################################################################################################################

    def min_max_mean(self, df):
        ret = {}

        ret["HR-min"] = df.groupby(["Day", "Side"])["HR"].min().mean()
        ret["HR-max"] = df.groupby(["Day", "Side"])["HR"].max().mean()
        ret["HR-min-quantile"] = df.groupby(["Day", "Side"])["HR"].quantile(q=0.05).mean()
        ret["HR-max-quantile"] = df.groupby(["Day", "Side"])["HR"].quantile(q=0.95).mean()
        ret["HRV-min"] = df.groupby(["Day", "Side"])["HRV"].min().mean()
        ret["HRV-max"] = df.groupby(["Day", "Side"])["HRV"].max().mean()
        ret["HRV-min-quantile"] = df.groupby(["Day", "Side"])["HRV"].quantile(q=0.05).mean()
        ret["HRV-max-quantile"] = df.groupby(["Day", "Side"])["HRV"].quantile(q=0.95).mean()
        ret["HRV-var"] = df.groupby(["Day", "Side"])["HRV"].var().mean()
        ret["SpO2-min"] = df.groupby(["Day", "Side"])["SpO2"].min().mean()
        ret["SpO2-max"] = df.groupby(["Day", "Side"])["SpO2"].max().mean()
        ret["SpO2-min-quantile"] = df.groupby(["Day", "Side"])["SpO2"].quantile(q=0.05).mean()
        ret["SpO2-max-quantile"] = df.groupby(["Day", "Side"])["SpO2"].quantile(q=0.95).mean()

        # ret["delta-time"] = (df.groupby(["DeMortonDay", "Side"])["timestamp"].max()
        #                      - df.groupby(["DeMortonDay", "Side"])["timestamp"].min()).dt.total_seconds()

        return pd.Series(ret)

#################################################################################################################

    def correlations(self, features_exercise):
        correlation_bmi = features_exercise.corrwith(target["BMI"])
        correlation_age = features_exercise.corrwith(target["age"])
        print(correlation_bmi)
        print(correlation_age)

        borg_cor = {}
        for patient, day in features_exercise.iterrows():

            pat_id = patient[0]
            day = patient[1]

            if day == 1.0:
                correlation_borg1 = features_exercise.corrwith(target['MeanBorg1'])
                borg_cor[(pat_id, day)] = correlation_borg1
            elif day == 2.0:
                correlation_borg2 = features_exercise.corrwith(target['MeanBorg2'])
                borg_cor[(pat_id, day)] = correlation_borg2
            else:
                correlation_borg3 = features_exercise.corrwith(target['MeanBorg3'])
                borg_cor[(pat_id, day)] = correlation_borg3

        target.columns.get_loc('MeanBorg1')

        return correlation_bmi, correlation_age, pd.DataFrame.from_dict(borg_cor)

#################################################################################################################

    def plot_correlations(self, df):
        if not df.empty:
            morton_cor = df.corr()

            self.save_plots(morton_cor)

    def save_plots(self, morton_cor):
        fig = plt.figure(figsize=(14, 10))
        mask = np.zeros_like(morton_cor)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(morton_cor, cmap='Spectral_r', annot=True, mask=mask, vmin=-1, vmax=1)

        plt.show()
        #plt.savefig(out_dir / "correlation_features.jpg", bbox_inches='tight')
        plt.close(fig)

#################################################################################################################

if __name__ == "__main__":
    in_dir = '/Users/reys/Desktop/ACLS_Master/MasterThesis/DataMining_covid/iMove/extracted/'

    in_dir = Path(in_dir)
    out_dir = Path("output/demmi_analysis")
    ensure_dir(out_dir)
    feat_path = out_dir / "features.csv"
    if feat_path.is_file():
        print("Lazy loading aggregated data...")
        features = pd.read_csv(feat_path, index_col=["Patient", "Task", "Day"], sep=",")

    else:
        morton_corr = DeMorton()
        sensor_dir = in_dir / "quality50_clipped_collected/csv/"
        morton_vital, morton_raw = morton_corr.load_files(dir_name=sensor_dir)

        print("Aggregating vital data...")
        features_vital = morton_vital.groupby(["Patient", "Task", "Day"]).apply(morton_corr.min_max_mean)

        print("Aggregating raw data...")
        features_raw = morton_raw.groupby(["Patient", "Task", "Day"]).apply(morton_corr.min_max_mean_acc)
        features = pd.concat([features_vital, features_raw], axis=1)
        features.to_csv(out_dir / "features.csv")

    print("Reading time measurements...")
    exercises_path = in_dir / "quality50_clipped/exercises.csv"
    exercises = pd.read_csv(exercises_path,
                            parse_dates=["StartDate", "EndDate"],
                            infer_datetime_format=True,
                            index_col=["Patient", "Task", "Day"],
                            dtype={"Day": float})
    exercises["Duration"] = pd.to_timedelta(exercises["Duration"]).dt.total_seconds()
    mask = ~exercises.index.get_level_values("Task").isin(["temp", "default"])
    exercises = exercises[mask]
    features["Duration"] = exercises["Duration"]

    print("Reading target data...")
    target_path = in_dir / "quality50_clipped_collected/borg_bmi_age.csv"
    target = pd.read_csv(target_path, sep=";", index_col=["Patient"])
    print(target)

    feat = []

    print("Computing features...")

    # ex 6: sit to stand
    # ex 12: walk independently
    # ex 15: jump

    ### Task Duration ex6
    mask_dur = features.index.get_level_values("Task") == "6"
    time_ex6 = features.loc[mask_dur, "Duration"]
    time_ex6 = time_ex6.droplevel("Task")
    time_ex6.name = "Duration ex6"
    feat.append(time_ex6)
    print(time_ex6)

    ### Task Duration ex12
    mask_dur = features.index.get_level_values("Task") == "12"
    time_ex12 = features.loc[mask_dur, "Duration"]
    time_ex12 = time_ex12.droplevel("Task")
    time_ex12.name = "Duration ex12"
    feat.append(time_ex12)
    print(time_ex12)

    ### Task Duration ex15
    mask_dur2 = features.index.get_level_values("Task") == "15"
    time_ex15 = features.loc[mask_dur2, "Duration"]
    time_ex15 = time_ex15.droplevel("Task")
    time_ex15.name = "Duration ex15"
    feat.append(time_ex15)
    print(time_ex15)

    ### Normalize calculated feature to experiment duration
    mask_norm = features.index.get_level_values("Task") == "12"
    acc_energy_norm12 = features["Acc-energy"] / features["Duration"]
    acc_energy_norm12 = acc_energy_norm12.loc[mask_norm]
    acc_energy_norm12 = acc_energy_norm12.droplevel("Task")
    acc_energy_norm12.name = "Acc norm to ex12 duration"
    feat.append(acc_energy_norm12)
    print(acc_energy_norm12)

    ### Normalize calculated feature to experiment duration
    mask_norm2 = features.index.get_level_values("Task") == "15"
    acc_energy_norm15 = features["Acc-energy"] / features["Duration"]
    acc_energy_norm15 = acc_energy_norm15.loc[mask_norm2]
    acc_energy_norm15 = acc_energy_norm15.droplevel("Task")
    acc_energy_norm15.name = "Acc norm to ex15 duration"
    feat.append(acc_energy_norm15)
    print(acc_energy_norm15)

    ### HR diff (quantile) of exercise 6
    mask = features.index.get_level_values("Task") == "6"
    hr_diff_q_ex6 = features.loc[mask, "HR-max-quantile"] - features.loc[mask, "HR-min-quantile"]
    hr_diff_q_ex6 = hr_diff_q_ex6.droplevel("Task")
    hr_diff_q_ex6.name = "HR quantile difference ex6"
    feat.append(hr_diff_q_ex6)
    print(hr_diff_q_ex6)

    ### HR diff (quantile) of exercise 12
    mask2 = features.index.get_level_values("Task") == "12"
    hr_diff_q_ex12 = features.loc[mask2, "HR-max-quantile"] - features.loc[mask2, "HR-min-quantile"]
    hr_diff_q_ex12 = hr_diff_q_ex12.droplevel("Task")
    hr_diff_q_ex12.name = "HR quantile difference ex12"
    feat.append(hr_diff_q_ex12)
    print(hr_diff_q_ex12)

    ### HR diff (quantile) of exercise 15
    mask3 = features.index.get_level_values("Task") == "15"
    hr_diff_q_ex15 = features.loc[mask3, "HR-max-quantile"] - features.loc[mask3, "HR-min-quantile"]
    hr_diff_q_ex15 = hr_diff_q_ex15.droplevel("Task")
    hr_diff_q_ex15.name = "HR quantile difference ex15"
    feat.append(hr_diff_q_ex15)
    print(hr_diff_q_ex15)

    ### ACC diff (quantile) of exercise 12
    mask_acc_q2 = features.index.get_level_values("Task") == "12"
    acc_diff_q_ex12 = features.loc[mask_acc_q2, "Acc-max-quantile"] - features.loc[mask_acc_q2, "Acc-min-quantile"]
    acc_diff_q_ex12 = acc_diff_q_ex12.droplevel("Task")
    acc_diff_q_ex12.name = "Acceleration quantile difference ex12"
    feat.append(acc_diff_q_ex12)
    print(acc_diff_q_ex12)

    ### ACC diff (quantile) of exercise 15
    mask_acc_q3 = features.index.get_level_values("Task") == "15"
    acc_diff_q_ex15 = features.loc[mask_acc_q3, "Acc-max-quantile"] - features.loc[mask_acc_q3, "Acc-min-quantile"]
    acc_diff_q_ex15 = acc_diff_q_ex15.droplevel("Task")
    acc_diff_q_ex15.name = "Acceleration quantile difference ex15"
    feat.append(acc_diff_q_ex15)
    print(acc_diff_q_ex15)

    ### Sum of ACC of exercise 12
    mask_acc_sum2 = features.index.get_level_values("Task") == "12"
    acc_sum_ex12 = features.loc[mask_acc_sum2, "Acc-sum"]
    acc_sum_ex12 = acc_sum_ex12.droplevel("Task")
    acc_sum_ex12.name = "Sum of acceleration ex12"
    feat.append(acc_sum_ex12)
    print(acc_sum_ex12)

    ### ACC variance of exercise 12
    mask_acc_va2 = features.index.get_level_values("Task") == "12"
    acc_var_ex12 = features.loc[mask_acc_va2, "Acc-var"]
    acc_var_ex12 = acc_var_ex12.droplevel("Task")
    acc_var_ex12.name = "Acceleration variance ex12"
    feat.append(acc_var_ex12)
    print(acc_var_ex12)

    # ### Normalize calculated feature to experiment duration (ex12)
    # mask_norm_acc = features.index.get_level_values("Task") == "12"
    # acc_var_norm12 = features["Acc-var"] / features["Duration"]
    # acc_var_norm12 = acc_var_norm12.loc[mask_norm_acc]
    # acc_var_norm12 = acc_var_norm12.droplevel("Task")
    # acc_var_norm12.name = "Acc-var norm to ex12 duration"
    # feat.append(acc_var_norm12)
    # print(acc_var_norm12)

    ### ACC variance of exercise 15
    mask_acc_var3 = features.index.get_level_values("Task") == "15"
    acc_var_ex15 = features.loc[mask_acc_var3, "Acc-var"]
    acc_var_ex15 = acc_var_ex15.droplevel("Task")
    acc_var_ex15.name = "Acceleration variance ex15"
    feat.append(acc_var_ex15)
    print(acc_var_ex15)

    # ### Normalize calculated feature to experiment duration (ex15)
    # mask_norm_acc2 = features.index.get_level_values("Task") == "15"
    # acc_var_norm15 = features["Acc-var"] / features["Duration"]
    # acc_var_norm15 = acc_var_norm15.loc[mask_norm_acc2]
    # acc_var_norm15 = acc_var_norm15.droplevel("Task")
    # acc_var_norm15.name = "Acc-var norm to ex15 duration"
    # feat.append(acc_var_norm15)
    # print(acc_var_norm15)

    features_exercise = pd.concat(feat, axis =1)

    print(features_exercise)
    features_exercise.to_csv(out_dir / "features_exercise.csv")

    print("Calculate correlations")
    morton_corr = DeMorton()
    #morton_corr.plot_correlations(features_exercise)
    correlation_bmi, correlation_age, borg_cor = morton_corr.correlations(features_exercise)
    print(borg_cor)
    correlation_bmi.to_csv(out_dir / "bmi_features.csv")
    correlation_age.to_csv(out_dir / "age_features.csv")
    borg_cor.to_csv(out_dir / "borg_features.csv")
    #morton_corr.save_plots(borg_cor)



#################################################################################################################

