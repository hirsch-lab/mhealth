import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

import context
from mhealth.utils.plotter_helper import save_figure, setup_plotting


def dataset_id(df):
    # Format for identifiers: "[0-9]{3}[LR]"
    # For example:            "018L" or "042R"
    pat = df["Patient"].map("{0:03d}".format)
    side = df["Side"].str[0].str.upper()
    return pat + side


def read_summary(path):
    df = pd.read_csv(path, header=[0,1], index_col=[0,1,2])
    df[("Time","Start")] = pd.to_datetime(df[("Time","Start")])
    df[("Time","End")] = pd.to_datetime(df[("Time","End")])
    return df


def read_exercises(path):
    df = pd.read_csv(path)
    df["StartDate"] = pd.to_datetime(df["StartDate"])
    df["Duration"] = pd.to_timedelta(df["Duration"])
    df["EndDate"] = pd.to_datetime(df["EndDate"])
    return df


def plot_qualities(df_before, df_after, out_dir):
    qualities = ["HRQ", "SpO2Q", "QualityClassification"]
    df_before = df_before.loc[:,pd.IndexSlice[qualities, "mean"]].copy()
    df_before["Label"] = "before"
    df_after = df_after.loc[:,pd.IndexSlice[qualities, "mean"]].copy()
    df_after["Label"] = "after"
    df = pd.concat([df_before, df_after], axis=0)
    df = df.droplevel(1, axis=1).reset_index()
    df = df.melt(id_vars=["Side", "Label"], value_vars=qualities,
                 var_name="Quality", value_name="Value")

    fig, ax = plt.subplots()
    hbox = sns.boxplot(x="Quality", y="Value", hue="Label",
                       data=df, palette=["m", "g"], ax=ax)
    hstrip = sns.stripplot(x="Quality", y="Value", hue="Label", dodge=True,
                           data=df, size=4, color="gray",
                           linewidth=0, alpha=0.4, ax=ax)
    hbox.legend_.set_title(None)
    handles, labels = ax.get_legend_handles_labels()
    n = df["Label"].nunique()
    l = plt.legend(handles[0:n], labels[0:n],
                   bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.grid(axis="y")
    plt.tight_layout()
    save_figure(out_dir/"qualities.pdf")


def plot_time_recorded(df, exercises, out_path, ncols=20):

    # Format dataframe
    df = df.loc[:,"Time"].copy()
    df = df.reset_index()
    df["Dataset"] = dataset_id(df=df)
    df["Row"] = ((df["Patient"].astype(int)-1)/ncols).astype(int)
    nrows = df["Row"].max()+1
    df = df.set_index(["Patient", "Side"])

    # Start dates for each patient and day
    # delta: hours elapsed since first recording (StartDate)
    exercises = exercises.groupby(["Patient", "Day"])["StartDate"].min()
    exercises = (pd.concat([exercises, exercises], axis=0,
                           keys=["left", "right"], names=["Side"])
                 .reset_index(level="Day"))
    # Requires index: ["Patient", "Side"]
    delta = exercises["StartDate"]-df["Start"]
    delta = delta.dt.total_seconds()/3600
    exercises = exercises.reset_index()
    exercises["Dataset"] = dataset_id(df=exercises)
    exercises.loc[:, "DeMortonStart"] = delta.values

    ylim = 80 # df["TotalHours"].max() * 1.05
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12,7))
    for i, dfrow in df.groupby("Row"):

        dfrow = dfrow.melt(id_vars=["Dataset"],
                           value_vars=["TotalHours", "ValidHours"],
                           var_name="Measurement", value_name="Hours")
        ax = sns.barplot(x="Dataset", y="Hours", hue="Measurement",
                         data=dfrow, ax=axes[i], palette=["lightgray", "b"],
                         dodge=False)
        dfex = exercises.loc[exercises["Dataset"].isin(dfrow["Dataset"])]
        if False:
            # Plot the start times of the 3 De Morton excercise sessions
            ax = sns.stripplot(x="Dataset", y="DeMortonStart", data=dfex,
                               ax=axes[i], facecolor="k", color="k", size=4,
                               linewidth=1, marker="_", jitter=False)
        ax.set_ylim([0, ylim])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.legend(bbox_to_anchor=(1.02, 1),
                  loc=2, borderaxespad=0.).set_visible(i==0)
        ax.set_xlabel(None)
        ax.grid(axis="y")

    fig.subplots_adjust(hspace=0.5)
    fig.tight_layout()
    save_figure(out_path, fig=fig)


def plot_time_gaps(df_before, df_after, out_dir):
    def _format_data(df, kind):
        total = df["Time"].loc[:,"TotalHours"]
        gaps = df.loc[:,"TimeGaps"].copy()
        box_data = pd.concat([total, gaps["MaxGap"]], axis=1)
        box_data["FilterStatus"] = kind
        gaps = gaps.drop("MaxGap", axis=1)
        gaps_rel = gaps.div(total, axis=0)
        gaps["FilterStatus"] = kind
        gaps_rel["FilterStatus"] = kind
        return box_data, gaps, gaps_rel

    def _plot_max_gap(data, filepath):
        fig, ax = plt.subplots()
        hbox = sns.boxplot(x="Measure", y="Hours", hue="FilterStatus",
                           data=data, palette=["m", "g"], ax=ax)
        hstrip = sns.stripplot(x="Measure", y="Hours", hue="FilterStatus",
                               data=data, dodge=True, size=4, color="gray",
                               linewidth=0, alpha=0.4, ax=ax)
        n = data["FilterStatus"].nunique()
        handles, labels = ax.get_legend_handles_labels()
        l = plt.legend(handles[0:n], labels[0:n], title="Filter Status")
        ax.grid(axis="y")
        ax.set_xlabel(None)
        save_figure(filepath, fig=fig)

    def _plot_gap_counts(data, filepath):
        fig, ax = plt.subplots()
        hbox = sns.boxplot(x="Measure", y="Counts", hue="FilterStatus",
                           data=data, palette=["m", "g"], ax=ax)
        hstrip = sns.stripplot(x="Measure", y="Counts", hue="FilterStatus",
                               data=data, dodge=True, size=4, color="gray",
                               linewidth=0, alpha=0.4, ax=ax)
        n = data["FilterStatus"].nunique()
        handles, labels = ax.get_legend_handles_labels()
        l = plt.legend(handles[0:n], labels[0:n], title="Filter Status")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_xlabel(None)
        ax.grid(axis="y")
        plt.tight_layout()
        save_figure(filepath, fig=fig)


    box_data_b, gaps_b, gaps_brel = _format_data(df_before, kind="before")
    box_data_a, gaps_a, gaps_arel = _format_data(df_after, kind="after")
    box_data = pd.concat([box_data_b, box_data_a], axis=0)
    gaps_data = pd.concat([gaps_b, gaps_a], axis=0)
    gaps_data_rel = pd.concat([gaps_brel, gaps_arel], axis=0)
    box_data = box_data.melt(id_vars=["FilterStatus"],
                             value_vars=["TotalHours", "MaxGap"],
                             var_name="Measure", value_name="Hours")
    gaps_data = gaps_data.melt(id_vars=["FilterStatus"],
                               var_name="Measure", value_name="Counts")
    gaps_data_rel = gaps_data_rel.melt(id_vars=["FilterStatus"],
                                       var_name="Measure", value_name="Counts")
    _plot_max_gap(data=box_data, filepath=out_dir/"max_gaps.pdf")
    _plot_gap_counts(data=gaps_data, filepath=out_dir/"gap_counts_full.pdf")
    _plot_gap_counts(data=gaps_data_rel, filepath=out_dir/"gap_counts_rel_full.pdf")
    gaps_data = gaps_data[gaps_data["Measure"]!="nGaps>1m"]
    gaps_data_rel = gaps_data_rel[gaps_data_rel["Measure"]!="nGaps>1m"]
    _plot_gap_counts(data=gaps_data, filepath=out_dir/"gap_counts.pdf")
    _plot_gap_counts(data=gaps_data_rel, filepath=out_dir/"gap_counts_rel.pdf")



def run(data_dir, out_dir):
    path_before = data_dir / "summary_vital_original.csv"
    path_after = data_dir / "summary_vital_original_filtered.csv"
    df_before = read_summary(path_before)
    df_after = read_summary(path_after)
    exercises = read_exercises(data_dir / "exercises.csv")

    setup_plotting()
    print("Plotting qualities...")
    plot_qualities(df_before=df_before, df_after=df_after, out_dir=out_dir)
    print("Plotting time recorded...")
    plot_time_recorded(df=df_before, exercises=exercises,
                       out_path=out_dir/"time_recorded_unfiltered.pdf")
    plot_time_recorded(df=df_after, exercises=exercises,
                       out_path=out_dir/"time_recorded_filtered.pdf")
    print("Plotting time gaps...")
    plot_time_gaps(df_before=df_before, df_after=df_after, out_dir=out_dir)
    print("Done!")
    #plt.show()


if __name__ == "__main__":
    data_dir = Path("../results/extraction")
    out_dir = Path("../results/analysis/everion_data")
    run(data_dir=data_dir, out_dir=out_dir)
