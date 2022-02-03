"""RUN_acceleration """
# Working directory (angegeben in upper right) must be located in
# /imove/analysis/acceleration, sost geht import context.py nicht!     

# LIBRARIES ----------------------------------------------------------------------------
import os
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import own modules
from demmi_ex_dict import demmi_ex
from acceleration import put_margins_around_ex, resample, align_timestamp
from fourier import fourier_transform
from feature_dev import feature_development


# DEFINE PARAMETERS ----------------------------------------------------------------------------
EXERCISES = ['2a', '5a','12','15']
LOAD = 'subset' # load demorton_pat001_pat002_pat006.h5. 3 Pat: 001, 002, 006, left&right.
# load = 'all'  # load demorton.h5 (all data)

MARGIN_SECONDS = 10     # Margins per exercise in seconds
ENABLE_RESAMPLE = True  # Enable resampling (from 51HZ to 1Hz)


PATH_DATA = Path('/Users/julien/GD/ACLS/TM/DATA/')
PATH_DATA = Path('/Users/norman/workspace/education/phd/data/wearables/studies/usb-imove')


# LOAD DEMMI (acc) DATA ----------------------------------------------------------------------------
def load_data(path_data, mode):
    if mode == 'subset':
        filepath = path_data / 'extracted/quality50_clipped_collected/store/demorton_pat001_pat002_pat006.h5'
        store = pd.HDFStore(filepath, mode='r')
        df = store["raw"]
        df = df.reset_index() # create new index -> timestamp becomes a normal col
        # df.info()  # 10 columns
        del store
    else:
        # Load demorton.pickle (=demorton.h5). takes ca 10 sec.
        filepath = path_data / 'pickle/demorton.pickle'
        with open(filepath, 'rb') as f:
            df = pickle.load(f)
    return df


def load_borg(path_data):
    filepath = Path(path_data, 'extracted/quality50_clipped_collected/borg_bmi_age.csv')
    df = pd.read_csv(filepath, sep=";")
    df = df[["Patient", "sex", "age", "weight", "BMI"]]
    df.loc[:,'Patient'] = df['Patient'].apply(str) # int -> str
    df.loc[:,'Patient'] = df['Patient'].str.rjust(3, "0") # add leading zeros
    df = df.set_index('Patient')
    df.index.name = "patient_ID"
    return df


def format_data(df, exercises, margin_seconds):
    df_margins = put_margins_around_ex(df=df, demmi_ex=exercises,
                                       delta_seconds=margin_seconds)

    # Resample from 51Hz to 1Hz
    df_resample = resample(df_margins, enable=ENABLE_RESAMPLE)

    # Align all acc-curves with common starting time = 0. [index64 -> index64]
    df_aligned = align_timestamp(df=df_resample)

    return df_aligned


def plot_amplitude_spectra_per_exercise(df, exercises, pat, day, side):
    for ex in exercises:
        xf, yf = fourier_transform(df=df, ex=ex, pat=pat, day=day, side=side)
        ex_text = demmi_ex[ex]
        plt.plot(xf, np.abs(yf))
        title = (f'FFT for Patient {pat}, Day {day}, Side {side} \n'
                 f'Ex {ex}: {ex_text} \n resample={ENABLE_RESAMPLE}')
        plt.title(title, fontsize=10)


def plot_exercises(df, exercises):
    """A_plot_demmi) 1 specific exercise. 3 days."""
    for ex in exercises:
        ex_text = demmi_ex[ex]

        plot = sns.relplot(
                data=df[df.DeMortonLabel.eq(ex)],  # subset: only specific ex
                x="time", y="A",
                col="Side",  row='DeMortonDay',
                hue="Patient", # style="event",
                kind="line"
            )
        title = (f'DEMMI Ex. {ex} \n {ex_text} \n'
                 f'margin={MARGIN_SECONDS} sec. resample={ENABLE_RESAMPLE}')
        plot.fig.suptitle(title, fontsize=25) # title
        ax = plt.gca()
        xticks = ax.get_xticks()
        ax.set_xticks(xticks)  # This avoids a warning in set_xticklabels
        xticks = [pd.to_datetime(tm, unit="ms").strftime('%Y-%m-%d\n %H:%M:%S')
                  for tm in xticks]
        ax.set_xticklabels(xticks, rotation=45)
        plt.tight_layout()


def plot_exercises_per_day(df, exercises, days):
    """For all days separately, All exercises."""
    for day in days:
        plot = sns.relplot(
                data=df[df.DeMortonDay.eq(day)],  # subset: only specific day
                x="time", y="A",
                col="Side",  row='DeMortonLabel',
                hue="Patient", # style="event",
                kind="line"
            )
        plot.fig.suptitle(f'Day: {day}', fontsize=40) # title
        ax = plt.gca()
        xticks = ax.get_xticks()
        ax.set_xticks(xticks)  # This avoids a warning in set_xticklabels
        xticks = [pd.to_datetime(tm, unit="ms").strftime('%Y-%m-%d\n %H:%M:%S')
                  for tm in xticks]
        ax.set_xticklabels(xticks, rotation=45)
        plt.tight_layout()


def generate_feature_scores(df, exercises):
    """Generate features dataframe of all Exercises and Patients.
    Save to csv.
    """
    scores_ALL_ex = pd.DataFrame()
    for ex in exercises:
        scores_per_Ex = feature_development(df=df, ex=ex)
        scores_ALL_ex = scores_ALL_ex.append(scores_per_Ex)
    scores_ALL_ex.to_csv('scores_ALL_ex.csv') # export as csv


def main():

    # Put margins of 'delta_seconds'  before & after all specified exercises. [RangeIndex -> DateTimeIndex]
    df_borg = load_borg(path_data=PATH_DATA)
    df_raw = load_data(path_data=PATH_DATA, mode=LOAD)
    df = format_data(df=df_raw, exercises=EXERCISES,
                     margin_seconds=MARGIN_SECONDS)

    # pat = '006'
    # day = '1'
    # side = 'right'
    # plot_amplitude_spectra_per_exercise(df=df_raw, exercises=EXERCISES,
    #                                     pat=pat, day=day, side=side)
    # plt.show()

    plot_exercises(df=df, exercises=EXERCISES)
    plt.show()

    # days = ['1', '2', '3']
    # plot_exercises_per_day(df=df, exercises=EXERCISES, days=days)
    # plt.show()

    # generate_feature_scores(df=df_raw, df_borg=df_borg,
    #                         exercises=EXERCISES)


if __name__ == "__main__":
    main()






