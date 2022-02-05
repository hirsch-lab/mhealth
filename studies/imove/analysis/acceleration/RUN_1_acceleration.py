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
# Variablen mit globalem Scope sind in CAPITAL
EXERCISES = ['2a', '5a','12','15'] 
# LOAD = 'subset' # load demorton_pat001_pat002_pat006.h5. 3 Pat: 001, 002, 006, left&right.
LOAD = 'all'  # load demorton.h5 (all data)

MARGIN_SECONDS = 10     # Margins per exercise in seconds
ENABLE_RESAMPLE = True  # Enable resampling (from 51HZ to 1Hz)
# METHOD = '2' (fuse) is active # 'fuse' fuses sensors 'left' and 'right' when calling score_kinetic_energy()


PATH_DATA = Path('/Users/julien/GD/ACLS/TM/DATA/') # Julien
# PATH_DATA = Path('/Users/norman/workspace/education/phd/data/wearables/studies/usb-imove') # Norman


# LOAD DATA ----------------------------------------------------------------------------
def load_data(path_data, mode):
    """Load acc data. Either demorton.h5 or a subset."""
    if mode == 'subset':
        filepath = Path(path_data , 'extracted/quality50_clipped_collected/store/demorton_pat001_pat002_pat006.h5') # Julien
        # filepath = path_data / 'extracted/quality50_clipped_collected/store/demorton_pat001_pat002_pat006.h5' # Norman
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
    # filepath = Path(path_data, 'extracted/quality50_clipped_collected/borg_bmi_age.csv') # Norman
    filepath = Path(path_data, 'Borg/borg_bmi_age.csv') # Julien

    df = pd.read_csv(filepath, sep=";") # borg_bmi_age.csv is ; separated
    df = df[["Patient", "sex", "age", "weight", "BMI"]]
    df.loc[:,'Patient'] = df['Patient'].apply(str) # int -> str
    df.loc[:,'Patient'] = df['Patient'].str.rjust(3, "0") # add leading zeros
    df = df.set_index('Patient')
    df.index.name = "patient_ID" # same col name as in demorton.h5
    return df

# FUNCTIONS ----------------------------------------------------------------------------
def format_data(df, exercises, margin_seconds):
    """Execute: put_margins_around_ex(), resample(), align_timestamp(). Used
    for visualization only."""
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
        plt.figure() # nötig, damit für jede Ex eine plot generiert wird. (falls es welche Problem gibt, ev hier outcommenten.)
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


def generate_feature_scores(df, df_borg, exercises): # input df: df_raw
    """Generate features dataframe of all Exercises and Patients, with calling
    feature_development(). Save as scores_ALL_ex.csv, to be loaded later in RUN_2_features.py.
    """
    scores_ALL_ex = pd.DataFrame()
    for ex in exercises:
        scores_per_Ex = feature_development(df=df, df_borg=df_borg, ex=ex) # input df: df_raw
        scores_ALL_ex = scores_ALL_ex.append(scores_per_Ex)
    scores_ALL_ex.to_csv('scores_ALL_ex.csv') # export as csv


def main():

    # Put margins of 'delta_seconds'  before & after all specified exercises. [RangeIndex -> DateTimeIndex]
    df_borg = load_borg(path_data=PATH_DATA)
    df_raw  = load_data(path_data=PATH_DATA, mode=LOAD) # load acc 
    df = format_data(df=df_raw, exercises=EXERCISES, # for viz
                     margin_seconds=MARGIN_SECONDS)

    # Plot ----------------------------------
    pat = '006'
    day = '1'
    side = 'right'
    plot_amplitude_spectra_per_exercise(df=df_raw, exercises=EXERCISES, # input: df_raw: demorton.h5
                                        pat=pat, day=day, side=side)
    plt.show()

    #  Plot ----------------------------------
    plot_exercises(df=df, exercises=EXERCISES)
    plt.show()

    # Plot (less importanat) ----------------------------------
    # days = ['1', '2', '3']
    # plot_exercises_per_day(df=df, exercises=EXERCISES, days=days)
    # plt.show()

    generate_feature_scores(df=df_raw, df_borg=df_borg,
                            exercises=EXERCISES)


if __name__ == "__main__":
    main()






