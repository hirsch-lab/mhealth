# LIBRARIES ----------------------------------------------------------------------------
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import timedelta

# Import Own modules
import context # it can oly import context.py when contained in the same folder as demmi.py
from demmi_ex_dict import demmi_ex, demmi_ex_short
from acceleration import put_margins_around_ex, resample, align_timestamp
from feature_dev import feature_development
from RUN_1_acceleration import EXERCISES, PATH_DATA, load_borg
from paths import path_output




#### FUNCTIONS ----------------------------------------------------------------------------
def get_deviation_of_ideal_bmi(df): # 
    """From col 'bmi' calculate new col 'ideal_bmi_deviation'.
    """
    bmi_mean = df['BMI'].mean()
    df["ideal_bmi_deviation"] = (df['BMI'] - bmi_mean).abs()
    return df
    
# dict: patient_ID_sex_dict: 'Patient' : 'sex'
borg = load_borg(path_data=PATH_DATA) # borg (i.e. from borg_bmi_age.csv). 60 rows.
patient_ID_sex_dict = borg["sex"].to_dict()

#### Prepare scores_agg ----------------------------------------------------------------------------
def main():
    
    # LOAD scores_disagg (contains all 60 pat)
    scores_disagg = pd.read_csv('scores_disagg.csv')  
    
    # groupby & aggregate
    counts = scores_disagg.groupby(["Exercise", "Patient"])["BMI"].count() # enthält keine col 'Exertion'
    df = scores_disagg.groupby(["Exercise", "Patient"]).mean()
    
    # counts
    df = df.drop('DeMortonDay', 1) # drop col DeMortonDay, bc makes no sense
    df["counts"] = counts # Shows how many counts were aggregated to calculate mean for each group (eg 4 obs).
    df = df.reset_index() # ev nötig damit pairplot geht.. Muss hue-variable factor sein?
    
    # Exercise_number
    df.rename(columns={'Exercise':'Exercise_number'}, inplace=True)
    df['Exercise'] = df['Exercise_number'].map(demmi_ex_short)
    
    # Patient
    df['Patient'] = df['Patient'].astype(str)
    df.loc[:,'Patient'] = df['Patient'].str.rjust(3, "0") # add trailing zeros
    
    # add col 'sex' from patient_ID_sex_dict
    df['sex'] =  df['Patient'].map(patient_ID_sex_dict)

    # new col: 'ideal_bmi_deviation' with get_deviation_of_ideal_bmi()
    g = df.groupby("sex") # different bmi-means for male and female
    df = g.apply(get_deviation_of_ideal_bmi) # added new col ideal_bmi_deviation
    
    # save
    scores_agg = df # rename
    scores_agg.to_csv('scores_agg.csv', index = False) # export as csv. do not include index in csv 
    scores_agg.info()
if __name__ == "__main__":
    main()

#### ANALYSIS CONTINUES IN JUPYTERLAB WITH FILE: feature_correlation.py ----------------------------------------------------------------------------





