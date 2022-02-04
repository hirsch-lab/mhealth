# LIBRARIES ----------------------------------------------------------------------------
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import timedelta

# Import Own modules
import context # it can oly import context.py when contained in the same folder as demmi.py
from demmi_ex_dict import demmi_ex
from acceleration import put_margins_around_ex, resample, align_timestamp
from feature_dev import feature_development
from RUN_1_acceleration import EXERCISES

#### a ----------------------------------------------------------------------------
# Load scores_ALL_ex
scores_ALL_ex = pd.read_csv('scores_ALL_ex.csv')  
counts = scores_ALL_ex.groupby(["Exercise", "Patient"])["BMI"].count() # enthält keine col 'Exertion'
df = scores_ALL_ex.groupby(["Exercise", "Patient"]).mean()
df["counts"] = counts # Aggregated by counts (eg 4 obs) per group.

df = df.reset_index() # ev nötig damit pairplot geht.. Muss hue-variable factor sein?
df.info()


#### Correlation ----------------------------------------------------------------------------

import seaborn as sns
sns.set_theme(style="ticks")
sns.pairplot(df, hue="Exercise")
sns.pairplot(df, hue="Patient")

