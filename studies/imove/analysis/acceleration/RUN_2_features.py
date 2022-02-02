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
from RUN_1_acceleration import exercises

# FEATURE DEVELOPMENT (scores) ----------------------------------------------------------------------------

# only for specific exercise
# scores = feature_development(df=acc, ex='12') # input is 'acc' (either subset or all)

scores_ALL_ex = pd.DataFrame()
for ex  in exercises:
    scores_per_Ex = feature_development(df=acc, ex=ex) # input is 'acc' (either subset or all)
    scores_ALL_ex = scores_ALL_ex.append(scores_per_Ex)
scores_ALL_ex.to_csv('scores_ALL_ex.csv') # export as csv


a = feature_development(df=acc, ex='12')


#### a ----------------------------------------------------------------------------


# Load scores_ALL_ex
scores_ALL_ex = pd.read_csv('scores_ALL_ex.csv')
scores_ALL_ex = scores_ALL_ex.set_index(["Patient", "Exercise"])

#### a ----------------------------------------------------------------------------


