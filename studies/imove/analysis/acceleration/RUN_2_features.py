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

#### a ----------------------------------------------------------------------------
# Load scores_ALL_ex
scores_ALL_ex = pd.read_csv('scores_ALL_ex.csv')  
scores_ALL_ex = scores_ALL_ex.set_index(["Exercise", "Patient"])

#### a ----------------------------------------------------------------------------


