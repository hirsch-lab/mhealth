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


#### a ----------------------------------------------------------------------------

# Compute some score
def score_std(df):
    return df["A"].std()

def score_kinetic_energy(df, masses):
    # Compute kinetic energy given acceleration and mass
    # https://en.wikipedia.org/wiki/Kinetic_energy
    # Argument masses: a lookup patient -> body mass
    # I thought this information is available somewhere...
    pat = df["Patient"].iloc[0]
    mass = masses[pat]
    ...
def score_spectrum(df):
    # ...
    
# Example: Compute a score for exercise 12
# data = df.loc[df["DeMortonLabel"]=="12"]               
# g = data.groupby(["Patient", "DeMortonDay", "Side"])

# scores_std = g.apply(score_std)   # Same as g["A"].std()
# scores_std.name = "Standard deviation"
# scores_kin = g.apply(score_kinetic_energy, masses)
# scores_kin.name = "Kinetic energy"
# scores_spect = g.apply(score_spectrum)
# scores_spect.name = "Characteristic frequency"

# more scores...
# scores_all = pd.concat([scores_std, scores_kin, ...], axis=1)

# Compute means over days and sides
# scores = scores_all.groupby(["Patient"]).mean()
