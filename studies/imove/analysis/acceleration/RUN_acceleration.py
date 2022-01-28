"""RUN_acceleration """
# Working directory (angegeben in upper right) must be located in
# /imove/analysis/acceleration, sost geht import context.py nicht!     

# LIBRARIES ----------------------------------------------------------------------------
import os
from pathlib import Path
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Import own modules
import context # it can oly import context.py when contained in the same folder as demmi.py
from demmi_ex_dict import demmi_ex                                                          
from acceleration import put_margins_around_ex, resample, align_timestamp
from fourier import fourier_transform           

# Modules by Susanne Suter
#from src.mhealth.utils.commons import print_title
#from src.mhealth.utils.context_info import dump_context
from mhealth.utils.plotter_helper import save_figure, setup_plotting

# PATHS ----------------------------------------------------------------------------

path_data = '/Users/julien/GD/ACLS/TM/DATA/'
#path_src = '/Users/julien/My Drive/20_STUDIUM/ACLS/05 Module/TM/mhealth/src'
#path_output = '/Users/julien/My Drive/20_STUDIUM/ACLS/05 Module/TM/OUTPUT'
#plots_path = '/Users/julien/My Drive/20_STUDIUM/ACLS/05 MODULE/TM/OUTPUT/plots/'

# LOAD DEMMI (acc) DATA ----------------------------------------------------------------------------

load = 'subset' # load demorton_pat001_pat002_pat006.h5. 3 Pat: 001, 002, 006, left&right.
# load = 'all'  # load demorton.h5 (all data)

if load == 'subset':
    filepath = '/Users/julien/GD/ACLS/TM/DATA/extracted/quality50_clipped_collected/store/demorton_pat001_pat002_pat006.h5'
    store = pd.HDFStore(filepath, mode='r')
    acc = store["raw"]
    acc = acc.reset_index() # create new index -> timestamp becomes a normal col
    # acc.info()  # 10 columns
else:
    # Load demorton.pickle (=demorton.h5). takes ca 10 sec.
    filepath = Path(path_data, 'pickle/demorton.pickle')
    with open(filepath, 'rb') as f:
        acc = pickle.load(f)

# LOAD exercises.csv ----------------------------------------------------------------------------

## 1 ## /quality50_clipped/exercises.csv
# # StartDate, EndDate for each combination of Patient, Day, Task
# filepath = '/Users/julien/GD/ACLS/TM/DATA/extracted/quality50_clipped/exercises.csv'
# exercises = pd.read_csv(filepath)
# exercises.Task.unique()

# EXECUTION ----------------------------------------------------------------------------

# Put margins of 'delta_seconds'  before & after all specified exercises.
input_df = acc
exercises = ['2a', '5a','12','15']
delta_seconds = 10
df_margins = put_margins_around_ex(df=input_df, demmi_ex=exercises, delta_seconds=delta_seconds)

# resample 51Hz to 1sec
enable = False
# df_resample = resample(df_margins, enable=True) # enable=False
df_resample = resample(df_margins, enable=enable)

# align all acc-curves with common starting time = 0
df_aligned = align_timestamp(df=df_resample)

# renaming
df = df_aligned

# FOURIER TRANSFORM (for all ex) ----------------------------------------------------------------------------
df_input = df_aligned # resample(df, enable=False) BEFORE, as no resampling needed!
pat = '006'
day = '1'
side = 'right'

for ex in exercises:
    xf, yf = fourier_transform(df=df_input, ex=ex, pat=pat, day=day, side=side)
    ex_text = demmi_ex[ex]
    plt.plot(xf, np.abs(yf))
    plt.title(f'FFT for Patient {pat}, Day {day}, Side {side} \n Ex {ex}: {ex_text} \n resample={enable}', fontsize=10)
    plt.show()

# PLOTS ----------------------------------------------------------------------------

## A_plot_demmi) 1 specific exercise. 3 days.
for ex in exercises:
    ex_text = demmi_ex[ex]
    
    plot = sns.relplot(
            data=df[df.DeMortonLabel.eq(ex)],  # subset: only specific ex
            x="time", y="A",
            col="Side",  row='DeMortonDay',
            hue="Patient", # style="event",
            kind="line"
        )
    plot.fig.suptitle(f'DEMMI Ex. {ex} \n {ex_text} \n margins={delta_seconds} sec. resample={enable}', fontsize=25) # title
    ax = plt.gca()
    xticks = ax.get_xticks()
    xticks = [pd.to_datetime(tm, unit="ms").strftime('%Y-%m-%d\n %H:%M:%S')
              for tm in xticks]
    ax.set_xticklabels(xticks, rotation=45)
    plt.tight_layout()


## B)  For all days separately, All exercises.
# days = ['1', '2', '3']
# for day in days:
#     plot = sns.relplot(
#             data=df[df.DeMortonDay.eq(day)],  # subset: only specific day
#             x="time", y="A",
#             col="Side",  row='DeMortonLabel',
#             hue="Patient", # style="event",
#             kind="line"
#         )
#     plot.fig.suptitle(f'Day: {day}', fontsize=40) # title
#     ax = plt.gca()
#     xticks = ax.get_xticks()
#     xticks = ax.get_xticks()
#     xticks = [pd.to_datetime(tm, unit="ms").strftime('%Y-%m-%d\n %H:%M:%S')
#               for tm in xticks]
#     ax.set_xticklabels(xticks, rotation=45)
    
#     plt.show()

# FEATURE DEVELOPMENT ----------------------------------------------------------------------------


                