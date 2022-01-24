"""  """
#sys.path
#os.environ
#os.environ["HOME"] 

# LIBRARIES ----------------------------------------------------------------------------
import os
import sys
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from pandasql import sqldf
from datetime import timedelta
from collections import defaultdict

# Working directory (angegeben in upper right) must be located in
# /imove/analysis/acceleration, sost geht import context.py nicht!

# Import Own modules
import context # it can oly import context.py when contained in the same folder as demmi.py
from demmi_ex_dict import demmi_ex

#from src.mhealth.utils.commons import print_title
#from src.mhealth.utils.context_info import dump_context
from mhealth.utils.plotter_helper import save_figure, setup_plotting

# LOAD DATA ----------------------------------------------------------------------------

## /quality50_clipped/
# exercises (): StartDate, EndDate for each combination of Patient, Day, Task
filepath = '/Users/julien/GD/ACLS/TM/DATA/extracted/quality50_clipped/exercises.csv'
exercises = pd.read_csv(filepath)
exercises.Task.unique()

# store/001.h5
filepath = '/Users/julien/GD/ACLS/TM/DATA/extracted/quality50_clipped/store/001.h5'
store = pd.HDFStore(filepath, mode='r')
store.keys()
h5_001 = store["/exercises"]
h5_001 = store["/vital/left"]
h5_001 = store["/vital/right"]
h5_001 = store["/raw/left"]
h5_001 = store["/raw/right"]


# demorton_pat001_pat002_pat006.h5 : to develop analysis code
# 3 Pat: 001, 002, 006, left%right, 
filepath = '/Users/julien/GD/ACLS/TM/DATA/extracted/quality50_clipped_collected/store/demorton_pat001_pat002_pat006.h5'
# filepath = '/Users/julien/GD/ACLS/TM/DATA/extracted/quality50_clipped_collected/store/demorton.h5'

store = pd.HDFStore(filepath, mode='r')
acc = store["raw"]
acc = acc.reset_index() # create new index -> timestamp becomes a normal col

# acc_all # data of demorton.h5


#### Margins around exercices ----------------------------------------------------------------------------

# demmi_ex = ['12'] # demmi exercises of interest: ex-12 (Gehen 50m), ex-15 (Springen)
# demmi_ex = ['12', '15'] 

delta_seconds = 15

def put_margins_around_ex(df=acc, demmi_ex=demmi_ex, delta_seconds=delta_seconds):
    """Puts margins of length 'delta_seconds' before and after every specified DEMMI ex.
    Groupings for exercices, days, Side, patients are considered. """
    
    # subset rows of Ex 12 and 15 
    mask = df["DeMortonLabel"].isin(demmi_ex)
    df_f = df[mask].copy()
    
    delta = timedelta(seconds=delta_seconds)
    
    # groupby: 'DeMortonLabel', 'DeMortonDay', 'Side', 'Patient'
    g = df_f.groupby(['DeMortonLabel', 'DeMortonDay', 'Side', 'Patient'])
    for key, item in g:
        print(key, "\n\n") # Print groups' keys: ('001', '3') :: ('PAT', 'Day')
    
    starts = g["timestamp"].min() - delta
    stops  = g["timestamp"].max() + delta
    
    # Extract rows for each group between lower and upper margin
    mask = pd.Series(index=df.index, dtype=bool)
    mask[:] = False # initiate empty BOOLEAN
    for t0, t1 in zip(starts, stops):
        mask |= ((df["timestamp"]>=t0) & (df["timestamp"]<=t1)) # for each iteration, add its 'new' rows to existing rows with OR
    df = df[mask].copy() # wenn sich Werte verändern ist .copy() sicherer
    
    # set timestamp as Datetimeindex
    df = df.set_index('timestamp')
        
    # stats for resulting df
    print('Contained exercises: ', df.DeMortonLabel.unique())
    print('Contained Days: ', df.DeMortonDay.unique())
    print('Contained sides: ', df.Side.unique())
    print('Contained patients: ', df.Patient.unique())
    
    return df


#### RESAMPLE ----------------------------------------------------------------------------

def resample(df, enable=True): 
    """Resample per specified time unit, for each grouping. 
    Concatenate all resampled groups. Weiss nicht, wie das ohne for loop
    gehen würde. """
    # groupby: 'DeMortonLabel', 'DeMortonDay', 'Side', 'Patient'
    g = df.groupby(['DeMortonDay', 'Side', 'Patient', 'DeMortonLabel'])

    # resample
    if enable:
        df_concat = pd.DataFrame()
        
        for gid, df_sub in g:
            df = df_sub.resample("1s").agg({
                                        'A': 'mean', 
                                       'AX': 'mean', 
                                       'AY': 'mean', 
                                       'AZ': 'mean',
                              'DeMortonDay': 'first', 
                                    'Side' : 'first',
                                 'Patient' : 'first',
                           'DeMortonLabel' : 'first'
                                     })
            df_concat = df_concat.append(df)
            
        # The index 'timestamp' is now no more unique. Must have a unique index 
        # for plotting. 
        df_concat = df_concat.reset_index() # timestamp becomes a normal col
        
    else:
        df_concat = df
              
    return df_concat

#### ALIGN ----------------------------------------------------------------------------

def align_timestamp(df):
    """Align timestamps of all subgroups so that all acc-curves can be superimposed on 
    each other."""
    # groupby: 'DeMortonDay', 'Patient'. (only 1 ex contained)
    g = df.groupby(['DeMortonDay', 'Side', 'Patient', 'DeMortonLabel'])
    
    df_concat = pd.DataFrame() # empty df
    
    for gid, df_sub in g:    
        #skipper to omit 'default' group. FUNKTIONIERT DAS WIE ANGEDACHT??
        if 'default' in gid: # eg: ('3', 'left', '006', 'default')
            print('gid with "default" excluded')
            pass
        else:
            df_sub["time"] = df_sub["timestamp"] - df_sub["timestamp"].min()
            df_concat = df_concat.append(df_sub)
        
    return df_concat
    

#### a ----------------------------------------------------------------------------


# g = df_resample.groupby(['DeMortonDay', 'Side', 'Patient', 'DeMortonLabel'])
# for gid, df_sub in g:      
#     if 'default' in gid:
#         print('gid including "default" detected and excluded')
#         pass
#     else:
#         print(gid)


