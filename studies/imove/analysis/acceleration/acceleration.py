# LIBRARIES ----------------------------------------------------------------------------
import pandas as pd
import numpy as np
from datetime import timedelta

# Import Own modules
import context # it can oly import context.py when contained in the same folder as demmi.py
from demmi_ex_dict import demmi_ex

# Modules by Susanne Suter
from mhealth.utils.plotter_helper import save_figure, setup_plotting

#### Margins around exercices ----------------------------------------------------------------------------

def put_margins_around_ex(df, demmi_ex=demmi_ex, delta_seconds=15):
    """Puts margins of length 'delta_seconds' before and after every specified 
    demmi_ex (eg ['12', '15']). Groupings for exercises, days, side, patients are considered. """
    
    # subset rows of Ex 12 and 15 
    mask = df["DeMortonLabel"].isin(demmi_ex)
    df_f = df[mask].copy()
    
    delta = timedelta(seconds=delta_seconds)
    
    # groupby: 'DeMortonLabel', 'DeMortonDay', 'Side', 'Patient'
    g = df_f.groupby(['DeMortonLabel', 'DeMortonDay', 'Side', 'Patient'])
    starts = g["timestamp"].min() - delta
    stops  = g["timestamp"].max() + delta
    
    # Extract rows for each group between lower and upper margin
    mask = pd.Series(index=df.index, dtype=bool)
    mask[:] = False  # Initialize all False
    for t0, t1 in zip(starts, stops):
        # For each iteration, add its 'new' rows to existing rows with OR
        mask |= ((df["timestamp"]>=t0) & (df["timestamp"]<=t1))
    df = df[mask].copy()  # Return a copy instead of a view.
    
    # set DeMortonLabel 'default' to NA
    df.loc[df['DeMortonLabel'] == 'default', 'DeMortonLabel'] = np.nan


    # Set timestamp as Datetimeindex
    df = df.set_index('timestamp')
        
    # Stats for resulting df
    print('Contained exercises: ', df.DeMortonLabel.unique())
    print('Contained Days: ', df.DeMortonDay.unique())
    print('Contained sides: ', df.Side.unique())
    print('Contained patients: ', df.Patient.unique())
    
    return df


#### RESAMPLE ----------------------------------------------------------------------------

def resample(df, rule="1s", enable=True):
    """Resample per specified time unit, for each grouping. 
    Concatenate all resampled groups."""
    if enable:
        aggregations = {
                         'A': 'mean',
                        'AX': 'mean',
                        'AY': 'mean',
                        'AZ': 'mean',
               'DeMortonDay': 'first',
                     'Side' : 'first',
                  'Patient' : 'first',
            'DeMortonLabel' : 'first'
            }
        g = df.groupby(["Patient", "Side"]) # groupby each sensor
        df_ret = pd.DataFrame()
        for gid, df_sub in g:
            df = df_sub.resample(rule).agg(aggregations)
            df_ret = df_ret.append(df)
        # Move "timestamp", "Patient", "Side" back as columns.
        df_ret = df_ret.reset_index()
        # Strangely, g.resample(rule).agg(aggregations), which should give
        # about the same result, is much much slower...
    else:
        df_ret = df
        df_ret = df_ret.reset_index()
              
    return df_ret

#### ALIGN ----------------------------------------------------------------------------

def align_timestamp(df):
    """Align timestamps of all subgroups so that all acc-curves can be superimposed on 
    each other."""
    def compute_time(df):
        df["time"] = df["timestamp"] - df["timestamp"].min()
        return df
    g = df.groupby(['Patient', 'DeMortonLabel', 'DeMortonDay', 'Side'])
    return g.apply(compute_time)







