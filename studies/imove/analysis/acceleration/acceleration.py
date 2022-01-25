# LIBRARIES ----------------------------------------------------------------------------
import pandas as pd
from datetime import timedelta

# Working directory (angegeben in upper right) must be located in
# /imove/analysis/acceleration, sonst geht import context.py nicht!

# Import Own modules
import context # it can oly import context.py when contained in the same folder as demmi.py
from demmi_ex_dict import demmi_ex

#from src.mhealth.utils.commons import print_title
#from src.mhealth.utils.context_info import dump_context
from mhealth.utils.plotter_helper import save_figure, setup_plotting

# LOAD DATA ----------------------------------------------------------------------------

## /quality50_clipped/
# exercises (): StartDate, EndDate for each combination of Patient, Day, Task
filepath = '/Users/norman/workspace/education/phd/data/wearables/studies/usb-imove/extracted/quality50_clipped/exercises.csv'
exercises = pd.read_csv(filepath)
exercises.Task.unique()

# store/001.h5
filepath = '/Users/norman/workspace/education/phd/data/wearables/studies/usb-imove/extracted/quality50_clipped/store/001.h5'
store = pd.HDFStore(filepath, mode='r')
store.keys()
h5_001 = store["/exercises"]
h5_001 = store["/vital/left"]
h5_001 = store["/vital/right"]
h5_001 = store["/raw/left"]
h5_001 = store["/raw/right"]


# demorton_pat001_pat002_pat006.h5 : to develop analysis code
# 3 Pat: 001, 002, 006, left%right, 
filepath = '/Users/norman/workspace/education/phd/data/wearables/studies/usb-imove/extracted/quality50_clipped_collected/store/demorton_pat001_pat002_pat006.h5'
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
    Groupings for exercises, days, side, patients are considered. """
    
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
        g = df.groupby(["Patient", "Side"])
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


