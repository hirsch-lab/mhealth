# =============================================================================
# LIBRARY
# =============================================================================
import os
import sys
import pandas as pd
from datetime import datetime
from datetime import timedelta
from pathlib import Path
# from natsort import natsort

sys.path.insert(1,'/Users/JULIEN/Google Drive/20_STUDIUM/ACLS/Module/TM1/imove/mhealth/src')

# =============================================================================
# LOAD DATA
# =============================================================================
def load_accelerometer(dir_name, filename):
    """Load accelerometer sensor data (csv) from raw_labeled_X and subset by De Morton Exercises. 
       Outputs: df, morton """
    ## Load csv
    csv_in_file = os.path.join(dir_name, filename)
    df1 = pd.read_csv(csv_in_file, sep=';') # , **kwargs
    # df1.info()
    df1.isna().sum() # get NaN of each col -> timestamp etc have NaNs
    
    ## Prepare df
    df2 = df1.loc[: , ["AX", "AY", "AZ", "timestamp", "de_morton_label", "de_morton"]]
    df = df2[df2['timestamp'].notnull()] # remove timestamp which are NaN # gibt Warnung
    df['date_time'] = pd.to_datetime(df['timestamp']) # convert to datetime64. gibt Warnung
    # df.drop('timestamp', inplace=True, axis=1) # drop col timestamp. Warnung
    df['date'] = df.date_time.dt.date # extract Datum
    # df.info()
    
    ## 1) subset: only de_morton == 1
    morton = df.loc[df["de_morton"] == 1 , :]
    morton.info()

    return df, morton


def create_accelerometer(df, morton, outname):
    """Extract uebungsserie for each date, subset accelerometer for 
    15mins timedelta around uebungsserie. Save as .csv to /output. 
    Output: accelerometer (df).
    """
    accelerometer = pd.DataFrame()
    
    # extract uebungsserie of each date
    for date in morton.date.unique():
        sub = morton[morton["date"]==date]
    
        # add 15' timedeltas
        start = sub.date_time.min() - timedelta(minutes=15)
        end   = sub.date_time.max() + timedelta(minutes=15)
        
        sel = (df.date_time >= start) & (df.date_time <= end) # Boolean Selector
        acc = df[sel]
        # acc.info()
        # acc.date_time.min()
        # acc.date_time.max()

        accelerometer = pd.concat([accelerometer, acc], axis=0) # stack dfs

    # save accelometer as csv into /output
    path_output = '/Users/JULIEN/GD/ACLS/TM1/OUTPUT/accelerometer'
    accelerometer.to_csv(Path(path_output)/outname , ';')
    
    return accelerometer



    


