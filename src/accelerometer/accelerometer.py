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
# PREPARE DATA
# =============================================================================
def load_accelerometer(dir_name, file_name, tz_to_zurich=True):
    """Load accelerometer sensor data (csv) from raw_labeled_X and subset by De Morton Exercises. 
       Outputs: df, morton """
    ## Load csv
    csv_in_file = os.path.join(dir_name, file_name)
    df = pd.read_csv(csv_in_file, sep=';') # , **kwargs
    
    # check data
    if 1 not in df.de_morton.unique():
        print(f"{file_name} contains no 1 in col: de_morton, thus cannot process it.")
        df = None
        morton = None
        return df, morton # return empty df, morton.
    
    # df.info()
    # df.isna().sum() # get NaN of each col -> timestamp etc have NaNs
    
    ## Prepare df
    df = df.loc[: , ["AX", "AY", "AZ", "timestamp", "de_morton_label", "de_morton"]]
    df = df[df['timestamp'].notnull()] # remove timestamp which are NaN # gibt Warnung
    
    # convert to datetime64
    if tz_to_zurich:
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert('Europe/Zurich')
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert('UTC')
        
    df['date'] = df.timestamp.dt.date # extract Datum
    df['hour'] = df.timestamp.dt.hour # extract hour
    df['minute'] = df.timestamp.dt.minute.map("{:02}".format) # extract minute (and keep leading zero)
    df["hh_mm"] = df["hour"].map(str) + ":" + df["minute"].map(str) 
    # df.info()
    
    ## 1) subset: only de_morton == 1
    morton = df.loc[df["de_morton"] == 1 , :]


    return df, morton


def create_accelerometer(df, morton, path_output, outname, time_offset=15):
    """Extract uebungsserie for each date, subset accelerometer for 
    15mins timedelta around uebungsserie. Convert wide -> long format. Aggregate
    by datetime & dimension.
    
    Save as .csv to /output. 
    Output: acc (df).
    """
    acc_wide = pd.DataFrame()
    
    # extract uebungsserie of each date
    for date in morton.date.unique():
        sub = morton[morton["date"]==date]
    
        # add 15' timedeltas
        start = sub.timestamp.min() - timedelta(minutes=time_offset)
        end   = sub.timestamp.max() + timedelta(minutes=time_offset)
        
        sel = (df.timestamp >= start) & (df.timestamp <= end) # Boolean Selector
        acc = df[sel]
        # acc.info()
        # acc.timestamp.min()
        # acc.timestamp.max()

        acc_wide = pd.concat([acc_wide, acc], axis=0) # stack dfs
        
        
    # wide -> long format
    acc_long = pd.melt(acc_wide, id_vars=['timestamp', 'date', 'hour', 'minute', 'hh_mm', 'de_morton_label', 'de_morton'], # vars to keep
                       var_name='dimension', value_name='acc')

    # clean de_morton_label: remove everything after dot. eg: 15.0 -> 15
    acc_long['de_morton_label'] = acc_long['de_morton_label'].str.split('.').str[0]

    # AGGREGATE BY SECOND AND CALCULATE MEANS    
    acc_long_agg = acc_long.groupby(["timestamp", "dimension"], 
                                      as_index=False).agg({'acc':'mean',
                                                          'date':'first', # all other cols: take first one
                                                          'hour':'first',
                                                          'minute':'first',
                                                          'hh_mm':'first',
                                                          'de_morton':'first', # not all in group have the same!
                                                          'de_morton_label':'first' # about all in group have the same
                                                          })

    # index in new col                                                        
    acc_long_agg["ind"] = acc_long_agg.index
    
    # rename                                                           
    acc = acc_long_agg                                                           

    # save acc as csv into /output
    acc.to_csv(Path(path_output)/outname , ';', index=False)
    
    return acc



