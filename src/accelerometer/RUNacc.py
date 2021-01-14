# =============================================================================
# 1: LIBRARIES 
# =============================================================================
import os
import sys
import pandas as pd
from datetime import datetime
from datetime import timedelta
from pathlib import Path
# from natsort import natsort

sys.path.insert(1,'/Users/JULIEN/Google Drive/20_STUDIUM/ACLS/Module/TM1/imove/mhealth/src')
sys.path.insert(2,'/Users/JULIEN/Google Drive/20_STUDIUM/ACLS/Module/TM1/imove/mhealth/src/accelerometer')
sys.path

# Import own modules
from accelerometer import load_accelerometer, create_accelerometer

# Define folder and filename
dir_name = '/Users/JULIEN/GD/ACLS/TM1/DATA/imove_data/raw_labeled_0-10' # acc sensor data

# Process all Accelerometer sensor files: XXXX_storage-vital_raw.csv
for file_name in os.listdir(dir_name)[0:2]:    
    ID = file_name[0:4] # file ID
    print(f"Processing: {file_name} ..." )

    # RUN imported modules
    df, morton = load_accelerometer(dir_name, file_name)
    accelerometer = create_accelerometer(df=df, morton=morton, outname=f"{ID}_acc_during_morton.csv", time_offset=15)


## acc_imported.info() # date_time wird nach import wieder object statt datetime64

# Process just on file
# ID = '001L'
# df, morton = load_accelerometer(dir_name, '001L_storage-vital_raw.csv')
# accelerometer = create_accelerometer(df=df, morton=morton, outname=f"{ID}_acc_during_morton.csv", time_offset=15)





