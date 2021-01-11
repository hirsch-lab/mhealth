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
filename = '001L_storage-vital_raw.csv' # acc sensor file

# RUN imported modules
df, morton = load_accelerometer(dir_name, filename)
accelerometer = create_accelerometer(df=df, morton=morton, outname='001L_accelometer.csv')



## Load prepared acc 
# dir_name = '/Users/JULIEN/GD/ACLS/TM1/DATA/imove_data/raw_labeled_0-10'
# filename = '001L_accelometer.csv'
# csv_in_file = os.path.join(Path('./output'), filename)
# acc_imported = pd.read_csv(csv_in_file, sep=';') # , **kwargs
## acc_imported.info() # date_time wird nach import wieder object statt datetime64
