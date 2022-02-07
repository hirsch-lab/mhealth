"""RUN_3_produce_acc """

# LIBRARIES ----------------------------------------------------------------------------
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import own modules
from RUN_1_acceleration import PATH_DATA, format_data, load_data

# DEFINE PARAMETERS ----------------------------------------------------------------------------
EXERCISES = ['2a', '5a','12','15'] 
LOAD = 'all'  # load demorton.h5 (all data)
MARGIN_SECONDS = 0     # Margins per exercise in seconds
ENABLE_RESAMPLE = True  # Enable resampling (from 51HZ to 1Hz)


# a ----------------------------------------------------------------------------
def main():
    
    df_raw  = load_data(path_data=PATH_DATA, mode=LOAD) # all pat
    df = format_data(df=df_raw,  
                     exercises=EXERCISES,               # all ex
                     margin_seconds=MARGIN_SECONDS      # margin_seconds = 0
                                                        # resample = True
                     )
    # save as csv
    df.to_csv('acc_allEx_allPat_margin0_resampleTrue.csv', index=False)



if __name__ == "__main__":
    main()






