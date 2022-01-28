# Save demorton.h5 as demorton.pickle. For faster loading afterwards.

# LIBRARIES ----------------------------------------------------------------------------
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import timedelta
import pickle

# Import Own modules
import context # it can oly import context.py when contained in the same folder as demmi.py
from demmi_ex_dict import demmi_ex
from acceleration import put_margins_around_ex, resample, align_timestamp

path_data = '/Users/julien/GD/ACLS/TM/DATA/'

# PICKLE ----------------------------------------------------------------------------

# Load demorton.h5 as acc_all (takes about 1 min)
filepath = '/Users/julien/GD/ACLS/TM/DATA/extracted/quality50_clipped_collected/store/demorton.h5'
store = pd.HDFStore(filepath, mode='r')
acc_all = store["raw"]
acc_all = acc_all.reset_index() # create new index -> timestamp becomes a normal col   

## Save acc_all object as demorton.pickle (takes about 1 min)
filepath = Path(path_data, 'pickle/demorton.pickle')
with open(filepath, 'wb') as f:
    pickle.dump(acc_all, f)    
