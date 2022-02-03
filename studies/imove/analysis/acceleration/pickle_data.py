# Save demorton.h5 as demorton.pickle. For faster loading afterwards.

# LIBRARIES ----------------------------------------------------------------------------
from pathlib import Path
import pandas as pd
import pickle

import context
from mhealth.utils.file_helper import ensure_dir

# PARAMETERS ----------------------------------------------------------------------------
PATH_DATA = Path('/Users/julien/GD/ACLS/TM/DATA/')

# PICKLE ----------------------------------------------------------------------------

# Load demorton.h5 as acc_all (takes about 1 min)
filepath = PATH_DATA / 'extracted/quality50_clipped_collected/store/demorton.h5'
store = pd.HDFStore(filepath, mode='r')
acc_all = store["raw"]
acc_all = acc_all.reset_index() # create new index -> timestamp becomes a normal col   

## Save acc_all object as demorton.pickle (takes about 1 min)
filepath = Path(path_data, 'pickle/demorton.pickle')
ensure_dir(filepath.parent)
with open(filepath, 'wb') as f:
    pickle.dump(acc_all, f)    
