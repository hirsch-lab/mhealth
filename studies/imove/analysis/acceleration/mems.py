"""This py is my initial acceleration data analysis file. """

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

import context # it can import context.py because it is contained in the same folder
# as mems.py

#from src.mhealth.utils.commons import print_title
#from src.mhealth.utils.context_info import dump_context
from mhealth.utils.plotter_helper import save_figure, setup_plotting

# PATHS ----------------------------------------------------------------------------
# Make sure that pwd is at: TM/mhealth/ ( SOLLTE GAR NICHT NÖTIG SEIN)
# os.chdir('/Users/julien/My Drive/20_STUDIUM/ACLS/05 Module/TM/mhealth') # set wd
wd = os.getcwd()

path_src = '/Users/julien/My Drive/20_STUDIUM/ACLS/05 Module/TM/mhealth/src'
path_output = '/Users/julien/My Drive/20_STUDIUM/ACLS/05 Module/TM/OUTPUT'


# LOAD DATA ----------------------------------------------------------------------------

filepath = '/Users/julien/GD/ACLS/TM/DATA/extracted/quality50_clipped_collected/store/ex-1.h5'
store = pd.HDFStore(filepath, mode='r')
print(store.keys())
df = store["raw"]
#df = store["vital"]
df.head

# ----------------------------------------------------------------------------







