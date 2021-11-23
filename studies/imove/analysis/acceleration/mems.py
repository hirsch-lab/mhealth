"""This py is my initial acceleration data analysis file. """


# LIBRARIES ----------------------------------------------------------------------------
import os
import sys
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

import context

from src.mhealth.utils.commons import print_title # starts from the pwd
from src.mhealth.utils.context_info import dump_context
from mhealth.utils.plotter_helper import save_figure, setup_plotting

# PATHS ----------------------------------------------------------------------------
# Make sure that pwd is at: TM/mhealth/
os.chdir('/Users/JULIEN/Google Drive/20_STUDIUM/ACLS/Module/TM/mhealth') # set wd
wd = os.getcwd()

path_src = '/Users/JULIEN/Google Drive/20_STUDIUM/ACLS/Module/TM/mhealth/src'
path_output = '/Users/JULIEN/Google Drive/20_STUDIUM/ACLS/Module/TM/OUTPUT'


# LOAD DATA ----------------------------------------------------------------------------

filepath = '/Users/JULIEN/GD/ACLS/TM/DATA/quality50_clipped_collected/store/ex-1.h5'
store = pd.HDFStore(filepath, mode='r')
print(store.keys())
df = store["raw"]
#df = store["vital"]
df.head

# ----------------------------------------------------------------------------



sys.path
os.environ
os.environ["HOME"] 



