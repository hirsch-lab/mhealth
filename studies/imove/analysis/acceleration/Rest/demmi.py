"""DEMMI analysis """

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

# Working directory (angegeben in upper right) must be located in
# /imove/analysis/acceleration, sost geht import context.py nicht!

import context # it can oly import context.py when contained in the same folder
# as demmi.py

#from src.mhealth.utils.commons import print_title
#from src.mhealth.utils.context_info import dump_context
from mhealth.utils.plotter_helper import save_figure, setup_plotting

# PATHS ----------------------------------------------------------------------------
wd = os.getcwd()
path_src = '/Users/julien/My Drive/20_STUDIUM/ACLS/05 Module/TM/mhealth/src'
path_output = '/Users/julien/My Drive/20_STUDIUM/ACLS/05 Module/TM/OUTPUT'
plots_path = '/Users/julien/My Drive/20_STUDIUM/ACLS/05 MODULE/TM/OUTPUT/plots/'


# LOAD DATA ----------------------------------------------------------------------------
# Demmi
filepath = '/Users/julien/GD/ACLS/TM/DATA/extracted/quality50_clipped_collected/csv/demorton-vital.csv'
vital = pd.read_csv(filepath)
vital.head




# Borg
filepath = '/Users/julien/GD/ACLS/TM/DATA/Borg/iMove_Borg_JB.csv'


# PROCESSING ----------------------------------------------------------------------------

