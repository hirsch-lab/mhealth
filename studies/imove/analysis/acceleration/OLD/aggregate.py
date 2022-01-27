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




# exercises (): StartDate, EndDate for each combination of Patient, Day, Task
filepath = '/Users/julien/GD/ACLS/TM/DATA/extracted/quality50_clipped/exercises.csv'
exercises = pd.read_csv(filepath)
exercises.Task.unique()

# demmi exercises of interest: ex-12 (Gehen 50m), ex-15 (Springen) ----------------------------------------------------------------------------
demmi_ex_interest = ['12', '15']
window_size = 2200 # 400 = 8sec
window_size = 400

# demorton_pat001_pat002_pat006.h5
# 3 Pat: 001, 002, 006, left%right, 
filepath = '/Users/julien/GD/ACLS/TM/DATA/extracted/quality50_clipped_collected/store/demorton_pat001_pat002_pat006.h5'
store = pd.HDFStore(filepath, mode='r')
acc = store["raw"]

acc.DeMortonDay.unique()
acc.Side.unique()
acc.Patient.unique()


ex12_pat020_2L = acc[acc.Patient.eq('020') & # filter pat, day, side
                acc.DeMortonDay.eq('2') &
                acc.Side.eq('left')]



