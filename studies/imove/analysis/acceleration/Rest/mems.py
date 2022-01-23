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

# Working directory (angegeben in upper right) must be located in
# /imove/analysis/acceleration, sost geht import context.py nicht!

import context # it can oly import context.py when contained in the same folder
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


# LOAD quality50_clipped_collected / EX-6 ----------------------------------------------------------------------------
# ex-6
filepath = '/Users/julien/GD/ACLS/TM/DATA/extracted/quality50_clipped_collected/store/ex-6.h5'
store = pd.HDFStore(filepath, mode='r')
print(store.keys())
acc = store["raw"]
#df = store["vital"]
# acc.head
acc.info()


# ----------------------------------------------------------------------------

# filter for ex6_pat006_1L
ex6_pat006_1L = acc[acc.Patient.eq('006') &
                acc.DeMortonDay.eq('1') &
                acc.Side.eq('left')]



sampling_rate = 50 # Sampling frequency: 50 Hz. Sensor output: 51.2 Hz. Which one to take?
cutoff_freq = 0.25 # Hz

# Apply high-pass filtering on raw accelerometer data to separate AC and DC component
# ADC [n] = a1 A[n] + b1 ADC [n − 1]	
# not done yet

# ----------------------------------------------------------------------------

