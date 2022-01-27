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



# specific pat ----------------------------------------------------------------------------
# DeMortonLabel are there for actual DEMMI ex (but not for the 15' margins)
filepath = '/Users/julien/GD/ACLS/TM/DATA/extracted/quality50_clipped/store/022.h5' # 001
store = pd.HDFStore(filepath, mode='r')
print(store.keys())
pat_022_R = store["raw/right"]
# acc.info()

# exercises (): StartDate, EndDate for each combination of Patient, Day, Task
filepath = '/Users/julien/GD/ACLS/TM/DATA/extracted/quality50_clipped/exercises.csv'
exercises = pd.read_csv(filepath)
exercises.Task.unique()



# demmi exercises of interest: ex-12 (Gehen 50m), ex-15 (Springen) ----------------------------------------------------------------------------
demmi_ex_interest = ['12', '15']
window_size = 2200 # 400 = 8sec
window_size = 400

# ex-12
filepath = '/Users/julien/GD/ACLS/TM/DATA/extracted/quality50_clipped_collected/store/ex-12.h5'
store = pd.HDFStore(filepath, mode='r')
acc = store["raw"]
ex12_pat020_2L = acc[acc.Patient.eq('020') & # filter pat, day, side
                acc.DeMortonDay.eq('2') &
                acc.Side.eq('left')]

# ex-15
filepath = '/Users/julien/GD/ACLS/TM/DATA/extracted/quality50_clipped_collected/store/ex-15.h5'
store = pd.HDFStore(filepath, mode='r')
acc = store["raw"]
ex15_pat020_2L = acc[acc.Patient.eq('020') & # filter
                acc.DeMortonDay.eq('2') &
                acc.Side.eq('left')]



# PROCESS  ----------------------------------------------------------------------------
# append ex-12 and ex-15
df = ex12_pat020_2L.append(ex15_pat020_2L)
# df = df.reset_index() # create new index col (before timestamp was the index)


# muss die beiden plots auseinandernehmen!
for i in demmi_ex_interest:
  acc = df[(df['DeMortonLabel'] == i)].reset_index() # [:400]
  
  # zwischenberechnung
  mid = round(acc.index[-1] / 2)
  window_begin = int(mid - window_size/2) # must convert float to int
  window_end   = int(mid + window_size/2)
  acc = acc[window_begin : window_end] # filter to window_size, centered around mid
  
  plt.figure(figsize = (15, 6))
  sns.lineplot(y = 'AX', x = 'timestamp', data = acc)
  sns.lineplot(y = 'AY', x = 'timestamp', data = acc)
  sns.lineplot(y = 'AZ', x = 'timestamp', data = acc)
  plt.legend(['AX', 'AY' 'AZ'])
  plt.ylabel(i)
  plt.title(i, fontsize = 15)
  plt.show()
  
  
  # test 12
  acc = df[(df['DeMortonLabel'] == '12')].reset_index() # [:400]
  
  # zwischenberechnung
  mid = round(acc.index[-1] / 2)
  window_begin = int(mid - window_size/2) # must convert float to int
  window_end   = int(mid + window_size/2)
  print(window_begin)
  print(window_end)
  acc[window_begin : window_end] # filter to window_size, centered around mid
  
  plt.figure(figsize = (15, 6))
  sns.lineplot(y = 'AX', x = 'timestamp', data = acc)
  sns.lineplot(y = 'AY', x = 'timestamp', data = acc)
  sns.lineplot(y = 'AZ', x = 'timestamp', data = acc)
  plt.legend(['AX', 'AY' 'AZ'])
  plt.ylabel('12')
  plt.title('12', fontsize = 15)
  plt.show()
  
    # test 15 (timestamp wird hier komisch)
  acc = df[(df['DeMortonLabel'] == '15')].reset_index() # [:400]
  
  # zwischenberechnung
  mid = round(acc.index[-1] / 2)
  window_begin = int(mid - window_size/2) # must convert float to int
  window_end   = int(mid + window_size/2)
  print(window_begin)
  print(window_end)
  acc[window_begin : window_end] # filter to window_size, centered around mid
  
  plt.figure(figsize = (15, 6))
  sns.lineplot(y = 'AX', x = 'timestamp', data = acc)
  sns.lineplot(y = 'AY', x = 'timestamp', data = acc)
  sns.lineplot(y = 'AZ', x = 'timestamp', data = acc)
  plt.legend(['AX', 'AY' 'AZ'])
  plt.ylabel('15')
  plt.title('15', fontsize = 15)
  plt.show()
  






