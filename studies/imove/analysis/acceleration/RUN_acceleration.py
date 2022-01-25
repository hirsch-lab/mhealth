"""RUN_acceleration """
# LIBRARIES ----------------------------------------------------------------------------
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Import own modules
from acceleration import *

# Working directory (angegeben in upper right) must be located in
# /imove/analysis/acceleration, sost geht import context.py nicht!

import context # it can oly import context.py when contained in the same folder
# as demmi.py

#from src.mhealth.utils.commons import print_title
#from src.mhealth.utils.context_info import dump_context
from mhealth.utils.plotter_helper import save_figure, setup_plotting


# PATHS ----------------------------------------------------------------------------

# @Julien: Pfade und Parameter bitte nur in top-level Scripts. Hier ja, nicht
# aber in acceleration.py oder context.py.
# @Julien: Nicht verwendeten Code entfernen

#wd = os.getcwd()
#path_src = '/Users/julien/My Drive/20_STUDIUM/ACLS/05 Module/TM/mhealth/src'
#path_output = '/Users/julien/My Drive/20_STUDIUM/ACLS/05 Module/TM/OUTPUT'
#plots_path = '/Users/julien/My Drive/20_STUDIUM/ACLS/05 MODULE/TM/OUTPUT/plots/'
#path_data = '/Users/julien/GD/ACLS/TM/DATA/'

# EXECUTION ----------------------------------------------------------------------------

# put Xsec margins before/after each exercise
exercises = ['2a',' 12', '15']
df_margins = put_margins_around_ex(df=acc, demmi_ex=exercises, delta_seconds=delta_seconds)

# resample 51Hz to 1sec
df_resample = resample(df_margins, enable=True) # enable=False

# align all acc-curves with common starting time = 0
df_aligned = align_timestamp(df=df_resample)

# renaming
df = df_aligned

#### PLOT ----------------------------------------------------------------------------

# 1) Aggregated data plot
fig, ax = plt.subplots()
sns.lineplot(data=df, x="time", y="A", hue="Patient", ax=ax)
xticks = ax.get_xticks()
xticks = [pd.to_datetime(tm, unit="ms").strftime('%Y-%m-%d\n %H:%M:%S')
          for tm in xticks]
ax.set_xticklabels(xticks, rotation=45)
fig.tight_layout()

# 2)
# sns.relplot(
#     data=df, x="time", y="A",
#     col="DeMortonDay",  row='Patient',
#     hue="Side", # style="event",
#     kind="line"
# )

# 3) Plot for only 1 exercise
ex = '12' # specify ex to visualize
ex_text = demmi_ex[ex]

plot = sns.relplot(
        data=df[df.DeMortonLabel.eq(ex)],  # subset: only specific ex
        x="time", y="A",
        col="Side",  row='DeMortonDay',
        hue="Patient", # style="event",
        kind="line"
    )
plot.fig.suptitle(f'DEMMI Ex. {ex} \n {ex_text}', fontsize=30) # title
ax = plt.gca()
xticks = ax.get_xticks()
xticks = [pd.to_datetime(tm, unit="ms").strftime('%Y-%m-%d\n %H:%M:%S')
          for tm in xticks]
ax.set_xticklabels(xticks, rotation=45)
plt.tight_layout()


# 4) All exercises. Specific day
day = '1'

plot = sns.relplot(
        data=df[df.DeMortonDay.eq(day)],  # subset: only specific day
        x="time", y="A",
        col="Side",  row='DeMortonLabel',
        hue="Patient", # style="event",
        kind="line"
    )
ax = plt.gca()
xticks = ax.get_xticks()
xticks = ax.get_xticks()
xticks = [pd.to_datetime(tm, unit="ms").strftime('%Y-%m-%d\n %H:%M:%S')
          for tm in xticks]
ax.set_xticklabels(xticks, rotation=45)

plt.show()
                
                
                



