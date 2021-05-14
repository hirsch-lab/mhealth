# iMove data analysis

<!--
IMOVE_DATA="$DATA_ROOT/wearables/studies/usb-imove"
-->

In the following, `$IMOVE_DATA` points to the root directory of the iMove project (as found on our data repository). More specifically, the directory is assumed to have the following subfolder `extractions/quality50/` that contains the data produced by [preprocessing step 3](https://github.com/hirsch-lab/mhealth/blob/main/studies/imove/preprocessing/protocol.md#3-de-morton-data-extraction): 

- store/*.h5: HDF stores with keys /exercises/, /vital/left, /vital/right, /raw/left, /raw/right 
- summary*.csv files with statistical summaries for some metrics (evaluated before or after quality filtering)


### hdf2csv utility

Utility to extract the tables stored in HDF files and write it as .csv files.

```bash
$MHEALTH_ROOT/bin/hdf2csv.py -i "<path/to/store/>" \
                             -o "<path/to/output>"
```


    
### data dimensions

    - P: patient (n=60)
    - E: exercise (n=20, 15 exercises, some with sub-exercises)
    - D: day (n=3, {1,2,3})
    - M: metric (n=5, {HR, AX, AY, AZ, A} ...)
    - S: side (n=2, {left, right})



### Available metrics
The signals available by default are:

- *HR* (1Hz)
- *HRQ* (1Hz)
- *SpO2* (1Hz)
- *SpO2Q* (1Hz)
- *BloodPressure* (1Hz)
- *BloodPerfusion* (1Hz)
- *Activity* (1Hz)
- *Classification* (1Hz)
- *QualityClassification* (1Hz)
- *RespRate* (1Hz)
- *HRV* (1Hz)
- *LocalTemp* (1Hz)
- *ObjTemp* (1Hz)
- *AX* (50Hz)
- *AY* (50Hz)
- *AZ* (50Hz)
- *A* (50Hz, **derived**, L2-norm of the acceleration vector)

See the header of [preprocess_everion.py](https://github.com/hirsch-lab/mhealth/blob/main/studies/imove/preprocessing/preprocess_everion.py) for valid choices. The union of `VITAL_COLS` and `RAW_COLS` are available. More options are possible, see the table [everion_columns.csv](https://github.com/hirsch-lab/mhealth/blob/main/studies/imove/preprocessing/everion_columns.csv) for possible alternatives. In the case additional, non-default parameters are required, the preprocessing must be rerun.



## 1. Summary data

Visualization of the summary data collected in extract\_demorton\_data.py. See [`measure_info()`](https://github.com/hirsch-lab/mhealth/blob/main/studies/imove/preprocessing/extract_demorton_data.py) and the summary files created by the aforementioned script. This data is based only on **vital** data.

```bash
# Summary visualizations
python "analysis_data.py" \
            --in-dir "$IMOVE_DATA/extracted/quality50/" \
            --out-dir "../output/analysis/everion_data_q50"
# Output:
#   A series of plots.
#           gap_counts: Visualize how the number of time gaps evolves 
#                       for increasing gap thresholds (5min, 10min, ...),
#                       comparing the data before and after quality 
#                       filtering. The plots suffixed with "_rel" divide 
#                       the counts by the total time; files suffixed with 
#                       "_full" just include an extra, rather narrow 
#                       time gap of 1min.
#           max_gaps:   Shows the total hours of measured data and the
#                       maximal time gap observed in the datasets, before
#                       and after quality filtering
#           qualities:  Distribution of mean values for selected quality
#                       signals, before and after quality filtering.
#           time_rec:   Overview of the total time (time between first 
#                       and last measurement) and the time with valid
#                       recordings (⇔ number of available timestamps),
#                       before ("unfiltered") and after ("filtered")
#                       quality filtering.
```



## 2. Visualization data during De Morton exercises

This script collects data from all patients (both left and right side) during the De Morton exercises as extracted by the extract\_demorton\_data.py. In addition, it combines and aligns the data of all patients. Temporal alignment is  achieved by measuring the time elapsed (in seconds) 

- since exercise start if a particular exercise is of interest or 
- since the start of exercise 1 if the total De Morton session is of interest. (If exercise 1 was omitted, which was the case for 1 or 2 cases, the start of the first documented exercise was used instead)

For temporal alignment, the timestamps recorded manually by the physiotherapist ("chronme.com") were used.

**Note**: The collection of De-Morton extraction data from all patients takes a couple of seconds. To speed up repetitive calls to the plotting routines, the collected data is stored and re-loaded lazily. This behavior can be disabled by using the flag `--force-read`.

**Note**: If problems occur, it may help to use `--force-read`.

```bash
# Visualize data for De Morton exercises.
python "analysis_demorton.py" \
            --in-dir "$IMOVE_DATA/extracted/quality50" \
            --out-dir "../output/analysis/demorton" \
            --metrics A HR
            
# Options:
#   --in-dir            Path to De Morton data extraction
#   --out-dir           Output directory
#   --metrics           List of parameters to include (e.g. A, HR)
#                       Default: [A] (A: magnitude of acceleration)
#   --labels            List of De Morton labels (exercises) to 
#                       include. Default: all exercises 
#   --patients          List of patients to include. Default: all
#   --side              Choose between 'left', 'right', 'both'
#   --resample          Resampling period in seconds. This option is
#                       experimental. Avoid too small values since 
#                       the memory requirements can explode quickly.
#   --n-pats            Maximal number of patients to load (make sure
#                       to use option --force-read)
#   --force-read        Disable lazy-loading of data
# Output:
#   Plots:
#       availability:   Data availability for the three De Morton
#                       exercises (overall, or with exercises 
#                       indicated).
#       data:           Visualize data in different modes:
#                       - by exercise (all patients)
#                       - by exercise (all patients, left vs. right)
#                       - by exercise and patient
#   Data:
#       demorton.h5:    Collected data of all patients and exercises 
#       ex-*.h5:        Collected data of all patients per exercise.
#       Those stores are used for lazy-loading the data.
```

Some further examples:

```bash            
# Visualize data for (selected) De Morton exercises with clipped data. 
# Not recommended for availability plots; the per-exercise plots are
# equivalent. Lazy loading of clipped data is faster (useful for 
# developing/testing).
python "analysis_demorton.py" \
            --in-dir "$IMOVE_DATA/extracted/quality50" \
            --out-dir "../output/analysis/demorton" \
            --labels 1 2a 2b 2c 2d 3a 3b 4 5a 5b 6 7 8 9 10 11 12 13 14 15

# Resample data. Warning: For resampling values <1, the memory usage
# temporarily peaks and may increase memory pressure on the system. 
# Better limit the number of patients shown/loaded.
python "analysis_demorton.py" \
            --in-dir "$IMOVE_DATA/extracted/quality50" \
            --out-dir "../output/analysis/demorton_resample0.2" \
            --resample 0.2 --n-pats 20
            
# Visualize data only for selected patients and exercises.
python "analysis_demorton.py" \
            --in-dir "$IMOVE_DATA/extracted/quality50/store" \
            --out-dir "../output/analysis/demorton" \
            --patients 001 002 042 \
            --labels 2a 2b 2c 2d
```


Below the availability plot for acceleration for all patients.

![Availability plots](resources/data_availability_ex.pdf)

    


