# iMove data analysis

To trigger all analysis: 

```bash
# Basic data analysis
python "analysis_data.py" \
            --in-dir "$IMOVE_DATA/extracted/quality50/" \
            --out-dir "../output/analysis/everion_data_q50"
            
# Visualize data for De Morton exercises. Note that data is loaded 
# "lazily" by reading HDF-stores from previous runs. This helps to
# save time for repeated runs. Use the flag --force-read to disable
# lazy loading of data.
python "analysis_demorton.py" \
            --in-dir "$IMOVE_DATA/extracted/quality50" \
            --out-dir "../output/analysis/demorton"
            
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

<!--
IMOVE_DATA="$DATA_ROOT/wearables/studies/usb-imove"
-->

`$IMOVE_DATA` points to the root directory of the iMove project (as found on our data repository). More specifically, the folder is assumed to have the following substructure: 

- **extractions/quality50/**: contains the data produced by [preprocessing step 3](https://github.com/hirsch-lab/mhealth/blob/feature/imove_processing/studies/imove/preprocessing/protocol.md#3-de-morton-data-extraction)
    -  **store/**: .h5 stores with extracted data per patient. The stores contain the following keys: 
        -  /exercises/
        -  /vital/left
        -  /vital/right
        -  /raw/left
        -  /raw/right
    -  **summary\*.csv**: Statistical summaries for some metrics, evaluated before or after quality filtering.

    
## 1. Basic data visualization

**TODO** This section is not up-to-date.

There are four data dimensions:

    - P: patient (n=60)
    - E: exercise (n=20 + 2 (temp, default)
    - D: day (n=3, {1,2,3})
    - M: metric (n=5, {HR, AX, AY, AZ, A}, more are possible)
    - S: side (n=2, {left, right})

In the extreme case, P x E x D x M x S separate plots can be created. However,
it is reasonable to reduce this number by plotting multiple times to the same plot. The following configurations make sense:

    - PSD x E x M:   plot data per exercise and metric
    - PSED x M:      plot data per metric (one time axis)

Besides subsampling, no means to reduce memory demand is applied. The data (<2GB) fits well into RAM.


### Parameters:

- `--sampling`: Sampling rate for the data in Hz. If the sampling rate of the measured data is lower than the requested one, the parameter is ignored.
- `--metrics`: List of metrics to include. Default: {HR, AX, AY, AZ, A}



### Available metrics:

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


See the header of [preprocess_everion.py](https://github.com/hirsch-lab/mhealth/blob/feature/imove_processing/studies/imove/preprocessing/preprocess_everion.py) for valid choices. The union of `VITAL_COLS` and `RAW_COLS` are available. More options are possible, see the table [everion_columns.csv](https://github.com/hirsch-lab/mhealth/blob/feature/imove_processing/studies/imove/preprocessing/everion_columns.csv) for possible alternatives. In the case additional, non-default parameters are required, the preprocessing must be rerun.

In addition to the above measured signals, the following derived metrics are available:

- *A* (50Hz): The L2-norm of the acceleration vector


