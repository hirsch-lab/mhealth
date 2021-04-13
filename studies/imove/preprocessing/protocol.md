## Data preprocessing

To trigger the complete preprocessing procedure

```bash
python "preprocess_exercises.py"    # a couple of seconds
python "preprocess_everion.py"      # 1-2 hours
python "extract_demorton_data.py"   # a couple of minutes
```


### 1. De Morton exercises

```bash
python "preprocess_exercises.py"
```

- The patients performed the De Morton exercises on three (consecutive) days.
- When the patients performed the exercises, timings were measured manually. This data was stored in .xlsx files, for each patient and exercise day separately.
- This preprocessing step comprises
    - The exercise data from the three days are combined per patient and formatted.
    - Some typos are fixed with regard the task identifiers:
        - t → temp
        - temo → temp
        - fault → default **??????**
        - df → default
        - def → default
        - defaukt → default
- **Output** is generated in the output directory:
    - outdir/exercises: .csv data
    - outdir/store: .h5 data


### 2. Sensor data

```bash
python "preprocess_everion.py"
```

- Two different types of readouts are available: the readouts of vital signals (sampling at 1Hz), and the raw sensor data (sampling at 50Hz). The latter includes also data from the optical sensors for the PPG (photoplethysmogram, pulse oximetry). Several vital signs are derived from those optical signals.
- Our analyses assume that the computation of heart rate, oxygen saturation and other PPG-derived measures were performed correctly. The raw data is used only to read out accelerometer data.
- Unfortunately, all timestamps in the data files are provided only in second resolution, which is bad for the raw sensor data sampled at 50Hz.
- Raw data files are malformatted: A few lines lack the data of four columns: `greencurr` (37), `redcurr` (38), `IRcurr` (39), `ADCoffs` (40). For those lines, the data of the subsequent columns are shifted by four columns, instead of keeping nan-values in those columns. This breaks the data type consistency of the columns, issue some warning when reading in .csv files and creates problems when storing the tables into different storage formats (e.g., HDF). The problem occurs only for a few entries (<0.1%), typically in blocks of 17.
Details: It's always a block of 17 lines missing - a 1/3 of the 51 entries available per second. It seems that for a sampling rate of 51Hz, the buffer read out is at 3Hz. The block of missing data is sometimes split in two parts (10 and 7 lines), one occurring chronologically at the correct place, the second one a couple of seconds later. In other words: the table is chronologically not in order! (*)
Different possibilities to fix the problem: (1) remove only the broken rows, (2) remove the lines with the same timestamp, or (3) shift the columns.
I decided to go with (1), assuming that the data in the correct format is correct, with the goal to have the least downstream impact. (2) would lead to "missing values" in the timeline (if represented in second resolution). (3) would require the table to be sorted, which increases processing time and might change the order of the values.
- Some raw files did not have any content. Those were ignored.
    - `iMove_019_storage-vital_raw__left.csv`
    - `iMove_021_storage-vital_raw__left.csv`
    - `iMove_022_storage-vital_raw__left.csv`
    - `iMove_023_storage-vital__right.csv`
    - `iMove_023_storage-vital_raw__right.csv`
    - `iMove_024_storage-vital_raw__left.csv`
    - `iMove_026_storage-vital_raw__left.csv`
    - `iMove_028_storage-vital__right.csv`
    - `iMove_028_storage-vital_raw__right.csv`
    - `iMove_029_storage-vital_raw__left.csv`
    - `iMove_032_storage-vital_raw__left.csv`
    - `iMove_033_storage-vital_raw__left.csv`
    - `iMove_038_storage-vital_raw__left.csv`
    - `iMove_041_storage-vital_raw__left.csv`
    - `iMove_045_storage-vital__right.csv`
    - `iMove_045_storage-vital_raw__left.csv`
    - `iMove_045_storage-vital_raw__right.csv`
    - `iMove_047_storage-vital_raw__left.csv`
    - `iMove_048_storage-vital_raw__left.csv`
    - `iMove_051_storage-vital__left.csv`
    - `iMove_051_storage-vital__right.csv`
    - `iMove_051_storage-vital_raw__left.csv`
    - `iMove_051_storage-vital_raw__right.csv`
    - `iMove_052_storage-vital_raw__left.csv`
- Renamed original column names to make the naming more consistent
    - SPo2 → SpO2
    - SPO2Q → SpO2Q
    - steps → Steps
    - phase → Phase
    - phase → PhaseInt
    - localtemp → LocalTemp
    - objtemp → ObjTemp
    - baromtemp → BaromTemp
    - pressure → Pressure
    - greencurr → GreenCurr
    - redcurr → RedCurr
    - IRcurr → IRCurr
    - ADCoffs → ADCoeffs
- The timing measurements from the exercises (preprocessing step 1) are also merged into the tables (both raw and vital)


### 3. De Morton data extraction

```bash
python "extract_demorton_data.py"
```

- In this third step, the data of interest is extracted
- Main parameters:
    - `delta_minutes`: time margin between first and last De Morton measurement.
    - `quality`: 
- Quality filtering:
    - This is always true: If the vital data is missing, the raw sensor data is 
    - The converse does not hold: if the raw data is missing, it is possible that the vital data is available.
    - **Question**: shall we discard also the vital data if the sensor data is missing? Currently, we keep it.
    - Filtering rules: 
        - Heart rate > 0 and 
        - HRQ > quality and 
        - QualityClassification > quality
- The data is stored in separate .h5 stores
- **Output** is generated in the output directory:
    - outdir/extraction/: .h5 data