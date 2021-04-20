# iMove data preprocessing

To trigger the complete preprocessing: 

```bash
# Step 1 (runs 2-3 minutes)
python "preprocess_exercises.py" \
            --in-dir "$IMOVE_DATA/original/exercises/" \
            --out-dir "../output/preprocessed/"
# Step 2 (runs 1-2 hours)
python "preprocess_everion.py" \
            --in-dir "$IMOVE_DATA/original/sensor/" \
            --out-dir "../output/preprocessed/"
# Step 3 (runs 15-20 minutes)
python "extract_demorton_data.py" \
            --in-dir "../output/preprocessed/" \
            --out-dir "../output/extracted/quality50" \
            --quality=50
```

<!--
IMOVE_DATA="$DATA_ROOT/wearables/studies/usb-imove"
-->

`$IMOVE_DATA` points to the root directory of the iMove project (as found on our data repository). More specifically, the folder is assumed to have the following substructure:

- **original/exercises/**: contains .xlsx files with manually collected data about the duration of De Morton exercises
- **original/sensor/**: contains the original .csv files representing the readouts of the Everion devices




## 1. De Morton exercises

```bash
python "preprocess_exercises.py" \
            --in-dir "$IMOVE_DATA/original/exercises/" \
            --out-dir "../output/preprocessed/"

# Output:
#   .../exercises/_all.csv: collection of all exercise data
#   .../exercises/:         .csv data per patient
#   .../store/:             .h5 data, with HDF-key "/exercises"
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



## 2. Sensor data

```bash
python "preprocess_everion.py" \
            --in-dir "$IMOVE_DATA/original/sensor/" \
            --out-dir "../output/preprocessed/"
  
# Options:
#   --csv:          Create also .csv output (in addition to .h5)
#
# Output:
#   .../store/:     Updates .h5 data from step 1 with new HDF-keys:
#                   "/vital/left": vital sign data from left device
#                   "/vital/right": vital sign data from right device
#                   "/raw/left": accelerometer data from left device
#                   "/raw/right": accelerometer data from right device
#   .../sensor/:    Optional, folder with .csv data
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
- See the file [everion_columns.csv](https://github.com/hirsch-lab/mhealth/blob/feature/imove_processing/studies/imove/preprocessing/everion_columns.csv), it determines the columns of the resulting DataFrames, alongside with the dtypes for those columns.
- Unfortunately, the timestamps are not monotonically increasing. See for example patient 003, vital/left, at date 2018-08-14 02:00:27+00:00. 
- Finally, the information about the the De Morton exercise session (c.f. preprocessing step 1) are also merged into the tables (both raw and vital). This adds three columns: 
    - DeMorton: boolean indicating if a De Morton exercise is currently executed
    - DeMortonLabel: str identifying the De Morton exercise, or None
    - DeMortonDay: int identifying the De Morton session, or None

## 3. De Morton data extraction

```bash
python "extract_demorton_data.py" \
            --in-dir "../output/preprocessed/" \
            --out-dir "../output/extracted/quality50" \
            --quality=50

# Parameters:
#   --quality:          Threshold for quality filtering. Must be a value 
#                       between 0 and 100. Default: 50
#   --margin:           Determines the amount of data extracted before and
#                       after the De Morton exercise sessions. Measured in 
#                       minutes. Default: 15 
#   --max-gap:          Maximal time gap tolerated, in hours. Data recorded
#                       after such an extremal time gap are clipped. 
#                       Default: 36
#
# Output:
#   .../store/:         .h5 stores with extracted data per patient. The 
#                       stores contain the following keys: /exercises/, 
#                       /vital/left, /vital/right, /raw/left, /raw/right
#   .../csv/:           Folder with .csv data per patient
#   .../exercises.csv:  Identifcal with the _all.csv from step 1
#   .../summary*.csv:   Statistical summaries for some metrics, evaluated
#                       before or after quality filtering.

```

- In this third step, the data of interest is extracted, defined by the (usually) three De Morton exercise sessions. The time windows are read from exercises collection "exercises/_all.csv" from step 1.
- In some cases, the Everion files contain data recorded several days after the last De Morton exercise session. This extra data is clipped away.
- Quality filtering:
    - This holds true always: If the vital data is missing, the raw sensor data is also missing.
    - The converse does not: if the raw data is missing, it is possible that the vital data is available.
    - (**Question**: shall we discard the vital data if the sensor data is missing? Currently, we keep it.)
    - Filtering rules: 
        - Heart rate > 0 and 
        - HRQ > quality and 
        - QualityClassification > quality

