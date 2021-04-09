## Data preprocessing

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
