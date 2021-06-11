# Tests with data from COVID19 station

Problem statement: In the COVID19 station of the University Hospital in Basel, wearable data was collected using new devices. Validate the wearable data with conventionally measured data.

In particular, the following should be checked per vital sign: For $n_{valid}$ conventional measurements available for comparison, determine the rate of wearable measurements $r_{correct}$ for which the data were within an acceptable range $\Delta x$. There were x false positive deviations of more than x (tbd for each vital sign) and x false negative deviations.


- Data: `$DATA_ROOT/wearables/studies/usb-covid-bbtest`
- Details: See the email archived with the data.


## Notes

- Renamed the following folders (with Bianca's approval)
  - `2652_F_-_jetzt_in_2656_FL` ⇒ `2652_F`  - `2652_T_-_jetzt_in_2664_F` ⇒ `2652_T`  - `2653_F_-_jetzt_in_2655_F` ⇒ `2653_F`
  - `2668` ⇒ `2668_E` (Einzelzimmer)
- Validation data (Validierung\_Daten\_manuell\_Mai2021\_alle): Timestamp for temperature is the same for as for heart rate.
- Further details: see mail correspondence in the data folder