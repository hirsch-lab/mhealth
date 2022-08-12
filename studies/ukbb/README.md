# UKBB study

<!--
Collaborators:
UKBB: Ines Mack, Julia Bielicki
ZHAW: Norman Juchler, Susanne Suter, Sofia Rey
-->

Data analysis contributing to an exploratory study on the utility of wearable technologies for monitoring pediatric patients with surgical infections.

**Study period**: 2021-2022  
**Publication**: Ines Mack et al., Wearable Technologies for Pediatric Patients with Surgical Infections — More than Counting Steps? Biosensors 2022, 12, 634. https://doi.org/10.3390/bios12080634   
**Data**: Available on request from the authors 




```bash
# Command to run the analysis:
python analysis_data.py \
   --in-dir "$DATA_ROOT/wearables/studies/ukbb/" \
   --quality=-1 \
   --max-gap=24
```