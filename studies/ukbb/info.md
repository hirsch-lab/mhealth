# UKBB study

<!--
Collaborators:
UKBB: Ines Mack, Julia Bielicki
ZHAW: Norman Juchler, Susanne Suter, Sofia Rey
-->

Data analysis contributing to an exploratory study on the utility of wearable technologies for monitoring pediatric patients with surgical infections.

Study period: 2021-2022  
Publication: Pending


```bash
# Command to run the analysis:
python analysis_data.py \
   --in-dir "$DATA_ROOT/wearables/studies/ukbb/" \
   --quality=-1 \
   --max-gap=24
```