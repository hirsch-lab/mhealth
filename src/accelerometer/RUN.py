# This file is to run diverse scripts
import os
import sys
import pandas as pd
from datetime import datetime
from datetime import timedelta
from pathlib import Path

sys.path.insert(1,'/Users/JULIEN/Google Drive/20_STUDIUM/ACLS/Module/TM1/imove/mhealth/src')
sys.path.insert(2,'/Users/JULIEN/Google Drive/20_STUDIUM/ACLS/Module/TM1/imove/mhealth/src/accelerometer')
sys.path

# =============================================================================
# utils/everion_keys.py
# =============================================================================
import unittest
from everion_keys import EverionKeys

ek = EverionKeys()
ek.all_vital
ek.major_vital
ek.short_names_vital
ek.tag_units_vital
ek.tag_names_vital
ek.tag_names_mixed_vital_raw
ek.all_signals_mixed_vital_raw
ek.major_imove
ek.major_mixed_vital_raw
ek.short_names_mixed_vital_raw
ek.activity_class

# =============================================================================
# patient/patient_data_loader.py
# =============================================================================



