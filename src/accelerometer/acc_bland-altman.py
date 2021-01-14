# =============================================================================
# LIBRARY
# =============================================================================
import os
import sys
import pandas as pd
from datetime import datetime
from datetime import timedelta
from pathlib import Path
# from natsort import natsort

sys.path.insert(1,'/Users/JULIEN/Google Drive/20_STUDIUM/ACLS/Module/TM1/imove/mhealth/src')

# =============================================================================
# LOAD DATA
# =============================================================================

from data_analysis.symmetry_checker import SymmetryChecker
from utils.file_helper import FileHelper

_MHEALTH_DATA = os.getenv("MHEALTH_DATA", "../resources")



data_dir = f"{_MHEALTH_DATA}/imove/data"
out_dir = FileHelper.get_out_dir(data_dir, '_symmetry')

check = SymmetryChecker(data_dir=data_dir,
                        out_dir=out_dir,
                        columns=["HR"],
                        resample=None)
check.run()


# see: https://github.com/hirsch-lab/mhealth/blob/feature/io_refactoring/studies/imove/symmetry_analysis.py
# auf neuem branch.