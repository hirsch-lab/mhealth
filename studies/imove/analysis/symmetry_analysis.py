import os
from pathlib import Path

import context
from data_analysis.symmetry_checker import SymmetryChecker

data_root = Path("/Users/norman/workspace/education/phd/data/wearables")
data_dir = data_root / "studies/usb-imove/processed_data/cleaned_data/"
#data_dir = data_dir / "cleaned2_labeled"
data_dir = data_dir / "cleaned2_labeled_quality_filtered_50"
out_dir = "./results/symmetry-new"

    #columns = ["HR"]
    columns = ["HR", "HRQ", "HRV", "RespRate",
               "SpO2", "BloodPressure",
               "Activity", "Classification"]

check = SymmetryChecker(data_dir=data_dir,
                        out_dir=out_dir,
                        columns=columns,
                        resample="30s") # resample="30s"
check.run()
