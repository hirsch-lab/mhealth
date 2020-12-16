import os
from pathlib import Path

from data_analysis.symmetry_checker import SymmetryChecker

data_root = Path("/Users/norman/workspace/education/phd/data/wearables")
data_dir = data_root / "studies/usb-imove/processed_data"
data_dir = data_dir / "cleaned2_quality_filtered_50_labeled"
out_dir = "./results/left-right-correlation"

check = SymmetryChecker(data_dir=data_dir,
                        out_dir=out_dir,
                        columns=["HR", "HRV", "HRQ", "SPo2", "Classification",
                                 "RespRate", "BloodPressure", "Activity"],
                        #resample="30s")
                        resample=None)
check.run()
