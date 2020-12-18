import os
from pathlib import Path

from data_analysis.symmetry_checker import SymmetryChecker

data_root = Path("/Users/norman/workspace/education/phd/data/wearables")
data_dir = data_root / "studies/usb-imove/processed_data/cleaned_data/"
data_dir = data_dir / "cleaned2_labeled_quality_filtered_80"
#data_dir = data_dir / "cleaned2_labeled"
out_dir = "./results/symmetry-new"

check = SymmetryChecker(data_dir=data_dir,
                        out_dir=out_dir,
                        columns=["HR"],
                        resample=None)
                        #resample=None)
check.run()
