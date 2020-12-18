import glob
import os
import unittest

from data_analysis.symmetry_checker import SymmetryChecker
from utils.file_helper import FileHelper

_MHEALTH_DATA = os.getenv("MHEALTH_DATA", "../resources")


class QualityFilterTest(unittest.TestCase):

    def test_symmetry_analysis(self):
        data_dir = f"{_MHEALTH_DATA}/imove/data"
        out_dir = FileHelper.get_out_dir(data_dir, '_symmetry')

        check = SymmetryChecker(data_dir=data_dir,
                                out_dir=out_dir,
                                columns=["HR"],
                                resample=None)
        check.run()

        files = glob.glob(os.path.join(os.path.join(out_dir, '**'), '*.png'), recursive=True)
        self.assertEqual(1, len(files))

        files = glob.glob(os.path.join(os.path.join(out_dir, '**'), '*.csv'), recursive=True)
        self.assertEqual(2, len(files))


if __name__ == "__main__":
    unittest.main()

