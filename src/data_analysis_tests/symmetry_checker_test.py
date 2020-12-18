import os
import unittest
from pathlib import Path
from data_analysis.symmetry_checker import SymmetryChecker

class QualityFilterTest(unittest.TestCase):

    @unittest.SkipTest
    def test_symmetry_analysis(self):
        data_dir = "../resources/imove/data"
        out_dir = "../results/symmetry-new"
        check = SymmetryChecker(data_dir=data_dir,
                                out_dir=out_dir,
                                columns=["HR"],
                                resample=None)
        check.run()
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()

