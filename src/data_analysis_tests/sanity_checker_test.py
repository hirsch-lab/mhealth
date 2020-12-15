import os
import unittest

import pandas as pd

from data_analysis.sanity_checker import SanityChecker


class SanityCheckerTest(unittest.TestCase):
    checker = SanityChecker()

    def test_run_vital(self):
        in_dir = '../resources/vital_signals/'
        out_file = 'sanity-test-vital.csv'
        self.checker.run_vital(in_dir, 30, '_storage-sig.csv', out_file)

        df = pd.read_csv(os.path.join('../resources/', out_file))
        self.check_signal_float(df, 'hours', 0.008333333, 0.009444444, 61.86916667)
        self.check_signal_float(df, '#ValidRows', 5, 14, 47825)
        self.check_signal_float(df, 'HR_mean', 44.66666667, 40.55172414, 55.11635642)
        self.check_signal_float(df, 'HR_std', 42.3792402, 42.72803212, 11.82079019)
        self.check_signal_float(df, 'HR_min', 0,0,0)
        self.check_signal_float(df, 'HR_max', 81, 86, 74)

    def test_run_mixed_raw_vital(self):
        in_dir = '../resources/mixed_vital_raw_signals/'
        out_file = 'sanity-test-mixed-raw-vital.csv'
        self.checker.run_mixed_raw_vital(in_dir, 30, '_Test_mixed_raw.csv', out_file)
        df = pd.read_csv(os.path.join('../resources/', out_file))
        self.check_signal_float(df, 'hours', 73.04111111, 0.004722222, 0.004722222)
        self.check_signal_float(df, '#ValidRows', 149, 18, 18)
        self.check_signal_float(df, 'HR_mean', 79.04697987, 77.88888889, 69.94444444)
        self.check_signal_float(df, 'HR_std', 6.408752008, 1.131832917, 3.038424945)
        self.check_signal_float(df, 'HR_min', 70,77,66)
        self.check_signal_float(df, 'HR_max', 91, 81, 76)


    def check_signal_float(self, df, signal, value1, value2, value3):
        self.assertAlmostEqual(df[signal][0], value1, 8)
        self.assertAlmostEqual(df[signal][1], value2, 8)
        self.assertAlmostEqual(df[signal][6], value3, 8)

    #@unittest.SkipTest
    def test_run_full(self):
        dir_name = '/Users/sues/Documents/wearables/imove/cleaned2_quality_filtered2_50'
        self.checker.run_imove(dir_name, 30, '_storage-vital.csv', 'quality-after.csv')
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
