import os
import unittest

import pandas as pd

from data_analysis.sanity_checker import SanityChecker

_MHEALTH_DATA = os.getenv('MHEALTH_DATA', '../../resources')
_MHEALTH_OUT_DIR = os.path.join(_MHEALTH_DATA, 'output')


class SanityCheckerTest(unittest.TestCase):
    checker = SanityChecker()
    out_dir = _MHEALTH_OUT_DIR

    def test_run_vital(self):
        in_dir = f'{_MHEALTH_DATA}/vital_signals/'
        out_file = os.path.join(self.out_dir, 'sanity-test-vital.csv')
        self.checker.run_vital(in_dir=in_dir,
                               id_range=30,
                               in_file_suffix='_storage-sig.csv',
                               out_file=out_file)

        df = pd.read_csv(out_file)
        self.check_signal_float(df, 'hours', 0.008333333, 0.009444444, 61.86916667)
        self.check_signal_float(df, '#ValidRows', 5, 14, 47825)
        self.check_signal_float(df, 'HR_mean', 44.66666667, 40.55172414, 55.11635642)
        self.check_signal_float(df, 'HR_std', 42.3792402, 42.72803212, 11.82079019)
        self.check_signal_float(df, 'HR_min', 0,0,0)
        self.check_signal_float(df, 'HR_max', 81, 86, 74)


    def test_run_mixed_raw_vital(self):
        in_dir = f'{_MHEALTH_DATA}/mixed_vital_raw_signals/'
        out_file = os.path.join(self.out_dir, 'sanity-test-mixed-raw-vital.csv')
        self.checker.run_mixed_raw_vital(in_dir=in_dir,
                                         id_range=30,
                                         in_file_suffix='_Test_mixed_raw.csv',
                                         out_file=out_file)
        df = pd.read_csv(out_file)
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


    @unittest.SkipTest
    def test_run_full(self):
        in_dir = ''
        in_file_suffix = '.csv'
        out_file = os.path.join(self.out_dir, '.csv')
        self.checker.run_vital(in_dir=in_dir,
                               id_range=84,
                               in_file_suffix=in_file_suffix,
                               out_file=out_file)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
