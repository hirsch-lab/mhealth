import glob
import os
import unittest

from data_analysis.cross_correlator import CrossCorrelator
from utils.data_aggregator import Normalization
from visualization.vis_properties import VisProperties
from utils.everion_keys import EverionKeys
from utils.file_helper import FileHelper


class CrossCorrelatorTest(unittest.TestCase):
    in_dir = '../resources/vital_signals/'
    in_dir_mixed = '../resources/mixed_vital_raw_signals/'
    correlator = CrossCorrelator()

    def test_cross_correlator_daily(self):
        out_dir = FileHelper.get_out_dir(self.in_dir, '_daily_cross')
        self.correlator.plot_daily_correlations(VisProperties(in_dir=self.in_dir, out_dir=out_dir,
                                                              normalization=Normalization.NONE,
                                                              keys={'heart_rate', 'respiration_rate'},
                                                              short_keys=EverionKeys.short_names_vital,
                                                              min_scale=0, max_scale=100,
                                                              start_idx=0, end_idx=3
                                                              ))

        files = glob.glob(os.path.join(os.path.join(out_dir, '**'), '*.png'), recursive=True)
        self.assertEqual(6, len(files))

    def test_cross_correlator_hours_vital(self):
        out_dir = FileHelper.get_out_dir(self.in_dir, '_cross')
        self.correlator.plot_hourly_correlations(VisProperties(in_dir=self.in_dir, out_dir=out_dir,
                                                               normalization=Normalization.NONE,
                                                               keys=EverionKeys.major_vital,
                                                               short_keys=EverionKeys.short_names_vital,
                                                               min_scale=0, max_scale=100,
                                                               start_idx=0, end_idx=3
                                                               ))

        files = glob.glob(os.path.join(os.path.join(out_dir, '**'), '*.png'), recursive=True)
        self.assertEqual(3, len(files))

    def test_cross_correlator_hours_mixed_vital_raw(self):
        out_dir = FileHelper.get_out_dir(self.in_dir_mixed, '_cross')
        self.correlator.plot_hourly_correlations(VisProperties(in_dir=self.in_dir_mixed, out_dir=out_dir,
                                                               normalization=Normalization.NONE,
                                                               keys=EverionKeys.major_mixed_vital_raw,
                                                               short_keys=EverionKeys.short_names_mixed_vital_raw,
                                                               min_scale=0, max_scale=100,
                                                               start_idx=0, end_idx=3
                                                               ))

        files = glob.glob(os.path.join(os.path.join(out_dir, '**'), '*.png'), recursive=True)
        self.assertEqual(5, len(files))

    def test_cross_correlator_daily_mixed_vital_raw(self):
        out_dir = FileHelper.get_out_dir(self.in_dir_mixed, '_daily_cross')
        self.correlator.plot_daily_correlations(VisProperties(in_dir=self.in_dir_mixed, out_dir=out_dir,
                                                              normalization=Normalization.NONE,
                                                              keys=EverionKeys.major_mixed_vital_raw,
                                                              short_keys=EverionKeys.short_names_mixed_vital_raw,
                                                              min_scale=0, max_scale=100,
                                                              start_idx=0, end_idx=3
                                                              ))

        files = glob.glob(os.path.join(os.path.join(out_dir, '**'), '*.png'), recursive=True)
        self.assertEqual(25, len(files))

    @unittest.SkipTest
    def test_cross_hours_vital(self):
        in_dir = ''
        out_dir = FileHelper.get_out_dir(in_dir, '_cross')
        self.correlator.plot_hourly_correlations(VisProperties(in_dir=in_dir, out_dir=out_dir,
                                                               normalization=Normalization.NONE,
                                                               keys=EverionKeys.major_vital,
                                                               short_keys=EverionKeys.short_names_vital,
                                                               min_scale=0, max_scale=100,
                                                               start_idx=0, end_idx=3
                                                               ))

        self.assertTrue(True)

    @unittest.SkipTest
    def test_cross_days_vital(self):
        in_dir = ''
        out_dir = FileHelper.get_out_dir(in_dir, '_daily_cross')
        self.correlator.plot_daily_correlations(VisProperties(in_dir=in_dir, out_dir=out_dir,
                                                              normalization=Normalization.NONE,
                                                              keys=EverionKeys.major_vital,
                                                              short_keys=EverionKeys.short_names_vital,
                                                              min_scale=0, max_scale=100,
                                                              start_idx=0, end_idx=3
                                                              ))

        self.assertTrue(True)

    @unittest.SkipTest
    def test_cross_correlator_hours_mixed_raw_vital(self):
        in_dir = ''
        out_dir = FileHelper.get_out_dir(in_dir, '_cross')
        self.correlator.plot_hourly_correlations(VisProperties(in_dir=in_dir, out_dir=out_dir,
                                                               normalization=Normalization.NONE,
                                                               keys=EverionKeys.major_mixed_vital_raw,
                                                               short_keys=EverionKeys.short_names_mixed_vital_raw,
                                                               min_scale=0, max_scale=100,
                                                               start_idx=0, end_idx=3
                                                               ))

        self.assertTrue(True)

    @unittest.SkipTest
    def test_cross_correlator_days_mixed_vital_raw(self):
        in_dir = ''
        out_dir = FileHelper.get_out_dir(in_dir, '_daily_cross')
        self.correlator.plot_daily_correlations(VisProperties(in_dir=in_dir, out_dir=out_dir,
                                                              normalization=Normalization.NONE,
                                                              keys=EverionKeys.major_mixed_vital_raw,
                                                              short_keys=EverionKeys.short_names_mixed_vital_raw,
                                                              min_scale=0, max_scale=100,
                                                              start_idx=0, end_idx=3
                                                              ))

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
