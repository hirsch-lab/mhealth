import os
import glob
import unittest

from ..utils import testing
from ..utils import everion_keys
from ..utils.file_helper import FileHelper
from ..utils.data_aggregator import Normalization
from ..visualization.vis_properties import VisProperties
from ..data_analysis.cross_correlator import CrossCorrelator

_MHEALTH_DATA = os.getenv('MHEALTH_DATA', '../../resources')
_MHEALTH_OUT_DIR = os.path.join(_MHEALTH_DATA, 'output')


class CrossCorrelatorTest(testing.TestCase):
    in_dir = f'{_MHEALTH_DATA}/vital_signals/'
    in_dir_mixed = f'{_MHEALTH_DATA}/mixed_vital_raw_signals/'
    out_dir = _MHEALTH_OUT_DIR
    correlator = CrossCorrelator()

    def test_cross_correlator_daily(self):
        out_dir = FileHelper.get_out_dir(in_dir=self.in_dir,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_daily_cross')
        self.correlator.plot_daily_correlations(
            VisProperties(in_dir=self.in_dir, out_dir=out_dir,
                          normalization=Normalization.NONE,
                          keys={'heart_rate', 'respiration_rate'},
                          short_keys=everion_keys.SHORT_NAMES_VITAL,
                          min_scale=0, max_scale=100,
                          start_idx=0, end_idx=3))

        files = list(out_dir.glob("**/*.png"))
        self.assertEqual(6, len(files))

    def test_cross_correlator_hours_vital(self):
        out_dir = FileHelper.get_out_dir(in_dir=self.in_dir,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_cross')
        self.correlator.plot_hourly_correlations(
            VisProperties(in_dir=self.in_dir, out_dir=out_dir,
                          normalization=Normalization.NONE,
                          keys=everion_keys.MAJOR_VITAL,
                          short_keys=everion_keys.SHORT_NAMES_VITAL,
                          min_scale=0, max_scale=100,
                          start_idx=0, end_idx=3))

        files = list(out_dir.glob("**/*.png"))
        self.assertEqual(3, len(files))

    def test_cross_correlator_hours_mixed_vital_raw(self):
        out_dir = FileHelper.get_out_dir(in_dir=self.in_dir_mixed,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_cross')
        self.correlator.plot_hourly_correlations(
            VisProperties(in_dir=self.in_dir_mixed, out_dir=out_dir,
                          normalization=Normalization.NONE,
                          keys=everion_keys.MAJOR_MIXED_VITAL_RAW,
                          short_keys=everion_keys.SHORT_NAMES_MIXED_VITAL_RAW,
                          min_scale=0, max_scale=100,
                          start_idx=0, end_idx=3))

        files = list(out_dir.glob("**/*.png"))
        self.assertEqual(5, len(files))

    def test_cross_correlator_daily_mixed_vital_raw(self):
        out_dir = FileHelper.get_out_dir(in_dir=self.in_dir_mixed,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_daily_cross')
        self.correlator.plot_daily_correlations(
            VisProperties(in_dir=self.in_dir_mixed, out_dir=out_dir,
                          normalization=Normalization.NONE,
                          keys=everion_keys.MAJOR_MIXED_VITAL_RAW,
                          short_keys=everion_keys.SHORT_NAMES_MIXED_VITAL_RAW,
                          min_scale=0, max_scale=100,
                          start_idx=0, end_idx=3))

        files = list(out_dir.glob("**/*.png"))
        self.assertEqual(25, len(files))

    @testing.skip_because_is_runner
    def test_cross_hours_vital(self):
        in_dir = ''
        out_dir = FileHelper.get_out_dir(in_dir=in_dir,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_cross')
        self.correlator.plot_hourly_correlations(
            VisProperties(in_dir=in_dir, out_dir=out_dir,
                          normalization=Normalization.NONE,
                          keys=everion_keys.MAJOR_VITAL,
                          short_keys=everion_keys.SHORT_NAMES_VITAL,
                          min_scale=0, max_scale=100,
                          start_idx=0, end_idx=3))

        self.assertTrue(True)

    @testing.skip_because_is_runner
    def test_cross_days_vital(self):
        in_dir = ''
        out_dir = FileHelper.get_out_dir(in_dir=in_dir,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_daily_cross')
        self.correlator.plot_daily_correlations(
            VisProperties(in_dir=in_dir, out_dir=out_dir,
                          normalization=Normalization.NONE,
                          keys=everion_keys.MAJOR_VITAL,
                          short_keys=everion_keys.SHORT_NAMES_VITAL,
                          min_scale=0, max_scale=100,
                          start_idx=0, end_idx=3))

        self.assertTrue(True)

    @testing.skip_because_is_runner
    def test_cross_correlator_hours_mixed_raw_vital(self):
        in_dir = ''
        out_dir = FileHelper.get_out_dir(in_dir=in_dir,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_cross')
        self.correlator.plot_hourly_correlations(
            VisProperties(in_dir=in_dir, out_dir=out_dir,
                          normalization=Normalization.NONE,
                          keys=everion_keys.MAJOR_MIXED_VITAL_RAW,
                          short_keys=everion_keys.SHORT_NAMES_MIXED_VITAL_RAW,
                          min_scale=0, max_scale=100,
                          start_idx=0, end_idx=3))

        self.assertTrue(True)

    @testing.skip_because_is_runner
    def test_cross_correlator_days_mixed_vital_raw(self):
        in_dir = ''
        out_dir = FileHelper.get_out_dir(in_dir=in_dir,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_daily_cross')
        self.correlator.plot_daily_correlations(
            VisProperties(in_dir=in_dir, out_dir=out_dir,
                          normalization=Normalization.NONE,
                          keys=everion_keys.MAJOR_MIXED_VITAL_RAW,
                          short_keys=everion_keys.SHORT_NAMES_MIXED_VITAL_RAW,
                          min_scale=0, max_scale=100,
                          start_idx=0, end_idx=3))

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
