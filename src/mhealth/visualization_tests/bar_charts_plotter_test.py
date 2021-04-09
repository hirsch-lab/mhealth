import glob
import os
import unittest

from ..utils import everion_keys
from ..utils.file_helper import FileHelper
from ..utils.data_aggregator import Normalization
from ..visualization.vis_properties import VisProperties
from ..visualization.bar_charts_plotter import BarChartsPlotter

_MHEALTH_DATA = os.getenv('MHEALTH_DATA', '../../resources')
_MHEALTH_OUT_DIR = os.path.join(_MHEALTH_DATA, 'output')


class BarChartsPlotterTest(unittest.TestCase):
    in_dir_vital = f'{_MHEALTH_DATA}/vital_signals/'
    in_dir_mixed_raw_vital = f'{_MHEALTH_DATA}/mixed_vital_raw_signals/'
    out_dir = _MHEALTH_OUT_DIR

    visualizer = BarChartsPlotter()

    def test_bar_charts_plotter_multiscale(self):
        out_dir = FileHelper.get_out_dir(in_dir=self.in_dir_vital,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_multiscale')
        props = VisProperties(in_dir=self.in_dir_vital, out_dir=out_dir,
                              normalization=Normalization.NONE,
                              keys={'respiration_rate'},
                              short_keys=everion_keys.SHORT_NAMES_VITAL,
                              min_scale=0, max_scale=100,
                              start_idx=0, end_idx=3)
        self.visualizer.plot_bars_multiscale(properties=props)

        files = list(out_dir.glob('**/*.png'))
        self.assertEqual(3, len(files))


    def test_bar_chart_plotter(self):
        out_dir = FileHelper.get_out_dir(in_dir=self.in_dir_vital,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_bars')
        props = VisProperties(in_dir=self.in_dir_vital, out_dir=out_dir,
                              normalization=Normalization.NONE,
                              keys=everion_keys.MAJOR_VITAL,
                              short_keys=everion_keys.SHORT_NAMES_VITAL,
                              min_scale=0, max_scale=100,
                              start_idx=0, end_idx=3)
        self.visualizer.plot_bars_hourly(properties=props)

        files = list(out_dir.glob('**/*.png'))
        self.assertEqual(3, len(files))

    def test_bar_chart_plotter_mixed_raw_vital(self):
        out_dir = FileHelper.get_out_dir(in_dir=self.in_dir_mixed_raw_vital,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_bars')
        props = VisProperties(in_dir=self.in_dir_mixed_raw_vital,
                              out_dir=out_dir,
                              normalization=Normalization.NONE,
                              keys=everion_keys.MAJOR_MIXED_VITAL_RAW,
                              short_keys=everion_keys.SHORT_NAMES_MIXED_VITAL_RAW,
                              min_scale=0, max_scale=100,
                              start_idx=0, end_idx=3)
        self.visualizer.plot_bars_hourly(properties=props)

        files = list(out_dir.glob('**/*.png'))
        self.assertEqual(5, len(files))


    @unittest.SkipTest
    def test_bar_charts_plotter_vital(self):
        in_dir = ''
        out_dir = FileHelper.get_out_dir(in_dir=in_dir,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_bars')
        props = VisProperties(in_dir=in_dir, out_dir=out_dir,
                              normalization=Normalization.NONE,
                              keys=everion_keys.MAJOR_VITAL,
                              short_keys=everion_keys.SHORT_NAMES_VITAL,
                              min_scale=0, max_scale=100,
                              start_idx=0, end_idx=3)
        self.visualizer.plot_bars_hourly(properties=props)

        self.assertTrue(True)


    @unittest.SkipTest
    def test_bar_charts_multiscale_vital(self):
        in_dir = ''
        out_dir = FileHelper.get_out_dir(in_dir=in_dir,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_multiscale')
        props = VisProperties(in_dir=in_dir, out_dir=out_dir,
                              normalization=Normalization.NONE,
                              keys={'respiration_rate'},
                              short_keys=everion_keys.SHORT_NAMES_VITAL,
                              min_scale=0, max_scale=100,
                              start_idx=0, end_idx=3)
        self.visualizer.plot_bars_multiscale(properties=props)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
