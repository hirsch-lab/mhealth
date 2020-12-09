import glob
import os
import unittest

from utils.data_aggregator import Normalization
from visualization.bar_charts_plotter import BarChartsPlotter
from visualization.vis_properties import VisProperties
from utils.everion_keys import EverionKeys
from utils.file_helper import FileHelper


class BarChartsPlotterTest(unittest.TestCase):
    in_dir_vital = '../resources/vital_signals/'
    in_dir_mixed_raw_vital = '../resources/mixed_vital_raw_signals/'

    visualizer = BarChartsPlotter()

    def test_bar_charts_plotter_multiscale(self):
        out_dir = FileHelper.get_out_dir(self.in_dir_vital, '_multiscale')
        self.visualizer.plot_bars_multiscale(VisProperties(in_dir=self.in_dir_vital, out_dir=out_dir,
                                                           normalization=Normalization.NONE,
                                                           keys={'respiration_rate'},
                                                           short_keys=EverionKeys.short_names_vital,
                                                           min_scale=0, max_scale=100,
                                                           start_idx=0, end_idx=3
                                                           ))

        files = glob.glob(os.path.join(os.path.join(out_dir, '**'), '*.png'), recursive=True)
        self.assertEqual(3, len(files))


    def test_bar_chart_plotter(self):
        out_dir = FileHelper.get_out_dir(self.in_dir_vital, '_bars')
        self.visualizer.plot_bars_hourly(VisProperties(in_dir=self.in_dir_vital, out_dir=out_dir,
                                                       normalization=Normalization.NONE,
                                                       keys=EverionKeys.major_vital,
                                                       short_keys=EverionKeys.short_names_vital,
                                                       min_scale=0, max_scale=100,
                                                       start_idx=0, end_idx=3
                                                       ))

        files = glob.glob(os.path.join(os.path.join(out_dir, '**'), '*.png'), recursive=True)
        self.assertEqual(3, len(files))

    def test_bar_chart_plotter_mixed_raw_vital(self):
        out_dir = FileHelper.get_out_dir(self.in_dir_mixed_raw_vital, '_bars')
        self.visualizer.plot_bars_hourly(VisProperties(in_dir=self.in_dir_mixed_raw_vital, out_dir=out_dir,
                                                       normalization=Normalization.NONE,
                                                       keys=EverionKeys.major_mixed_vital_raw,
                                                       short_keys=EverionKeys.short_names_mixed_vital_raw,
                                                       min_scale=0, max_scale=100,
                                                       start_idx=0, end_idx=3
                                                       ))

        files = glob.glob(os.path.join(os.path.join(out_dir, '**'), '*.png'), recursive=True)
        self.assertEqual(5, len(files))


    @unittest.SkipTest
    def test_bar_charts_plotter_vital(self):
        in_dir = ''
        out_dir = FileHelper.get_out_dir(in_dir, '_bars')
        self.visualizer.plot_bars_hourly(VisProperties(in_dir=in_dir, out_dir=out_dir,
                                                       normalization=Normalization.NONE,
                                                       keys=EverionKeys.major_vital,
                                                       short_keys=EverionKeys.short_names_vital,
                                                       min_scale=0, max_scale=100,
                                                       start_idx=0, end_idx=3
                                                       ))

        self.assertTrue(True)

    @unittest.SkipTest
    def test_bar_charts_multiscale_vital(self):
        in_dir = ''
        out_dir = FileHelper.get_out_dir(in_dir, '_multiscale')
        self.visualizer.plot_bars_multiscale(VisProperties(in_dir=in_dir, out_dir=out_dir,
                                                           normalization=Normalization.NONE,
                                                           keys={'respiration_rate'},
                                                           short_keys=EverionKeys.short_names_vital,
                                                           min_scale=0, max_scale=100,
                                                           start_idx=0, end_idx=3
                                                           ))

        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
