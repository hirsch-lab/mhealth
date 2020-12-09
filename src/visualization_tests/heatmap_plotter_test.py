import glob
import os
import shutil
import unittest

from utils.data_aggregator import Normalization
from visualization.heatmap_plotter import HeatmapPlotter
from visualization.vis_properties import VisProperties
from utils.everion_keys import EverionKeys
from utils.file_helper import FileHelper


class HeatmapPlotterTest(unittest.TestCase):
    in_dir_vital = '../resources/vital_signals/'
    in_dir_mixed_raw_vital = '../resources/mixed_vital_raw_signals/'
    plotter = HeatmapPlotter()

    def test_plot_heatmaps_all_patients_days(self):
        out_dir = FileHelper.get_out_dir(self.in_dir_vital, '_heatmaps')
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
            os.mkdir(out_dir)

        self.plotter.plot_heatmaps(VisProperties(in_dir=self.in_dir_vital, out_dir=out_dir,
                                                 normalization=Normalization.NONE,
                                                 keys=EverionKeys.major_vital,
                                                 short_keys=EverionKeys.short_names_vital,
                                                 min_scale=0, max_scale=100,
                                                 start_idx=0, end_idx=3
                                                 ))

        files = glob.glob(os.path.join(os.path.join(out_dir, '**'), '*.png'), recursive=True)
        self.assertEqual(3, len(files))

    def test_plot_heatmaps_all_patients_days_mixed_raw_vital(self):
        out_dir = FileHelper.get_out_dir(self.in_dir_mixed_raw_vital, '_heatmaps')
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
            os.mkdir(out_dir)

        self.plotter.plot_heatmaps(VisProperties(in_dir=self.in_dir_mixed_raw_vital, out_dir=out_dir,
                                                 normalization=Normalization.NONE,
                                                 keys=EverionKeys.major_mixed_vital_raw,
                                                 short_keys=EverionKeys.short_names_mixed_vital_raw,
                                                 min_scale=0, max_scale=100,
                                                 start_idx=0, end_idx=3
                                                 ))

        files = glob.glob(os.path.join(os.path.join(out_dir, '**'), '*.png'), recursive=True)
        self.assertEqual(5, len(files))

    def test_plot_heatmaps_all_patients_days_maxnorm(self):
        out_dir = FileHelper.get_out_dir(self.in_dir_vital, '_heatmaps-maxnorm')
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
            os.mkdir(out_dir)

        self.plotter.plot_heatmaps(VisProperties(in_dir=self.in_dir_vital, out_dir=out_dir,
                                                 normalization=Normalization.MAX_NORM,
                                                 keys=EverionKeys.major_vital,
                                                 short_keys=EverionKeys.short_names_vital,
                                                 min_scale=0, max_scale=1,
                                                 start_idx=0, end_idx=3
                                                 ))

        files = glob.glob(os.path.join(os.path.join(out_dir, '**'), '*.png'), recursive=True)
        self.assertEqual(3, len(files))



    @unittest.SkipTest
    def test_plot_heatmaps_mixed_vital_raw(self):
        in_dir = ''
        out_dir = FileHelper.get_out_dir(in_dir, '_heatmaps')
        self.plotter.plot_heatmaps(VisProperties(in_dir=in_dir, out_dir=out_dir,
                                                 normalization=Normalization.NONE,
                                                 keys=EverionKeys.major_mixed_vital_raw,
                                                 short_keys=EverionKeys.short_names_mixed_vital_raw,
                                                 min_scale=0, max_scale=100,
                                                 start_idx=15, end_idx=19
                                                 ))
        self.assertTrue(True)


    @unittest.SkipTest
    def test_plot_heatmaps_vital(self):
        in_dir = ''
        out_dir = FileHelper.get_out_dir(in_dir, '_heatmaps')
        self.plotter.plot_heatmaps(VisProperties(in_dir=in_dir, out_dir=out_dir,
                                                 normalization=Normalization.NONE,
                                                 keys=EverionKeys.major_vital,
                                                 short_keys=EverionKeys.short_names_vital,
                                                 min_scale=0, max_scale=100,
                                                 start_idx=0, end_idx=3
                                                 ))
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
