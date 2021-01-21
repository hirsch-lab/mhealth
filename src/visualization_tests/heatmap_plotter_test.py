import glob
import os
import shutil
import unittest

from utils.data_aggregator import Normalization
from visualization.heatmap_plotter import HeatmapPlotter
from visualization.vis_properties import VisProperties
from utils.everion_keys import EverionKeys
from utils.file_helper import FileHelper

_MHEALTH_DATA = os.getenv('MHEALTH_DATA', '../resources')
_MHEALTH_OUT_DIR = os.path.join(_MHEALTH_DATA, 'output')


class HeatmapPlotterTest(unittest.TestCase):
    in_dir_vital = f'{_MHEALTH_DATA}/vital_signals/'
    in_dir_mixed_raw_vital = f'{_MHEALTH_DATA}/mixed_vital_raw_signals/'
    out_dir = _MHEALTH_OUT_DIR
    plotter = HeatmapPlotter()

    def test_plot_heatmaps_all_patients_days(self):
        out_dir = FileHelper.get_out_dir(in_dir=self.in_dir_vital,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_heatmaps')
        if out_dir.is_dir():
            shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True)

        props = VisProperties(in_dir=self.in_dir_vital, out_dir=out_dir,
                              normalization=Normalization.NONE,
                              keys=EverionKeys.major_vital,
                              short_keys=EverionKeys.short_names_vital,
                              min_scale=0, max_scale=100,
                              start_idx=0, end_idx=3)
        self.plotter.plot_heatmaps(properties=props)

        files = list(out_dir.glob('**/*.png'))
        self.assertEqual(3, len(files))


    def test_plot_heatmaps_all_patients_days_mixed_raw_vital(self):
        out_dir = FileHelper.get_out_dir(in_dir=self.in_dir_mixed_raw_vital,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_heatmaps')
        if out_dir.is_dir():
            shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True)

        props = VisProperties(in_dir=self.in_dir_mixed_raw_vital, out_dir=out_dir,
                              normalization=Normalization.NONE,
                              keys=EverionKeys.major_mixed_vital_raw,
                              short_keys=EverionKeys.short_names_mixed_vital_raw,
                              min_scale=0, max_scale=100,
                              start_idx=0, end_idx=3)
        self.plotter.plot_heatmaps(properties=props)

        files = list(out_dir.glob('**/*.png'))
        self.assertEqual(5, len(files))


    def test_plot_heatmaps_all_patients_days_maxnorm(self):
        out_dir = FileHelper.get_out_dir(in_dir=self.in_dir_vital,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_heatmaps-maxnorm')
        if out_dir.is_dir():
            shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True)

        props = VisProperties(in_dir=self.in_dir_vital, out_dir=out_dir,
                              normalization=Normalization.MAX_NORM,
                              keys=EverionKeys.major_vital,
                              short_keys=EverionKeys.short_names_vital,
                              min_scale=0, max_scale=1,
                              start_idx=0, end_idx=3)
        self.plotter.plot_heatmaps(properties=props)

        files = list(out_dir.glob('**/*.png'))
        self.assertEqual(3, len(files))


    @unittest.SkipTest
    def test_plot_heatmaps_mixed_vital_raw(self):
        in_dir = ''
        out_dir = FileHelper.get_out_dir(in_dir=in_dir,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_heatmaps')
        props = VisProperties(in_dir=in_dir, out_dir=out_dir,
                              normalization=Normalization.NONE,
                              keys=EverionKeys.major_mixed_vital_raw,
                              short_keys=EverionKeys.short_names_mixed_vital_raw,
                              min_scale=0, max_scale=100,
                              start_idx=15, end_idx=19
                              )
        self.plotter.plot_heatmaps(properties=props)
        self.assertTrue(True)


    @unittest.SkipTest
    def test_plot_heatmaps_vital(self):
        in_dir = ''
        out_dir = FileHelper.get_out_dir(in_dir=in_dir,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_heatmaps')
        props = VisProperties(in_dir=in_dir, out_dir=out_dir,
                              normalization=Normalization.NONE,
                              keys=EverionKeys.major_vital,
                              short_keys=EverionKeys.short_names_vital,
                              min_scale=0, max_scale=100,
                              start_idx=0, end_idx=3
                              )
        self.plotter.plot_heatmaps(properties=props)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
