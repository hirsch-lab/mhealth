import glob
import shutil
import unittest
import os

from utils.data_aggregator import Normalization
from visualization.plotter import Plotter
from visualization.vis_properties import VisProperties
from utils.file_helper import FileHelper
from utils import everion_keys

_MHEALTH_DATA = os.getenv("MHEALTH_DATA", "../resources")
_MHEALTH_OUT_DIR = os.path.join(_MHEALTH_DATA, 'output')


class PlotterTest(unittest.TestCase):
    in_dir = f'{_MHEALTH_DATA}/vital_signals/'
    out_dir = _MHEALTH_OUT_DIR
    plotter = Plotter()


    def test_plot_hourly_lines_subplots(self):
        out_dir = FileHelper.get_out_dir(in_dir=self.in_dir,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_hourly_lines2')
        props = VisProperties(in_dir=self.in_dir, out_dir=out_dir,
                              normalization=Normalization.NONE,
                              keys=everion_keys.MAJOR_VITAL,
                              short_keys=everion_keys.SHORT_NAMES_VITAL,
                              min_scale=0, max_scale=100,
                              start_idx=0, end_idx=3)
        self.plotter.plot_hourly_lines_subplots(properties=props)

        files = list(out_dir.glob('**/*.png'))
        self.assertEqual(3, len(files))


    def test_plot_hourly_lines(self):
        out_dir = FileHelper.get_out_dir(in_dir=self.in_dir,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_hourly_lines')
        props = VisProperties(in_dir=self.in_dir, out_dir=out_dir,
                              normalization=Normalization.NONE,
                              keys=everion_keys.MAJOR_VITAL,
                              short_keys=everion_keys.SHORT_NAMES_VITAL,
                              min_scale=0, max_scale=100,
                              start_idx=0, end_idx=3)
        self.plotter.plot_hourly_lines(properties=props)

        files = list(out_dir.glob('**/*.png'))
        self.assertEqual(3, len(files))


    def test_plot_one_signal(self):
        out_dir = FileHelper.get_out_dir(in_dir=self.in_dir,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_plots')
        if out_dir.is_dir():
            shutil.rmtree(out_dir)
        filename = '002_storage-sig.csv'
        self.plotter.plot_patient(in_dir=self.in_dir,
                                  out_dir=out_dir,
                                  in_file_name=filename)

        files = list(out_dir.glob('**/*.png'))
        self.assertEqual(8, len(files))


    def test_plot_one_signal_mixed_vital_raw(self):
        in_dir = f'{_MHEALTH_DATA}/mixed_vital_raw_signals/'
        out_dir = FileHelper.get_out_dir(in_dir=in_dir,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_plots')
        if out_dir.is_dir():
            shutil.rmtree(out_dir)

        filename = '001_Test_mixed_raw.csv'
        self.plotter.plot_patient_mixed_vital_raw(in_dir=in_dir,
                                                  out_dir=out_dir,
                                                  in_file_name=filename,
                                                  keys=everion_keys.MAJOR_MIXED_VITAL_RAW,
                                                  start_idx=0,
                                                  end_idx=3)

        files = list(out_dir.glob('**/*.png'))
        self.assertEqual(1, len(files))


    def test_plot_all_signals(self):
        out_dir = FileHelper.get_out_dir(in_dir=self.in_dir,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_plots')
        if out_dir.is_dir():
            shutil.rmtree(out_dir)
        for filename in os.listdir(self.in_dir):
            print('processing ', filename, ' ...')
            if not (filename.endswith('csv')):
                continue
            self.plotter.plot_patient(in_dir=self.in_dir,
                                      out_dir=out_dir,
                                      in_file_name=filename)

        files = list(out_dir.glob('**/*.png'))
        self.assertEqual(24, len(files))


    def test_plot_all_signals_mixed(self):
        in_dir = f'{_MHEALTH_DATA}/mixed_vital_raw_signals/'
        out_dir = FileHelper.get_out_dir(in_dir=in_dir,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_plots')
        if out_dir.is_dir():
            shutil.rmtree(out_dir)
        for filename in os.listdir(in_dir):
            print('processing ', filename, ' ...')
            if not (filename.endswith('csv')):
                continue
            self.plotter.plot_patient_mixed_vital_raw(in_dir=in_dir,
                                                      out_dir=out_dir,
                                                      in_file_name=filename,
                                                      keys=everion_keys.MAJOR_IMOVE,
                                                      start_idx=0,
                                                      end_idx=3)
        files = list(out_dir.glob('**/*.png'))
        self.assertEqual(5, len(files))


    @unittest.SkipTest
    def test_plot_signals_vital(self):
        dir_name = ''
        out_dir = FileHelper.get_out_dir(in_dir=dir_name,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_plots')
        for filename in os.listdir(dir_name):
            print('processing ', filename, ' ...')
            if not (filename.endswith('csv')):
                continue
            self.plotter.plot_patient(in_dir=dir_name,
                                      out_dir=out_dir,
                                      in_file_name=filename)

        self.assertTrue(True)


    @unittest.SkipTest
    def test_plot_all_signals_mixed_vital_raw(self):
        in_dir = ''
        out_dir = FileHelper.get_out_dir(in_dir=in_dir,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_plots')
        if out_dir.is_dir():
            shutil.rmtree(out_dir)
        for filename in os.listdir(in_dir):
            print('processing ', filename, ' ...')
            if not (filename.endswith('csv')):
                continue
            self.plotter.plot_patient_mixed_vital_raw(in_dir=in_dir,
                                                      out_dir=out_dir,
                                                      in_file_name=filename,
                                                      keys=everion_keys.MAJOR_IMOVE,
                                                      start_idx=0,
                                                      end_idx=3)


    @unittest.SkipTest
    def test_plot_hourly_lines_vital(self):
        in_dir = ''
        out_dir = FileHelper.get_out_dir(in_dir=in_dir,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_hourly_lines')
        props = VisProperties(in_dir=in_dir, out_dir=out_dir,
                              normalization=Normalization.NONE,
                              keys=everion_keys.MAJOR_VITAL,
                              short_keys=everion_keys.SHORT_NAMES_VITAL,
                              min_scale=0, max_scale=100,
                              start_idx=0, end_idx=3)
        self.plotter.plot_hourly_lines(properties=props)


    @unittest.SkipTest
    def test_plot_labels_mixed_vital_raw(self):
        in_dir = ''
        out_dir = FileHelper.get_out_dir(in_dir=in_dir,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_label_plots')
        props = VisProperties(in_dir=in_dir, out_dir=out_dir,
                              normalization=Normalization.NONE,
                              keys=everion_keys.MAJOR_VITAL,
                              short_keys=everion_keys.SHORT_NAMES_VITAL,
                              min_scale=0, max_scale=100,
                              start_idx=0, end_idx=3
                              )
        self.plotter.plot_signals_and_labels(properties=props)


if __name__ == '__main__':
    unittest.main()
