import glob
import shutil
import unittest
import os

from utils.data_aggregator import Normalization
from visualization.plotter import Plotter
from visualization.vis_properties import VisProperties
from utils.everion_keys import EverionKeys
from utils.file_helper import FileHelper


class PlotterTest(unittest.TestCase):
    in_dir = '../resources/vital_signals/'
    plotter = Plotter()
    out_dir = FileHelper.get_out_dir(in_dir, '_plots')

    def test_plot_hourly_lines_subplots(self):
        out_dir = FileHelper.get_out_dir(self.in_dir, '_hourly_lines2')
        self.plotter.plot_hourly_lines_subplots(VisProperties(in_dir=self.in_dir, out_dir=out_dir,
                                                              normalization=Normalization.NONE,
                                                              keys=EverionKeys.major_vital,
                                                              short_keys=EverionKeys.short_names_vital,
                                                              min_scale=0, max_scale=100,
                                                              start_idx=0, end_idx=3
                                                              ))

        files = glob.glob(os.path.join(os.path.join(out_dir, '**'), '*.png'), recursive=True)
        self.assertEqual(3, len(files))

    def test_plot_hourly_lines(self):
        out_dir = FileHelper.get_out_dir(self.in_dir, '_hourly_lines')
        self.plotter.plot_hourly_lines(VisProperties(in_dir=self.in_dir, out_dir=out_dir,
                                                     normalization=Normalization.NONE,
                                                     keys=EverionKeys.major_vital,
                                                     short_keys=EverionKeys.short_names_vital,
                                                     min_scale=0, max_scale=100,
                                                     start_idx=0, end_idx=3
                                                     ))

        files = glob.glob(os.path.join(os.path.join(out_dir, '**'), '*.png'), recursive=True)
        self.assertEqual(3, len(files))

    def test_plot_one_signal(self):
        if os.path.exists(self.out_dir):
            shutil.rmtree(self.out_dir)
        filename = '002_storage-sig.csv'
        self.plotter.plot_patient(self.in_dir, self.out_dir, filename)

        files = glob.glob(os.path.join(os.path.join(self.out_dir, '**'), '*.png'), recursive=True)
        self.assertEqual(8, len(files))

    def test_plot_one_signal_mixed_vital_raw(self):
        in_dir = '../resources/mixed_vital_raw_signals/'
        out_dir = FileHelper.get_out_dir(in_dir, '_plots')

        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)

        filename = '001_Test_mixed_raw.csv'
        self.plotter.plot_patient_mixed_vital_raw(in_dir, out_dir, filename, EverionKeys.major_mixed_vital_raw, 0, 3)

        files = glob.glob(os.path.join(os.path.join(out_dir, '**'), '*.png'), recursive=True)
        self.assertEqual(1, len(files))

    def test_plot_all_signals(self):
        if os.path.exists(self.out_dir):
            shutil.rmtree(self.out_dir)
        for filename in os.listdir(self.in_dir):
            print('processing ', filename, ' ...')
            if not (filename.endswith('csv')):
                continue
            self.plotter.plot_patient(self.in_dir, self.out_dir, filename)

        files = glob.glob(os.path.join(os.path.join(self.out_dir, '**'), '*.png'), recursive=True)
        self.assertEqual(24, len(files))

    def test_plot_all_signals_mixed(self):
        in_dir = '../resources/mixed_vital_raw_signals/'
        out_dir = FileHelper.get_out_dir(in_dir, '_plots')

        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        for filename in os.listdir(in_dir):
            print('processing ', filename, ' ...')
            if not (filename.endswith('csv')):
                continue
            self.plotter.plot_patient_mixed_vital_raw(in_dir, out_dir, filename, EverionKeys.major_imove, 0, 3)

        files = glob.glob(os.path.join(os.path.join(out_dir, '**'), '*.png'), recursive=True)
        self.assertEqual(5, len(files))

    @unittest.SkipTest
    def test_plot_signals_vital(self):
        dir_name = ''
        out_dir = FileHelper.get_out_dir(dir_name, '_plots')
        for filename in os.listdir(dir_name):
            print('processing ', filename, ' ...')
            if not (filename.endswith('csv')):
                continue
            self.plotter.plot_patient(dir_name, out_dir, filename)

        self.assertTrue(True)

    @unittest.SkipTest
    def test_plot_all_signals_mixed_vital_raw(self):
        in_dir = ''
        out_dir = FileHelper.get_out_dir(in_dir, '_plots')

        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        for filename in os.listdir(in_dir):
            print('processing ', filename, ' ...')
            if not (filename.endswith('csv')):
                continue
            self.plotter.plot_patient_mixed_vital_raw(in_dir, out_dir, filename, EverionKeys.major_imove, 0, 3)

    @unittest.SkipTest
    def test_plot_hourly_lines_vital(self):
        in_dir = ''
        out_dir = FileHelper.get_out_dir(in_dir, '_hourly_lines')
        self.plotter.plot_hourly_lines(VisProperties(in_dir=in_dir, out_dir=out_dir,
                                                     normalization=Normalization.NONE,
                                                     keys=EverionKeys.major_vital,
                                                     short_keys=EverionKeys.short_names_vital,
                                                     min_scale=0, max_scale=100,
                                                     start_idx=0, end_idx=3
                                                     ))


if __name__ == '__main__':
    unittest.main()
