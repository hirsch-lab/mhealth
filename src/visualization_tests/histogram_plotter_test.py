import glob
import os
import shutil
import unittest

from visualization.histogram_plotter import HistogramPlotter
from patient.patient_data_loader import PatientDataLoader
from utils.everion_keys import EverionKeys
from utils.file_helper import FileHelper

_MHEALTH_DATA = os.getenv('MHEALTH_DATA', '../resources')


class HistogramPlotterTest(unittest.TestCase):
    in_dir = f'{_MHEALTH_DATA}/vital_signals/'
    out_dir = FileHelper.get_out_dir(in_dir, '_histograms')


    plotter = HistogramPlotter()


    def test_plot_all_histograms_vital(self):
        if os.path.exists(self.out_dir):
            shutil.rmtree(self.out_dir)
        self.plotter.plot_all_histograms(self.in_dir, self.out_dir, 0, 3, EverionKeys.all_vital)

        files = glob.glob(os.path.join(os.path.join(self.out_dir, '**'), '*.png'), recursive=True)
        self.assertEqual(66, len(files))

    def test_plot_all_histograms_mixed_raw_vital(self):
        in_dir = f'{_MHEALTH_DATA}/mixed_vital_raw_signals/'
        out_dir = FileHelper.get_out_dir(in_dir, '_histograms')

        self.plotter.plot_all_histograms(in_dir, out_dir, 0, 3, EverionKeys.major_mixed_vital_raw)

        files = glob.glob(os.path.join(os.path.join(out_dir, '**'), '*.png'), recursive=True)
        self.assertEqual(25, len(files))


    def test_plot_histogram_one_patient_vital(self):
        if os.path.exists(self.out_dir):
            shutil.rmtree(self.out_dir)

        filename = '002_storage-sig.csv'
        loader = PatientDataLoader()
        df = loader.load_everion_patient_data(self.in_dir, filename,';')
        if df.empty:
            self.assertFalse()

        self.plotter.plot_histogram(self.out_dir, '002', df, EverionKeys.all_vital)

        files = glob.glob(os.path.join(os.path.join(self.out_dir, '**'), '*.png'), recursive=True)
        self.assertEqual(22, len(files))

    @unittest.SkipTest
    def test_plot_histogram_vital(self):
        in_dir = ''
        out_dir = FileHelper.get_out_dir(in_dir, '_histograms')

        self.plotter.plot_histogram(in_dir, out_dir, EverionKeys.all_vital)
        self.assertTrue(True)

    @unittest.SkipTest
    def test_plot_all_histograms_mixed_vital_raw(self):
        in_dir = ''
        out_dir = FileHelper.get_out_dir(in_dir, '_histograms')

        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        self.plotter.plot_all_histograms(in_dir, out_dir, 15, 39, EverionKeys.major_mixed_vital_raw)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
