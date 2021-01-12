import glob
import os
import shutil
import unittest

from visualization.histogram_plotter import HistogramPlotter
from patient.patient_data_loader import PatientDataLoader
from utils.everion_keys import EverionKeys
from utils.file_helper import FileHelper

_MHEALTH_DATA = os.getenv('MHEALTH_DATA', '../resources')
_MHEALTH_OUT_DIR = os.path.join(_MHEALTH_DATA, 'output')


class HistogramPlotterTest(unittest.TestCase):
    out_dir = _MHEALTH_OUT_DIR
    plotter = HistogramPlotter()


    def test_plot_all_histograms_vital(self):
        in_dir = f'{_MHEALTH_DATA}/vital_signals/'
        out_dir = FileHelper.get_out_dir(in_dir=in_dir,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_histograms')
        if out_dir.is_dir():
            shutil.rmtree(out_dir)
        self.plotter.plot_all_histograms(in_dir=in_dir,
                                         out_dir=out_dir,
                                         start_idx=0,
                                         end_idx=3,
                                         keys=EverionKeys.all_vital)

        files = list(out_dir.glob('**/*.png'))
        self.assertEqual(66, len(files))


    def test_plot_all_histograms_mixed_raw_vital(self):
        in_dir = f'{_MHEALTH_DATA}/mixed_vital_raw_signals/'
        out_dir = FileHelper.get_out_dir(in_dir=in_dir,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_histograms')

        self.plotter.plot_all_histograms(in_dir=in_dir,
                                         out_dir=out_dir,
                                         start_idx=0,
                                         end_idx=3,
                                         keys=EverionKeys.major_mixed_vital_raw)

        files = list(out_dir.glob('**/*.png'))
        self.assertEqual(25, len(files))


    def test_plot_histogram_one_patient_vital(self):
        in_dir = f'{_MHEALTH_DATA}/vital_signals/'
        out_dir = FileHelper.get_out_dir(in_dir=in_dir,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_histograms')
        if out_dir.is_dir():
            shutil.rmtree(out_dir)

        filename = '002_storage-sig.csv'
        loader = PatientDataLoader()
        df = loader.load_everion_patient_data(dir_name=in_dir,
                                              filename=filename,
                                              csv_delimiter=';')
        if df.empty:
            self.assertFalse()

        self.plotter.plot_histogram(out_dir=out_dir,
                                    patient_id='002',
                                    df=df,
                                    keys=EverionKeys.all_vital)

        files = list(out_dir.glob('**/*.png'))
        self.assertEqual(22, len(files))


    @unittest.SkipTest
    def test_plot_histogram_vital(self):
        in_dir = ''
        out_dir = FileHelper.get_out_dir(in_dir=in_dir,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_histograms')
        self.plotter.plot_histogram(in_dir=in_dir,
                                    out_dir=out_dir,
                                    keys=EverionKeys.all_vital)
        self.assertTrue(True)


    @unittest.SkipTest
    def test_plot_all_histograms_mixed_vital_raw(self):
        in_dir = ''
        out_dir = FileHelper.get_out_dir(in_dir=in_dir,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_histograms')

        if out_dir.is_dir():
            shutil.rmtree(out_dir)
        self.plotter.plot_all_histograms(in_dir=in_dir,
                                         out_dir=out_dir,
                                         start_idx=15,
                                         end_idx=39,
                                         keys=EverionKeys.major_mixed_vital_raw)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
