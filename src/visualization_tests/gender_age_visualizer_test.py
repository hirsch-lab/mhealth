import glob
import os
import unittest

from visualization.gender_age_visualizer import GenderAgeVisualizer
from patient.patient_data_loader import PatientDataLoader
from utils.everion_keys import EverionKeys
from utils.file_helper import FileHelper


class GenderVisualizerTest(unittest.TestCase):

    def test_plot_age(self):
        in_dir = '../resources/vital_signals/'
        extra_data_dir_name = '../resources/extra_data/'

        out_dir = FileHelper.get_out_dir(in_dir, '_gender_plot')
        plotter = GenderAgeVisualizer()
        lookup_table = PatientDataLoader.load_extra_patient_data(os.path.join(extra_data_dir_name, 'extra_data.csv'))

        plotter.plot_data(in_dir, out_dir, 0, 3, lookup_table, EverionKeys.major_vital, EverionKeys.short_names_vital)

        files = glob.glob(os.path.join(os.path.join(out_dir, '**'), '*.png'), recursive=True)
        self.assertEqual(1, len(files))

    @unittest.SkipTest
    def test_plot_age_vital(self):
        in_dir = ''
        extra_data_dir_name = ''

        out_dir = FileHelper.get_out_dir(in_dir, '_gender_plot')
        plotter = GenderAgeVisualizer()
        lookup_table = PatientDataLoader.load_extra_patient_data(os.path.join(extra_data_dir_name, 'extra_data.csv'))

        plotter.plot_data(in_dir, out_dir, 0, 3, lookup_table, EverionKeys.major_vital, EverionKeys.short_names_vital)

        files = glob.glob(os.path.join(os.path.join(out_dir, '**'), '*.png'), recursive=True)
        self.assertEqual(1, len(files))

    @unittest.SkipTest
    def test_plot_age_mixed_vital_raw(self):
        in_dir = ''
        extra_data_dir_name = ''

        out_dir = FileHelper.get_out_dir(in_dir, '_gender_plot2')
        plotter = GenderAgeVisualizer()
        lookup_table = PatientDataLoader.load_extra_patient_data(os.path.join(extra_data_dir_name, '.csv'))

        plotter.plot_data(in_dir, out_dir, 0, 3, lookup_table, EverionKeys.major_mixed_vital_raw,
                          EverionKeys.short_names_mixed_vital_raw)

        files = glob.glob(os.path.join(os.path.join(out_dir, '**'), '*.png'), recursive=True)
        self.assertEqual(1, len(files))


if __name__ == '__main__':
    unittest.main()
