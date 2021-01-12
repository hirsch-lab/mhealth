import glob
import os
import unittest

from visualization.gender_age_visualizer import GenderAgeVisualizer
from patient.patient_data_loader import PatientDataLoader
from utils.everion_keys import EverionKeys
from utils.file_helper import FileHelper

_MHEALTH_DATA = os.getenv('MHEALTH_DATA', '../resources')
_MHEALTH_OUT_DIR = os.path.join(_MHEALTH_DATA, 'output')


class GenderVisualizerTest(unittest.TestCase):
    out_dir = _MHEALTH_OUT_DIR

    def test_plot_age(self):
        in_dir = f'{_MHEALTH_DATA}/vital_signals/'
        extra_data_dir_name = f'{_MHEALTH_DATA}/extra_data/'

        out_dir = FileHelper.get_out_dir(in_dir=in_dir,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_gender_plot')

        filepath = os.path.join(extra_data_dir_name, 'extra_data.csv')
        lookup_table = PatientDataLoader.load_extra_patient_data(filepath)

        plotter = GenderAgeVisualizer()
        plotter.plot_data(in_dir=in_dir,
                          out_dir=out_dir,
                          start_idx=0,
                          end_idx=3,
                          lookup_table=lookup_table,
                          keys=EverionKeys.major_vital,
                          short_keys=EverionKeys.short_names_vital)

        files = list(out_dir.glob('**/*.png'))
        self.assertEqual(1, len(files))


    @unittest.SkipTest
    def test_plot_age_vital(self):
        in_dir = ''
        extra_data_dir_name = ''
        out_dir = FileHelper.get_out_dir(in_dir=in_dir,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_gender_plot')
        filepath = os.path.join(extra_data_dir_name, 'extra_data.csv')
        lookup_table = PatientDataLoader.load_extra_patient_data(filepath)
        plotter = GenderAgeVisualizer()
        plotter.plot_data(in_dir=in_dir,
                          out_dir=out_dir,
                          start_idx=0,
                          end_idx=3,
                          lookup_table=lookup_table,
                          keys=EverionKeys.major_vital,
                          short_keys=EverionKeys.short_names_vital)

        files = list(out_dir.glob('**/*.png'))
        self.assertEqual(1, len(files))


    @unittest.SkipTest
    def test_plot_age_mixed_vital_raw(self):
        in_dir = ''
        extra_data_dir_name = ''
        out_dir = FileHelper.get_out_dir(in_dir=in_dir,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_gender_plot2')
        filepath = os.path.join(extra_data_dir_name, '.csv')
        lookup_table = PatientDataLoader.load_extra_patient_data(filepath)
        plotter = GenderAgeVisualizer()
        plotter.plot_data(in_dir=in_dir,
                          out_dir=out_dir,
                          start_idx=0,
                          end_idx=3,
                          lookup_table=lookup_table,
                          keys=EverionKeys.major_mixed_vital_raw,
                          short_keys=EverionKeys.short_names_mixed_vital_raw)

        files = list(out_dir.glob('**/*.png'))
        self.assertEqual(1, len(files))


if __name__ == '__main__':
    unittest.main()
