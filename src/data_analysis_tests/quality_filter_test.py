import shutil
import unittest
import os
import numpy as np

from data_analysis.quality_filter import QualityFilter
from patient.patient_data_loader import PatientDataLoader
from utils.file_helper import FileHelper

_MHEALTH_DATA = os.getenv('MHEALTH_DATA', '../resources')
_MHEALTH_OUT_DIR = os.path.join(_MHEALTH_DATA, 'output')


class QualityFilterTest(unittest.TestCase):
    in_dir_vital = f'{_MHEALTH_DATA}/vital_signals/'
    in_dir_mixed_vital_raw = f'{_MHEALTH_DATA}/mixed_vital_raw_signals/'
    out_dir = _MHEALTH_OUT_DIR

    loader = PatientDataLoader()
    filter = QualityFilter()
    quality = 50


    def test_filter_one_signal_mixed_vital_raw(self):
        out_dir_mixed_vital_raw = FileHelper.get_out_dir(
            in_dir=self.in_dir_mixed_vital_raw,
            out_dir=self.out_dir,
            out_dir_suffix='_quality_filtered_' + str(self.quality))

        if out_dir_mixed_vital_raw.is_dir():
            shutil.rmtree(out_dir_mixed_vital_raw)

        filename = '009_Test_mixed_raw.csv'
        self.filter.filter_bad_quality_mixed_vital_raw(
            in_dir=self.in_dir_mixed_vital_raw,
            out_dir=out_dir_mixed_vital_raw,
            in_file_name=filename,
            min_quality=self.quality)
        df_out = self.loader.load_everion_patient_data(
            dir_name=out_dir_mixed_vital_raw,
            filename=filename,
            csv_delimiter=';',
            tz_to_zurich=False)

        self.assertFalse(np.isnan(df_out['HR'][0]))
        self.assertTrue(np.isnan(df_out['HR'][2]))
        self.assertTrue(np.isnan(df_out['Activity'][2]))
        self.assertTrue(np.isnan(df_out['Classification'][2]))
        self.assertTrue(np.isnan(df_out['steps'][2]))
        self.assertTrue(np.isnan(df_out['pressure'][2]))
        self.assertTrue(np.isnan(df_out['Classification'][7]))
        self.assertEqual(48, df_out['QualityClassification'][7])
        self.assertEqual(79, df_out['HR'][0])
        self.assertEqual(77, df_out['HR'][10])
        self.assertEqual(77, df_out['HR'][17])


    def test_filter_one_signal_vital(self):
        out_dir_vital = FileHelper.get_out_dir(
            in_dir=self.in_dir_vital,
            out_dir=self.out_dir,
            out_dir_suffix='_quality_filtered_' + str(self.quality))
        if out_dir_vital.is_dir():
            shutil.rmtree(out_dir_vital)

        filename = '002_storage-sig.csv'
        self.filter.filter_bad_quality_vital(
            in_dir=self.in_dir_vital,
            out_dir=out_dir_vital,
            in_file_name=filename,
            min_quality=self.quality)
        df_out = self.loader.load_everion_patient_data(
            dir_name=out_dir_vital,
            filename=filename,
            csv_delimiter=";",
            tz_to_zurich=False)

        self.assertTrue(np.isnan(df_out['heart_rate'][0]))
        self.assertTrue(np.isnan(df_out['heart_rate'][14]))
        self.assertEqual(85, df_out['heart_rate'][16])
        self.assertEqual(82, df_out['heart_rate'][28])
        self.assertTrue(np.isnan(df_out['heart_rate_variability'][0]))
        self.assertTrue(np.isnan(df_out['heart_rate_variability'][28]))


    def test_filter_signals_vital(self):
        out_dir_vital = FileHelper.get_out_dir(
            in_dir=self.in_dir_vital,
            out_dir=self.out_dir,
            out_dir_suffix='_quality_filtered_' + str(self.quality))
        if os.path.exists(out_dir_vital):
            shutil.rmtree(out_dir_vital)

        for filename in os.listdir(self.in_dir_vital):
            print('processing ', filename, ' ...')
            if not (filename.endswith('csv')):
                continue
            self.filter.filter_bad_quality_vital(
                in_dir=self.in_dir_vital,
                out_dir=out_dir_vital,
                in_file_name=filename,
                min_quality=self.quality)

            if filename == '030_storage-sig.csv':
                self.assertFalse((out_dir_vital / filename).is_file())
            else:
                self.assertTrue((out_dir_vital / filename).is_file())


    @unittest.SkipTest
    def test_filter_signal_all_vital(self):
        quality = 50
        dir_name = ''
        out_dir = FileHelper.get_out_dir(
            in_dir=dir_name,
            out_dir=self.out_dir,
            out_dir_suffix='_quality_filtered_' + str(quality))

        for filename in os.listdir(dir_name):
            print('processing ', filename, ' ...')
            if not (filename.endswith('csv')):
                continue
            self.filter.filter_bad_quality_vital(in_dir=dir_name,
                                                 out_dir=out_dir,
                                                 in_file_name=filename,
                                                 min_quality=quality)

        self.assertTrue(True)


    @unittest.SkipTest
    def test_filter_signal_all_mixed_vital_raw(self):
        quality = 50
        dir_name = ''
        out_dir = FileHelper.get_out_dir(
            in_dir=dir_name,
            out_dir=self.out_dir,
            out_dir_suffix='_quality_filtered_' + str(quality))

        for filename in os.listdir(dir_name):
            print('processing ', filename, ' ...')
            if not (filename.endswith('csv')):
                continue
            self.filter.filter_bad_quality_mixed_vital_raw(
                in_dir=dir_name,
                out_dir=out_dir,
                in_file_name=filename,
                min_quality=quality)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
