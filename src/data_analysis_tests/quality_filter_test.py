import shutil
import unittest
import os
import numpy as np

from data_analysis.quality_filter import QualityFilter
from patient.patient_data_loader import PatientDataLoader
from utils.file_helper import FileHelper


class QualityFilterTest(unittest.TestCase):
        in_dir_vital = '../resources/vital_signals/'
        in_dir_mixed_vital_raw = '../resources/mixed_vital_raw_signals/'

        filter = QualityFilter()
        quality = 50
        out_dir_vital = FileHelper.get_out_dir(in_dir_vital, '_quality_filtered_' + str(quality))
        out_dir_mixed_vital_raw = FileHelper.get_out_dir(in_dir_mixed_vital_raw, '_quality_filtered_' + str(quality))

        loader = PatientDataLoader()

        def test_filter_one_signal_mixed_vital_raw(self):
            if os.path.exists(self.out_dir_mixed_vital_raw):
                shutil.rmtree(self.out_dir_mixed_vital_raw)

            filename = '009_Test_mixed_raw.csv'
            self.filter.filter_bad_quality_mixed_vital_raw(self.in_dir_mixed_vital_raw, self.out_dir_mixed_vital_raw, filename,
                                                           self.quality)
            df_out = self.loader.load_everion_patient_data(self.out_dir_mixed_vital_raw, filename, ';', False)

            self.assertFalse(np.isnan(df_out['HR'][0]))
            self.assertTrue(np.isnan(df_out['HR'][2]))
            self.assertTrue(np.isnan(df_out['Activity'][2]))
            self.assertTrue(np.isnan(df_out['Classification'][2]))
            self.assertTrue(np.isnan(df_out['steps'][2]))
            self.assertTrue(np.isnan(df_out['pressure'][2]))
            self.assertTrue(np.isnan(df_out['Classification'][7]))
            self.assertEqual(48, df_out['QualityClassification'][7])
            self.assertEqual(77, df_out['HR'][10])
            self.assertEqual(77, df_out['HR'][17])

        def test_filter_one_signal_vital(self):
            if os.path.exists(self.out_dir_vital):
                shutil.rmtree(self.out_dir_vital)

            filename = '002_storage-sig.csv'
            self.filter.filter_bad_quality_vital(self.in_dir_vital, self.out_dir_vital, filename, self.quality)
            df_out = self.loader.load_everion_patient_data(self.out_dir_vital, filename, ";", False)

            self.assertTrue(np.isnan(df_out['heart_rate'][0]))
            self.assertTrue(np.isnan(df_out['heart_rate'][14]))
            self.assertEqual(85, df_out['heart_rate'][16])
            self.assertEqual(82, df_out['heart_rate'][28])
            self.assertTrue(np.isnan(df_out['heart_rate_variability'][0]))
            self.assertTrue(np.isnan(df_out['heart_rate_variability'][28]))


        def test_filter_signals_vital(self):
            if os.path.exists(self.out_dir_vital):
                shutil.rmtree(self.out_dir_vital)

            for filename in os.listdir(self.in_dir_vital):
                print('processing ', filename, ' ...')
                if not (filename.endswith('csv')):
                    continue
                self.filter.filter_bad_quality_vital(self.in_dir_vital, self.out_dir_vital, filename, self.quality)

                if filename == '030_storage-sig.csv':
                    self.assertFalse(os.path.exists(os.path.join(self.out_dir_vital, filename)))
                else:
                    self.assertTrue(os.path.exists(os.path.join(self.out_dir_vital, filename)))

        @unittest.SkipTest
        def test_filter_signal_all_vital(self):
            quality = 50
            dir_name = ''
            out_dir = FileHelper.get_out_dir(dir_name, '_quality_filtered_' + str(quality))

            for filename in os.listdir(dir_name):
                print('processing ', filename, ' ...')
                if not (filename.endswith('csv')):
                    continue
                self.filter.filter_bad_quality_vital(dir_name, out_dir, filename, self.quality)

            self.assertTrue(True)

        @unittest.SkipTest
        def test_filter_signal_all_mixed_vital_raw(self):
            quality = 50
            dir_name = ''
            out_dir = FileHelper.get_out_dir(dir_name, '_quality_filtered_' + str(quality))

            for filename in os.listdir(dir_name):
                print('processing ', filename, ' ...')
                if not (filename.endswith('csv')):
                    continue
                self.filter.filter_bad_quality_mixed_vital_raw(dir_name, out_dir, filename, self.quality)

            self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
