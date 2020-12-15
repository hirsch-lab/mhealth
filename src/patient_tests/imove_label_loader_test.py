import os
import unittest

from patient.imove_label_loader import ImoveLabelLoader
from utils.file_helper import FileHelper


class ImoveLabelLoaderTest(unittest.TestCase):
    label_loader = ImoveLabelLoader()

    def test_load_labels(self):
        dir_name = '../resources/imove/labels'
        df = self.label_loader.load_labels(dir_name, '001-2.xlsx')

        self.assertEqual((21, 9), df.shape, 'df shape not matching')
        self.assertEqual('datetime64[ns]', df['start_date'].dtypes,
                         'start_date has not correct datetime format')
        self.assertEqual('timedelta64[ns]', df['duration'].dtypes,
                         'duration has not correct timedelta format')

    def test_load_labels_all(self):
        dir_name = '../resources/imove/labels'

        for filename in os.listdir(dir_name):
            if not (filename.endswith('xlsx')):
                continue

            print("processing file " + filename + " ...")

            df = self.label_loader.load_labels(dir_name, filename)

            self.assertEqual('datetime64[ns]', df['start_date'].dtypes,
                             'start_date has not correct datetime format')
            self.assertEqual('timedelta64[ns]', df['duration'].dtypes,
                             'duration has not correct timedelta format')


    def test_merge_data_and_labels(self):
        label_dir = '../resources/imove/labels'
        data_dir = '../resources/imove/data'
        out_dir = FileHelper.get_out_dir(data_dir, '_labeled')

        self.label_loader.merge_data_and_labels(data_dir, label_dir, out_dir, 1, 1, '_storage-vital')

        self.assertTrue(True)


    @unittest.SkipTest
    def test_merge_data_and_labels_all(self):
        label_dir = ''
        data_dir = ''
        out_dir = FileHelper.get_out_dir(data_dir, '_labeled')

        self.label_loader.merge_data_and_labels(data_dir, label_dir, out_dir, 1, 30, '_storage-vital')

        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
