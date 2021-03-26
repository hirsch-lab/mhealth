import os
import unittest

from patient.imove_label_loader import ImoveLabelLoader
from utils.file_helper import FileHelper

import pandas as pd
import numpy as np

_MHEALTH_DATA = os.getenv('MHEALTH_DATA', '../resources')
_MHEALTH_OUT_DIR = os.path.join(_MHEALTH_DATA, 'output')


class ImoveLabelLoaderTest(unittest.TestCase):
    label_loader = ImoveLabelLoader()
    out_dir = _MHEALTH_OUT_DIR

    def test_load_labels(self):
        dir_name = f'{_MHEALTH_DATA}/imove/labels'
        df = self.label_loader.load_labels(dir_name, '123-2.xlsx')

        self.assertEqual((3, 9), df.shape, 'df shape not matching')
        self.assertEqual('datetime64[ns, Europe/Zurich]', df['start_date'].dtypes,
                         'start_date has not correct datetime format')
        self.assertEqual('timedelta64[ns]', df['duration'].dtypes,
                         'duration has not correct timedelta format')


    def test_load_labels_all(self):
        dir_name = f'{_MHEALTH_DATA}/imove/labels'

        for filename in os.listdir(dir_name):
            if not (filename.endswith('xlsx')):
                continue

            print("processing file " + filename + " ...")

            df = self.label_loader.load_labels(dir_name, filename)

            self.assertEqual('datetime64[ns, Europe/Zurich]', df['start_date'].dtypes,
                             'start_date has not correct datetime format')
            self.assertEqual('timedelta64[ns]', df['duration'].dtypes,
                             'duration has not correct timedelta format')


    def test_merge_data_and_labels(self):
        label_dir = f'{_MHEALTH_DATA}/imove/labels'
        data_dir = f'{_MHEALTH_DATA}/imove/data'
        out_dir = FileHelper.get_out_dir(in_dir=data_dir,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_labeled')

        self.label_loader.merge_data_and_labels(data_dir=data_dir,
                                                label_dir=label_dir,
                                                out_dir=out_dir,
                                                start_range=123,
                                                end_range=123,
                                                in_file_suffix='_storage-vital')

        df = pd.read_csv(os.path.join(out_dir, '123L_storage-vital.csv'), ';')
        self.assertEqual((64, 25), df.shape, 'df shape not matching')
        self.assertTrue(np.isnan(df['DeMorton'][0]))
        self.assertEqual(1, df['DeMorton'][1])
        self.assertEqual(1, df['DeMorton'][39])
        self.assertEqual(1, df['DeMorton'][57])
        self.assertEqual('temp', df['DeMortonLabel'][1])
        self.assertEqual('3', df['DeMortonLabel'][39])
        self.assertEqual('5a', df['DeMortonLabel'][57])


    @unittest.SkipTest
    def test_merge_data_and_labels_all(self):
        label_dir = '/Users/sues/Documents/wearables/imove/labels'
        data_dir = '/Users/sues/Documents/wearables/imove/raw_cleaned'
        out_dir = FileHelper.get_out_dir(in_dir=data_dir,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_labeled')

        self.label_loader.merge_data_and_labels(
            data_dir=data_dir,
            label_dir=label_dir,
            out_dir=out_dir,
            start_range=1,
            end_range=30,
            in_file_suffix='_storage-vital_raw')

        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
