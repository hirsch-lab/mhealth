import os
import unittest
import pandas as pd

from ..utils.file_helper import FileHelper
from ..utils.column_deleter import ColumnDeleter
from ..patient.patient_data_loader import PatientDataLoader

_MHEALTH_DATA = os.getenv('MHEALTH_DATA', '../../resources')
_MHEALTH_OUT_DIR = os.path.join(_MHEALTH_DATA, 'output')


class ColumnDeleterTest(unittest.TestCase):
    loader = PatientDataLoader()
    out_dir = _MHEALTH_OUT_DIR

    def test_column_deleter_vital(self):
        directory = f'{_MHEALTH_DATA}/vital_signals/'
        out_dir = FileHelper.get_out_dir(in_dir=directory,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_cleaned')

        deleter = ColumnDeleter()
        deleter.delete_columns(dir_name=directory,
                               out_dir=out_dir,
                               start_del=0,
                               end_del=7,
                               delimiter=';')
        df = pd.read_csv(out_dir/'001_storage-sig.csv', delimiter=',')
        self.assertEqual(24, df.shape[1])
        self.assertEqual('energy', df.columns[1])

    def test_column_deleter_mixed_vital_raw(self):
        directory = f'{_MHEALTH_DATA}/mixed_vital_raw_signals/'
        out_dir = FileHelper.get_out_dir(in_dir=directory,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_cleaned')

        deleter = ColumnDeleter()
        deleter.delete_columns(dir_name=directory,
                               out_dir=out_dir,
                               start_del=0,
                               end_del=7,
                               delimiter=';')
        df = pd.read_csv(out_dir/'001_Test_mixed_raw.csv', delimiter=',')
        self.assertEqual(14, df.shape[1])
        self.assertEqual('Classification', df.columns[1])


    @unittest.SkipTest
    def test_clean_imove_raw(self):
        directory = ''
        out_dir = FileHelper.get_out_dir(in_dir=directory,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_cleaned')

        deleter = ColumnDeleter()
        deleter.delete_columns_and_rename_header(dir_name=directory,
                                                 out_dir=out_dir,
                                                 start_del=0,
                                                 end_del=11,
                                                 delimiter=';')
        self.assertTrue()


if __name__ == '__main__':
    unittest.main()
