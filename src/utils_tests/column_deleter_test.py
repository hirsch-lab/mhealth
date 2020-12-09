import unittest
import pandas as pd

from utils.column_deleter import ColumnDeleter
from utils.file_helper import FileHelper
from patient.patient_data_loader import PatientDataLoader


class ColumnDeleterTest(unittest.TestCase):
    loader = PatientDataLoader()

    def test_column_deleter_vital(self):
        directory = '../resources/vital_signals/'
        out_dir = FileHelper.get_out_dir(directory, '_cleaned')

        deleter = ColumnDeleter()
        deleter.delete_columns(directory, out_dir, 0, 7, ';')

        df = pd.read_csv('../resources/vital_signals_cleaned/001_storage-sig.csv', delimiter=',')

        self.assertEqual(24, df.shape[1])
        self.assertEqual('energy', df.columns[1])


    def test_column_deleter_mixed_vital_raw(self):
        directory = '../resources/mixed_vital_raw_signals/'
        out_dir = FileHelper.get_out_dir(directory, '_cleaned')

        deleter = ColumnDeleter()
        deleter.delete_columns(directory, out_dir, 0, 7, ';')

        df = pd.read_csv('../resources/mixed_vital_raw_signals_cleaned/001_Test_mixed_raw.csv', delimiter=',')

        self.assertEqual(14, df.shape[1])
        self.assertEqual('Classification', df.columns[1])



if __name__ == '__main__':
    unittest.main()
