import os
import glob
import unittest
import numpy as np

from ..utils import everion_keys
from ..utils.file_helper import FileHelper
from ..utils.data_aggregator import DataAggregator, Normalization
from ..patient.patient_data_loader import PatientDataLoader
from ..visualization.vis_properties import VisProperties

_MHEALTH_DATA = os.getenv('MHEALTH_DATA', '../../resources')
_MHEALTH_OUT_DIR = os.path.join(_MHEALTH_DATA, 'output')


class DataAggregatorTest(unittest.TestCase):
    in_dir = f'{_MHEALTH_DATA}/vital_signals/'
    out_dir = _MHEALTH_OUT_DIR

    aggregator = DataAggregator()
    loader = PatientDataLoader()


    def test_aggregate_data(self):
        out_dir = FileHelper.get_out_dir(in_dir=self.in_dir,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_agg')
        self.aggregator.aggregate_data(dir_name=self.in_dir,
                                       out_dir=out_dir,
                                       start_idx=0,
                                       end_idx=3)
        files = list(out_dir.glob('**/*.csv'))
        self.assertEqual(3, len(files))


    def test_aggregate_data_hourly(self):
        filename = '007_storage-sig.csv'
        patient_id = filename[0:3]
        print("processing file " + filename + " with pid=" + patient_id + " ...")

        df = self.loader.load_everion_patient_data(dir_name=self.in_dir,
                                                   filename=filename,
                                                   csv_delimiter=';')
        if not df.empty:
            properties = VisProperties(in_dir=self.in_dir, out_dir='',
                                       normalization=Normalization.NONE,
                                       keys=everion_keys.MAJOR_VITAL,
                                       short_keys=everion_keys.SHORT_NAMES_VITAL,
                                       min_scale=0, max_scale=100,
                                       start_idx=0, end_idx=3)
            df_a = self.aggregator.aggregate_data_hourly(df=df,
                                                         properties=properties)

            self.assertEqual(64, df_a.shape[0])
            self.assertEqual(5, df_a.shape[1])

            self.check_row_values(df_a.loc[(2020, 3, 28, 15)], 0, 0, 60, 21, 0)
            self.is_row_nan(df_a.loc[(2020, 3, 28, 16)])

            self.is_row_nan(df_a.loc[(2020, 3, 29, 15)])
            self.check_row_values(df_a.loc[(2020, 3, 29, 16)], 0, 0, 60, 21, 0)
            self.is_row_nan(df_a.loc[(2020, 3, 29, 17)])

            self.is_row_nan(df_a.loc[(2020, 3, 30, 15)])
            self.check_row_values(df_a.loc[(2020, 3, 30, 16)], 0, 0, 60, 21, 0)


    def check_row_values(self, row, hr, hrv, spo2, t, rr):
        self.assertEqual(hr, row['HR'])
        self.assertEqual(hrv, row['HRV'])
        self.assertEqual(spo2, row['SPO2'])
        self.assertEqual(t, row['Temp'])
        self.assertEqual(rr, row['RR'])


    def is_row_nan(self, nan_row):
        self.assertTrue(np.isnan(nan_row['HR']))
        self.assertTrue(np.isnan(nan_row['HRV']))
        self.assertTrue(np.isnan(nan_row['SPO2']))
        self.assertTrue(np.isnan(nan_row['Temp']))
        self.assertTrue(np.isnan(nan_row['RR']))


if __name__ == '__main__':
    unittest.main()
