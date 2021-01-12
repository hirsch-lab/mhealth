import os
import unittest

from patient.patient_data_loader import PatientDataLoader

_MHEALTH_DATA = os.getenv('MHEALTH_DATA', '../resources')


class PatientDataLoaderTest(unittest.TestCase):

    def test_load_everion_data_tz_zurich(self):
        dir_name = f'{_MHEALTH_DATA}/vital_signals/'
        loader = PatientDataLoader()
        df = loader.load_everion_patient_data(dir_name, '002_storage-sig.csv', ';')

        self.assertEqual(30, len(df.columns), 'not correct amount of keys loaded')
        self.assertEqual((29,30), df.shape, 'df shape not matching')
        self.assertEqual('datetime64[ns, Europe/Zurich]', df['timestamp'].dtypes,
                         'timestamp has not correct datetime format')

    def test_load_everion_data_tz_utc(self):
        dir_name = f'{_MHEALTH_DATA}/vital_signals/'
        loader = PatientDataLoader()
        df = loader.load_everion_patient_data(dir_name, '002_storage-sig.csv', ';', False)

        self.assertEqual(30, len(df.columns), 'not correct amount of keys loaded')
        self.assertEqual((29, 30), df.shape, 'df shape not matching')
        self.assertEqual('datetime64[ns, UTC]', df['timestamp'].dtypes,
                         'timestamp has not correct datetime format')

    def test_load_extra_patient_data(self):
        extra_data_dir_name = f'{_MHEALTH_DATA}/extra_data/'

        loader = PatientDataLoader()
        dict = loader.load_extra_patient_data(os.path.join(extra_data_dir_name, 'extra_data.csv'))

        self.assertEqual(8, len(dict), 'not correct amount of patients loaded')
        self.assertEqual(1, dict['001']['gender_code'], 'gender code should be 1')
        self.assertEqual(50, dict['001']['age'], 'age should be 50')
        self.assertEqual(0, dict['003']['gender_code'], 'gender code should be 0')
        self.assertEqual(32, dict['003']['age'], 'age should be 32')
        self.assertEqual(0, dict['007']['gender_code'], 'gender code should be 0')
        self.assertEqual(41, dict['007']['age'], 'age should be 41')



if __name__ == '__main__':
    unittest.main()
