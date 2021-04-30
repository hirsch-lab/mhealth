import unittest

from patient.rename_header import RenameHeader


class RenameHeaderTest(unittest.TestCase):

    #@unittest.SkipTest
    def test_change_header_mixed_vital_raw(self):
        directory = '/Users/reys/Desktop/ACLS_Master/MasterThesis/DataMining_covid/UKBB/data_short/'
        header = RenameHeader()
        header.change_header(directory)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
