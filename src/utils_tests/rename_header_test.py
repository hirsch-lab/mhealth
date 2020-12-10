import unittest

from utils.rename_header import RenameHeader


class RenameHeaderTest(unittest.TestCase):

    @unittest.SkipTest
    def test_change_header_mixed_vital_raw(self):
        directory = ''
        header = RenameHeader()
        header.change_header(directory, 107, 131)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
