import unittest

from ..utils import testing
from ..utils.rename_header import RenameHeader


class RenameHeaderTest(testing.TestCase):

    @testing.skip_because_is_runner
    def test_change_header_mixed_vital_raw(self):
        directory = ''
        header = RenameHeader()
        header.change_header(in_dir=directory,
                             start_idx=107,
                             end_idx=131)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
