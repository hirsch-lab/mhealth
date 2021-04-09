import os
import glob
import unittest

from ..data_analysis.symmetry_checker import SymmetryChecker
from ..utils.file_helper import FileHelper

_MHEALTH_DATA = os.getenv('MHEALTH_DATA', '../../resources')
_MHEALTH_OUT_DIR = os.path.join(_MHEALTH_DATA, 'output')


class QualityFilterTest(unittest.TestCase):

    def test_symmetry_analysis(self):
        data_dir = f'{_MHEALTH_DATA}/imove/data'
        out_dir = FileHelper.get_out_dir(in_dir=data_dir,
                                         out_dir=_MHEALTH_OUT_DIR,
                                         out_dir_suffix='_symmetry')

        check = SymmetryChecker(data_dir=data_dir,
                                out_dir=out_dir,
                                columns=['HR'],
                                resample=None)
        check.run()

        files = list(out_dir.glob('**/*.png'))
        self.assertEqual(1, len(files))

        files = list(out_dir.glob('**/*.csv'))
        self.assertEqual(2, len(files))


if __name__ == '__main__':
    unittest.main()

