import shutil
import unittest

from pathlib import Path

from ..utils import testing
from ..utils.plotter_helper import save_figure


class TestSaveFigure(testing.TestCase):
    def setUp(self):
        self.out_dir = self.make_test_dir()
        self.formats = [".pdf", ".png", ".tiff"]

    def tearDown(self):
        self.remove_test_dir(self.out_dir)

    def _test_format(self, ext):
        path1 = save_figure(path=self.out_dir/("plot"+ext),
                            override=True)
        self.assertIsFile(path1)
        path2 = save_figure(path=self.out_dir/("plot"+ext),
                            override=True)
        self.assertEqual(path1, path2)
        path3 = save_figure(path=self.out_dir/("plot"+ext),
                            override=False)
        self.assertIsFile(path3)
        self.assertEqual(path3, self.out_dir/("plot-001"+ext))

    def test_basic_formats(self):
        for ext in self.formats:
            with self.subTest(ext=ext):
                self._test_format(ext)

    def test_invalid_format(self):
        with self.assertRaises(ValueError) as cm:
            path = save_figure(path=self.out_dir/"plot.invalid_extension")
        exception = str(cm.exception)
        self.assertIn("is not supported (supported formats", exception)

