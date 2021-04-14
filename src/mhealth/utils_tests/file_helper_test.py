import shutil
import unittest
import tempfile
from pathlib import Path

from ..utils import testing
from ..utils.file_helper import (strip_path_annotation)


class TestStripAnnotations(testing.TestCase):
    def test_basic(self):
        path, annot = strip_path_annotation("path/file.csv", ".csv")
        self.assertEqual(path, Path("path/file.csv"))
        self.assertEqual(annot, None)
        path, annot = strip_path_annotation("path/files.csv/file.csv", ".csv")
        self.assertEqual(path, Path("path/files.csv/file.csv"))
        self.assertEqual(annot, None)
        path, annot = strip_path_annotation("path/file.h5/group/id", ".h5")
        self.assertEqual(path, Path("path/file.h5"))
        self.assertEqual(annot, "group/id")
        path, annot = strip_path_annotation("path/.h5/file.h5/group/id", ".h5")
        self.assertEqual(path, Path("path/.h5/file.h5"))
        self.assertEqual(annot, "group/id")


if __name__ == "__main__":
    unittest.main()
