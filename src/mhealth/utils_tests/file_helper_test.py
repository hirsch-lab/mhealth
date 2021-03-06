import unittest
from pathlib import Path

from ..utils import testing
from ..utils.file_helper import (ensure_counted_path,
                                 strip_path_annotation)


class TestCountedPath(testing.TestCase):
    def setUp(self):
        self.fmt = "_%04d"
        self.name = "file.txt"
        self.name_template = f"file{self.fmt}.txt"
        self.empty_dir = self.make_test_dir(prefix="empty")
        self.nonempty_dir = self.make_test_dir(prefix="nonempty")
        self.n_files = 10
        for i in range(self.n_files):
            path = self.nonempty_dir / (self.name_template % (i+1))
            path.touch()

    def tearDown(self):
        self.remove_test_dir(self.empty_dir)
        self.remove_test_dir(self.nonempty_dir)

    def test_basic(self):
        ret = ensure_counted_path(self.empty_dir/"file.txt",
                                  fmt=self.fmt, skip_first=True)
        self.assertEqual(ret, self.empty_dir/"file.txt")
        ret = ensure_counted_path(self.empty_dir/"file.txt",
                                  fmt=self.fmt, skip_first=False)
        self.assertEqual(ret, self.empty_dir/"file_0001.txt")
        ret = ensure_counted_path(self.empty_dir/"file.txt",
                                  fmt=self.fmt, skip_first=True, start=42)
        self.assertEqual(ret, self.empty_dir/"file.txt")
        ret = ensure_counted_path(self.empty_dir/"file.txt",
                                  fmt=self.fmt, skip_first=False, start=42)
        self.assertEqual(ret, self.empty_dir/"file_0042.txt")
        ret = ensure_counted_path(self.empty_dir/"file.txt",
                                  fmt=self.fmt, skip_first=True, step=42)
        self.assertEqual(ret, self.empty_dir/"file.txt")
        ret = ensure_counted_path(self.empty_dir/"file.txt",
                                  fmt=self.fmt, skip_first=False, step=42)
        self.assertEqual(ret, self.empty_dir/"file_0001.txt")

        ret = ensure_counted_path(self.nonempty_dir/"file.txt",
                                  fmt=self.fmt, skip_first=True)
        self.assertEqual(ret, self.nonempty_dir/"file_0011.txt")
        ret = ensure_counted_path(self.nonempty_dir/"file.txt",
                                  fmt=self.fmt, skip_first=False)
        self.assertEqual(ret, self.nonempty_dir/"file_0011.txt")
        ret = ensure_counted_path(self.nonempty_dir/"file.txt",
                                  fmt=self.fmt, skip_first=False, start=42)
        self.assertEqual(ret, self.nonempty_dir/"file_0042.txt")
        ret = ensure_counted_path(self.nonempty_dir/"file.txt",
                                  fmt=self.fmt, skip_first=False, start=5)
        self.assertEqual(ret, self.nonempty_dir/"file_0011.txt")
        ret = ensure_counted_path(self.nonempty_dir/"file.txt",
                                  fmt=self.fmt, skip_first=False, step=10)
        self.assertEqual(ret, self.nonempty_dir/"file_0020.txt")
        ret = ensure_counted_path(self.nonempty_dir/"file.txt",
                                  fmt=self.fmt, skip_first=True, step=10)
        self.assertEqual(ret, self.nonempty_dir/"file_0020.txt")

        # File with different extension
        ret = ensure_counted_path(self.nonempty_dir/"file.csv",
                                  fmt=self.fmt, skip_first=True)
        self.assertEqual(ret, self.nonempty_dir/"file.csv")
        ret = ensure_counted_path(self.nonempty_dir/"file_0001.csv",
                                  fmt=self.fmt, skip_first=True)
        self.assertEqual(ret, self.nonempty_dir/"file_0001.csv")


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
