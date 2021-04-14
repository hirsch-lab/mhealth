import os
import shutil
import tempfile
import unittest
import importlib
import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial
from contextlib import redirect_stdout


def is_module_available(name):
    try:
        ret = importlib.import_module(name)
        return bool(ret)
    except:
        return False


def is_env_available(env):
    ret = os.getenv(env)
    return bool(ret)


def check_module(name):
    ret = not is_module_available(name)
    msg = "Module %s is not available." % name
    return ret, msg


def check_env(env):
    ret = not is_env_available(env)
    msg = "Environment variable %s is not set." % env
    return ret, msg


def skip_if_module_not_found(name):
    """
    Use as decorator for tests or test classes, just like @skip(reason)
    """
    return unittest.skipIf(*check_module(name))


def skip_if_env_not_set(env):
    """
    Use as decorator for tests or test classes, just like @skip(reason)
    """
    return unittest.skipIf(*check_env(env))


def skip_because_is_runner(func=None):
    """
    Use as decorator for tests or test classes, just like @skip(reason)

    Use this decorator to mark tests creating functional output
    (legacy from sues).
    """
    @unittest.skip("Running script")
    def decorator(func):
        return func
    def wrapper(func):
        return decorator
    if callable(func):
        return decorator
    return wrapper


class StdoutRedirectionContext():
    # https://stackoverflow.com/questions/59201313/
    class ListIO():
        def __init__(self):
            self.output = []
        def write(self, s):
            if s in ("\n", ""): return
            self.output.append(s)

    def __enter__(self):
        self._buf = self.ListIO()
        self._ctx = redirect_stdout(self._buf)
        self._ctx.__enter__()
        return self._buf

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._ctx.__exit__(exc_type, exc_value, exc_traceback)
        del self._ctx


class TestCase(unittest.TestCase):
    @staticmethod
    def make_test_dir(prefix="test"):
        return Path(tempfile.mkdtemp(prefix=prefix))

    @staticmethod
    def remove_test_dir(path):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)

    def assertIsFile(self, path):
        if path is None:
            raise AssertionError("None is not a valid path.")
        if not Path(path).resolve().is_file():
            raise AssertionError("File does not exist: %s" % str(path))

    def assertIsNotFile(self, path):
        if path is None:
            raise AssertionError("None is not a valid path.")
        if Path(path).resolve().is_file():
            raise AssertionError("File should not exist: %s" % str(path))

    def assertIsDir(self, path):
        if path is None:
            raise AssertionError("None is not a valid path.")
        if not Path(path).resolve().is_dir():
            raise AssertionError("Dir does not exist: %s" % str(path))

    def assertIsNotDir(self, path):
        if path is None:
            raise AssertionError("None is not a valid path.")
        if Path(path).resolve().is_dir():
            raise AssertionError("Dir should not exist: %s" % str(path))

    def assertExists(self, path):
        if path is None:
            raise AssertionError("None is not a valid path.")
        if not Path(path).resolve().exists():
            raise AssertionError("Path does not exist: %s" % str(path))

    def assertNotExists(self, path):
        if path is None:
            raise AssertionError("None is not a valid path.")
        if Path(path).resolve().exists():
            raise AssertionError("Path should not exist: %s" % str(path))

    def assertArrayEqual(self, x, y):
        np.testing.assert_array_equal(x, y)

    def assertAlmostEqual(self, x, y, places=7):
        # Overrides the corresponding method of unittest.TestCase.
        np.testing.assert_almost_equal(x, y, decimal=places)

    def assertFrameEqual(self, x, y, **kwargs):
        pd.testing.assert_frame_equal(x, y, **kwargs)

    def assertStdout(self):
        """
        Use similarly as assertLogs():
            https://docs.python.org/3/library/unittest.html
            https://stackoverflow.com/questions/59201313

            class SomeTest(TestCase):
                def test_stdout(self):
                    with self.assertStdout() as cm:
                        print("foo!")
                        print("bar!")
                    self.assertIn("foo!", cm.output)
                    self.assertIn("baz!", cm.output)
        """
        return StdoutRedirectionContext()
