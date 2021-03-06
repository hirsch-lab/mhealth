import unittest
from pathlib import Path

from ..utils import testing
from ..utils.io_manager import (IOManager, extract_infos)


def write_csv(data, path):
    #print("Writing CSV %s..." % path)
    return path
def write_hdf(data, path):
    #print("Writing HDF %s..." % path)
    return path


class TestExtractInfos(testing.TestCase):
    def test_basic(self):
        extract_initials = lambda ret: (ret.group(1)+ret.group(2)).upper()
        infos = extract_infos("walt_disney",
                              patterns={"surname": ".*_(.*)",
                                        "initials": "(.).*_(.).*"},
                              transformers={"initials": extract_initials})
        self.assertEqual(infos["surname"], "disney")
        self.assertEqual(infos["initials"], "WD")


class TestIOManager(testing.TestCase):
    def setUp(self):
        self.in_dir = Path("path/to/data")
        self.files = [
            self.in_dir / "a.txt",
            self.in_dir / "b.txt",
            self.in_dir / "c.txt",
            self.in_dir / "d.txt",
        ]
        self.out_dir = self.make_test_dir(prefix="output")

    def tearDown(self):
        self.remove_test_dir(self.out_dir)

    def test_constr_init_equivalence(self):
        info_patterns = { "name_upper": "(.*)", }
        info_transformers = { "name_upper": lambda ret: ret.group(1).upper() }
        targets = [".csv", ".h5"]
        target_writers = {".csv": write_csv, ".h5":  write_hdf}
        target_names = {".csv": "{name}_{info1}{info3:03d}.csv"}

        iom1 = IOManager(out_dir=self.out_dir,
                         targets=None, # infer targets
                         info_patterns=info_patterns,
                         info_transformers=info_transformers,
                         target_writers=target_writers,
                         target_names=target_names,
                         skip_existing=False)
        iom2 = IOManager()
        iom2.init(out_dir=self.out_dir,
                  targets=None, # infer targets
                  info_patterns=info_patterns,
                  info_transformers=info_transformers,
                  target_writers=target_writers,
                  target_names=target_names,
                  skip_existing=False)
        self.assertDictEqual(iom1.__dict__, iom2.__dict__)

    def test_noop(self):
        iom = IOManager()
        for filepath in self.files:
            with iom.current(filepath):
                if iom.skip_existing():
                    continue
                ret = iom.write_data(data=123)
                self.assertIsInstance(ret, dict)
                self.assertFalse(ret)

    def test_basic(self):
        iom = IOManager(out_dir=self.out_dir,
                        target_writers={".csv": write_csv},
                        skip_existing=False)
        for filepath in self.files:
            with iom.current(filepath):
                if iom.skip_existing():
                    continue
                # path = iom.path
                # ... processing ...
                data = 123
                ret = iom.write_data(data=data)
                self.assertIsInstance(ret, dict)
                self.assertIn(".csv", ret)
                self.assertEqual(ret[".csv"],
                                 self.out_dir / (iom.name + ".csv"))

    def test_advanced(self):

        info_patterns = { "name_upper": "(.*)", }
        info_transformers = { "name_upper": lambda ret: ret.group(1).upper() }
        targets = [".csv", ".h5"]
        target_writers = {".csv": write_csv, ".h5":  write_hdf}
        target_names = {".csv": "{name}_{info1}{info3:03d}.csv"}
        # The target name, if not specified, is by default: {name}.{key}
        # For instance: target_names[".h5"] = "{name}.h5"
        iom = IOManager(out_dir=self.out_dir,
                        targets=None, # infer targets
                        info_patterns=info_patterns,
                        info_transformers=info_transformers,
                        target_writers=target_writers,
                        target_names=target_names,
                        skip_existing=False)

        for i, filepath in enumerate(self.files):
            with iom.current(filepath,
                             info1="foo",  # pass extra info
                             info2="bar"):
                if iom.skip_existing():
                    continue
                iom.set_info(info3=i)
                # ... processing ...
                data = 123
                ret = iom.write_data(data=data)

if __name__ == "__main__":
    unittest.main()
