import os
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from ..utils import testing
from ..utils.file_helper import FileHelper
from ..patient.patient_data_loader import PatientDataLoader
from ..patient.imove_label_loader import merge_labels, load_labels


_MHEALTH_DATA = os.getenv("MHEALTH_DATA", "../../resources")


class ImoveLabelLoaderTest(testing.TestCase):
    data_dir = Path(f"{_MHEALTH_DATA}")

    def test_load_labels(self):
        dir_name =  self.data_dir / "imove/labels"
        df = load_labels(dir_name / "123-2.xlsx", tz_to_zurich=True)
        self.assertEqual(df.shape, (3, 5), "Shape doesn't match")
        self.assertListEqual(df.columns.tolist(),
                             ["Lap", "Task", "StartDate", "Duration", "EndDate"],
                             "Columns don't match")
        self.assertEqual(df["StartDate"].dtypes,
                         "datetime64[ns, Europe/Zurich]",
                         "StartDate has wrong datetime format")
        self.assertEqual(df["Duration"].dtypes, "timedelta64[ns]",
                         "Duration has wrong timedelta format")
        self.assertListEqual(df["Task"].tolist(), ["temp", "3", "5a"])


    def test_merge_data_and_labels(self):
        labels_dir = self.data_dir / "imove/labels"
        data_dir = self.data_dir / "imove/data"

        df_labels = load_labels(labels_dir / "123-2.xlsx", tz_to_zurich=True)
        loader = PatientDataLoader()
        for path in data_dir.glob("*.csv"):
            df = loader.load_everion_patient_data(dir_name=path.parent,
                                                  filename=path.name,
                                                  csv_delimiter=';',
                                                  tz_to_zurich=True)

            ret = merge_labels(df=df, df_labels=df_labels)

            self.assertEqual((64, 25), df.shape, "df shape not matching")
            self.assertFalse(df["DeMorton"][0])
            self.assertEqual(1, df["DeMorton"][1])
            self.assertEqual(1, df["DeMorton"][39])
            self.assertEqual(1, df["DeMorton"][57])
            self.assertEqual("temp", df["DeMortonLabel"][1])
            self.assertEqual("3", df["DeMortonLabel"][39])
            self.assertEqual("5a", df["DeMortonLabel"][57])


if __name__ == "__main__":
    unittest.main()
