import re
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from patient.patient_data_loader import PatientDataLoader
from utils.everion_keys import EverionKeys


def ensure_dir(path):
    path = Path(path)
    if not path:
        return False
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)
    return path.is_dir()


def save_figure(path, fig=None, dpi=300):
    path = Path(path)
    if fig is not None:
        # Make the figure with fig the current figure
        plt.figure(fig.number)
    if not ensure_dir(path.parent):
        assert False, "Failed to create output directory: %s " % path.parent
    plt.savefig(path, bbox_inches="tight", dpi=dpi)


class SymmetryChecker:
    """
    Patients wear a sensor on the left and right side. This utility analyzes
    and visualizes the relationship between these data sources. Most notably
    it estimates (per-patient) the Pearson correlation coefficient between
    the left and right signals, and creates Bland-Altman plots for a graphical
    analysis.
    """

    def __init__(self, data_dir, out_dir, columns, resample=None):
        """
        Arguments:
            data_dir: Input folder with the .csv files. The following naming
                      convention applies for the .csv files:
                            ([0-9]*)(L|R).*
                      The first capture group identifies the patient id, the
                      second one whether the data is from left or right side.
                            001L_storage-vital.csv
                            001R_storage-vital.csv
            out_dir: Path to output directory
            columns: Columns to select ("columns of interest", coi)
            resample: Optionally resample the data. For instance, setting
                      resample="30s" aggregates the data into 30s bins.
                      See doc of pd.DataFrame.resample for details.

        """
        self._loader = PatientDataLoader()
        self._data_dir = Path(data_dir)
        self._out_dir = Path(out_dir)
        self._resample = resample
        # Columns of interest.
        self._coi = columns

    def _load_data(self, file_left, file_right):
        def cols_filter(x):
            # Ensure only columns with dtype float.
            return x not in ["de_morton_label"]
        from collections import defaultdict
        dtypes = defaultdict(lambda: float)
        dtypes["timestamp"] = str
        df_left = self._loader.load_everion_patient_data(dir_name=self._data_dir,
                                                         filename=file_left.name,
                                                         csv_delimiter=',',
                                                         usecols=cols_filter,
                                                         dtype=dtypes)
        df_right = self._loader.load_everion_patient_data(dir_name=self._data_dir,
                                                          filename=file_right.name,
                                                          csv_delimiter=',',
                                                          usecols=cols_filter,
                                                          dtype=dtypes)
        df_left = df_left.set_index("timestamp")
        df_right = df_right.set_index("timestamp")
        # select columns of interest
        df_left = df_left[self._coi+["de_morton"]]
        df_right = df_right[self._coi+["de_morton"]]
        # trick to prepend column level
        df_left = pd.concat([df_left], keys=["left"], axis=1)
        df_right = pd.concat([df_right], keys=["right"], axis=1)
        df_left = df_left.reorder_levels([1,0], axis=1)
        df_right = df_right.reorder_levels([1,0], axis=1)
        # join on index
        df = df_left.join(df_right, how="outer")
        df = df.sort_index(axis=1)

        if self._resample is not None:
            # Warning: this introduces a fixed sampling pattern!
            df = df.resample(self._resample).mean()
            df["de_morton"] = df["de_morton"] > 0.5

        return df


    def _analyze_per_patient(self, df, col, pid):
        x = df[col]
        n = len(x)

        # Bland-Altman
        diff = x["left"] - x["right"]
        avg = x.mean(axis=1)
        diff_mean = diff.mean()
        diff_std = diff.std()

        mask = x.isnull().any(axis=1)
        mask += 2*(x == 0).any(axis=1)
        mask += 4*df["de_morton"].any(axis=1)

        fig, ax = plt.subplots()
        ax.scatter(avg, diff, c=mask, alpha=0.1)
        xlim = ax.get_xlim()
        ax.plot(xlim, diff_mean*np.ones(2), "b",
                label="Mean: %.3f" % diff_mean)
        ax.plot(xlim, (diff_mean+1.96*diff_std)*np.ones(2), ":r")
        ax.plot(xlim, (diff_mean-1.96*diff_std)*np.ones(2), ":r",
                label="95%% CI: ±%.3f" % (1.96*diff_std))
        ax.grid(True)
        ax.set_title(f"Bland-Altman: {col}, pid={pid}")
        ax.set_xlabel("Mean: (Left+Right)/2")
        ax.set_ylabel("Difference: (Left-Right)")
        leg = ax.legend(title="Difference:",
                        loc="upper left",
                        bbox_to_anchor=(1.05, 1.02))
        ax.set_axisbelow(True)
        plt.tight_layout()
        leg._legend_box.align = "left"

        if self._out_dir:
            filename = ("bland-altman-%s-%s.png" % (col.lower(), pid))
            path = self._out_dir / "plots" / filename
            save_figure(path=path, fig=fig)
        plt.close(fig)

        info = pd.DataFrame(columns=x.columns)
        info.loc["counts"] = n
        # null: {NaN, None NaT}
        info.loc["nans"] = x.isnull().sum(axis=0)
        info.loc["nan_ratio"] = info.loc["nans"] / n
        info.loc["zero_ratio"] = (x == 0).sum(axis=0) / n
        info.loc["const_ratio"] = ((x.shift(1)-x) == 0).sum(axis=0) / (n-1)

        info_diff = pd.Series(name="diff", dtype=float)
        info_diff["mean"] = diff_mean
        info_diff["std"] = diff_std
        info_diff["5%"]  = diff.quantile(0.05)
        info_diff["25%"] = diff.quantile(0.25)
        info_diff["50%"] = diff.quantile(0.50)
        info_diff["75%"] = diff.quantile(0.75)
        info_diff["95%"] = diff.quantile(0.95)
        info_diff["corr"] = x["left"].corr(x["right"])

        # print()
        # print(col)
        # print(info)
        # print()
        # print("Correlation: %.3f" % info_diff["corr"])

        # Format output as series with multi-level index.
        ret = pd.concat([info.unstack(),
                         info_diff.to_frame().unstack()],
                        axis=0)
        ret.name = (pid, col)
        return ret


    def run(self):
        file_pattern = re.compile(r"([0-9]*)(L|R).*")
        files = sorted(self._data_dir.glob("*.csv"))
        files = {file_pattern.match(f.stem).groups():f for f in files}
        files = pd.Series(files)
        rets = []
        for key, df in files.groupby(level=0):
            if len(df)!=2:
                warnings.warn("No matching files found for patient '%s'" % key)
                continue
            file_left = df[(key, "L")]
            file_right = df[(key, "R")]
            df = self._load_data(file_left=file_left, file_right=file_right)
            print("processing data for patient %s ..." % key)
            for col in self._coi:
                ret = self._analyze_per_patient(df, col=col, pid=key)
                rets.append(ret)

        rets = pd.concat(rets, axis=1)
        rets.to_csv(self._out_dir / "results.csv")
