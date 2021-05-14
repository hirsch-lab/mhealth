import os
import argparse
from pathlib import Path

import context
from mhealth.utils.context_info import dump_context
from mhealth.data_analysis.symmetry_checker import SymmetryChecker


def run(args):
    data_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    metrics = args.metrics
    resample = args.resample

    if metrics is None:
        metrics = ["HR", "HRQ", "HRV", "RespRate",
                   "SpO2", "BloodPressure",
                   "Activity", "Classification"]

    dump_context(out_dir=out_dir)
    check = SymmetryChecker(data_dir=data_dir,
                            out_dir=out_dir,
                            columns=metrics,
                            resample=resample)
    check.run()


def parse_args():
    description = ("Collect and format data from Everion devices.")
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(add_help=False,
                                     formatter_class=formatter,
                                     description=description)
    parser.add_argument("-h", "--help", action="help",
                        help="Show this help text")
    parser.add_argument("-i", "--in-dir", required=True,
                        help="Input directory")
    parser.add_argument("-o", "--out-dir",
                        default="../output/analysis/symmetry",
                        help="Output directory")
    parser.add_argument("--metrics", default=None, nargs="+",
                        help="Select a subset of metrics for the analysis")
    parser.add_argument("--resample", type=str, default=None,
                        help=("Resampling expression. "
                              "Example: '--resample=30s' Default: None"))
    parser.set_defaults(func=run)
    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
