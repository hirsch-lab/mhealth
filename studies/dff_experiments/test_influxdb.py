import sys
import argparse
import datetime
import pandas as pd
from pathlib import Path
from influxdb_client import InfluxDBClient

import context
from mhealth.utils.file_helper import ensure_dir

# Query to create the raw data plots.
FLUX_QUERY_1 = """
from(bucket: "testing-wehs-juch")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "USB_test")
  |> filter(fn: (r) =>
            r["Vitalparameter"] == "Herzfrequenz" and r["_value"] > 0
            )
  |> aggregateWindow(every: 2m, fn: mean, createEmpty: true)
"""

# Query to create the "data availability plots".
FLUX_QUERY_2 = """
import "dict"

// Define a new dictionary using an array of records
bedToInt = dict.fromList(
  pairs: [
    {key: "2617FR", value: 1},
    {key: "2618FR", value: 2}
  ]
)

from(bucket: "testing-wehs-juch")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "USB_test")
  |> filter(fn: (r) =>
            r["Vitalparameter"] == "Herzfrequenz" and r["_value"] > 0
            )
  |> map(fn: (r) => ({ r with _value: int(v: r._value > 0)*dict.get(dict: bedToInt, key: r.Bettenstellplatz, default: 0) }))
  //|> map(fn: (r) => ({ r with _value: r._value > 0}))
  //|> toInt()
  |> aggregateWindow(every: 2m, fn: mean, createEmpty: true)
"""


def confirm(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".

    Source: https://stackoverflow.com/a/3041990/3388962
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            msg = "Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n"
            sys.stdout.write(msg)


def load_subset_covid_bbtest(in_path, out_dir):
    # beds = ["2617FR", "2618FR", "2656FR", "2661FL"]
    out_path = out_dir / "covid_bbtest_store.h5"
    if out_path.is_file():
        print("Reading data lazily...")
        store = pd.HDFStore(out_path, mode="r")
        df_sensor = store["sensor"]
        df_valid = store["valid"]
        store.close()
    else:
        if in_path is None:
            print("Error: Requiring in_path to point to a HDF store.")
            exit()
        print("Reading data...")
        beds = ["2617FR", "2618FR"]
        store = pd.HDFStore(in_path)
        df_bb = store["bb"]
        df_core = store["core"]
        df_valid = store["valid"]
        store.close()
        df_bb = df_bb[df_bb["Bettenstellplatz"].isin(beds)]
        df_core = df_core[df_core["Bettenstellplatz"].isin(beds)]
        df_valid = df_valid[df_valid["Bettenstellplatz"].isin(beds)]
        df_valid = df_valid.drop(["Bemerkungen", "Abweichung_Trageort"], axis=1)
        df_sensor = pd.concat([df_bb, df_core], axis=0)
        #df_sensor = df_bb.copy()
        ensure_dir(out_dir)
        df_sensor.to_hdf(out_path, key="sensor")
        df_valid.to_hdf(out_path, key="valid")

    df_sensor = df_sensor.set_index("Timestamp")
    return df_sensor, df_valid


def run(args):
    in_path = Path(args.in_path) if args.in_path else None
    out_dir = Path(args.out_dir)
    db_bucket = args.bucket
    db_token = args.token
    db_url = args.url
    db_org = "my-org"
    action = args.action
    measurement = args.measurement

    print("Opening client...")
    client = InfluxDBClient(url=db_url, token=db_token, org=db_org)

    if action=="upload":
        df_sensor, df_valid = load_subset_covid_bbtest(in_path=in_path,
                                                       out_dir=out_dir)
        # Create "measurement":
        with client.write_api() as writer:
            print("Writing data to InfluxDB...")
            print(df_sensor)
            ret = writer.write(db_bucket,
                               org=db_org,
                               record=df_sensor,
                               data_frame_measurement_name=measurement,
                               data_frame_tag_columns=["Vitalparameter",
                                                       "Bettenstellplatz",
                                                       "DeviceID",
                                                       "Signatur"])
            print(ret)
    elif action=="query":
        query = FLUX_QUERY_1
        print("Querying data from InfluxDB...")
        print("Query:")
        print(query)
        system_stats = client.query_api().query_data_frame(org=db_org,
                                                           query=query)
    elif action=="delete":
        msg = "Do you want to delete the measurement '%s'?" % measurement
        ret = False
        ret = confirm(question=msg, default="no")
        if ret:
            msg = "Do you really want to delete the measurement?"
            ret = confirm(question=msg, default="no")
        if ret:
            print("Deleting data...")
            start = "1970-01-01T00:00:00Z"
            stop = datetime.datetime.now(datetime.timezone.utc).isoformat()
            delete_api = client.delete_api()
            delete_api.delete(start, stop,
                              predicate=f'_measurement="{measurement}"',
                              bucket=db_bucket, org=db_org)
        else:
            print("No action performed.")

    print("Closing client...")
    client.close()



def parse_args():
    description = "Tests with InfluxDB."
    formatter = argparse.RawTextHelpFormatter
    parser = argparse.ArgumentParser(add_help=False,
                                     formatter_class=formatter,
                                     description=description)
    parser.add_argument("-h", "--help", action="help",
                        help="Show this help text")
    parser.add_argument("-a", "--action", choices=["upload", "query", "delete"],
                        help="Action to perform")
    parser.add_argument("-i", "--in-path", type=str, default=None,
                        help="Path to data")
    parser.add_argument("-m", "--measurement", type=str, default="USB_test",
                        help="Name of measurement.")
    parser.add_argument("-o", "--out-dir", default="./results/", type=str,
                        help="Output directory")
    parser.add_argument("-b", "--bucket", required=True, type=str,
                        help="InfluxDB bucket name")
    parser.add_argument("-t", "--token", required=True, type=str,
                        help="Token for InfluxDB")
    parser.add_argument("-u", "--url", required=True, type=str,
                        help="URL to InfluxDB instance")
    parser.set_defaults(func=run)
    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
