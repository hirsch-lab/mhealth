import csv
import os
import pandas as pd

import natsort

from patient.patient_data_loader import PatientDataLoader


class SanityChecker:
        loader = PatientDataLoader()

        def run_vital(self, dir_name, id_range, in_file_suffix, out_file_name):
            csv_out_file = os.path.join(os.path.join(dir_name, os.pardir), out_file_name)
            if os.path.exists(csv_out_file):
                os.remove(csv_out_file)

            with open(csv_out_file, mode='a+') as out_file:
                csv_writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(['Id', 'Available', '#ValidRows', '#ValidHours',
                                     'timestamp_min', 'timestamp_max', 'hours',
                                     'HR_mean', 'HR_std', 'HR_min', 'HR_max',
                                     'HRV_mean', 'HRV_std', 'HRV_min', 'HRV_max',
                                     'Spo2_mean', 'Spo2_std', 'Spo2_min', 'Spo2_max',
                                     'RR_mean', 'RR_std', 'RR_min', 'RR_max',
                                     'T_mean', 'T_std', 'T_min', 'T_max',
                                     'HR_q_mean', 'HR_q_std', 'HR_q_min', 'HR_q_max',
                                     'HRV_q_mean', 'HRV_q_std', 'HRV_q_min', 'HRV_q_max',
                                     'Spo2_q_mean', 'Spo2_q_std', 'Spo2_q_min', 'Spo2_q_max',
                                     'RR_q_mean', 'RR_q_std', 'RR_q_min', 'RR_q_max',
                                     'T_q_mean', 'T_q_std', 'T_q_min', 'T_q_max',
                                     'AC_q_mean', 'AC_q_std', 'AC_q_min', 'AC_q_max',
                                     'E_q_mean', 'E_q_std', 'E_q_min', 'E_q_max'
                                     ])

                files_sorted = natsort.natsorted(os.listdir(dir_name))

                for count in range(id_range):
                    id = str(count + 1).zfill(3)

                    found = False
                    for i, filename in enumerate(files_sorted):
                        if filename.__contains__(id):
                            found = True

                    if not (found):
                        print("file not found with id: ", id)
                        csv_writer.writerow([id, 'na', '', '', '', '', '', '', ''])

                    else:
                        filename = id + in_file_suffix
                        print("processing file: ", filename, " ...")
                        file_path = os.path.join(dir_name, filename)

                        with open(file_path) as csv_file:
                            csv_reader = csv.reader(csv_file, delimiter=';') # TODO: use df from below
                            line_count = 0
                            for row in csv_reader:
                                if line_count == 0:
                                    line_count += 1
                                    continue;
                                if int(row[0]) > 0:
                                    line_count += 1

                        if line_count > 0:
                            line_count -= 1  # -1 to subtract header row

                            valid_hours = line_count / 3600.0

                            df = self.loader.load_everion_patient_data(dir_name,filename, ';')

                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                            ts_max = max(df['timestamp'])
                            ts_min = min(df['timestamp'])
                            hours = pd.Timedelta(ts_max - ts_min).seconds / 3600.0
                            hours += pd.Timedelta(ts_max - ts_min).days * 24

                            key = 'heart_rate'
                            hr_mean = self.get_mean(df, key)
                            hr_min = self.get_min(df, key)
                            hr_max = self.get_max(df, key)
                            hr_std = self.get_std(df, key)

                            key = 'heart_rate_variability'
                            hrv_mean = self.get_mean(df, key)
                            hrv_min = self.get_min(df, key)
                            hrv_max = self.get_max(df, key)
                            hrv_std = self.get_std(df, key)

                            key = 'oxygen_saturation'
                            spo2_mean = self.get_mean(df, key)
                            spo2_min = self.get_min(df, key)
                            spo2_max = self.get_max(df, key)
                            spo2_std = self.get_std(df, key)

                            key = 'respiration_rate'
                            rr_mean = self.get_mean(df, key)
                            rr_min = self.get_min(df, key)
                            rr_max = self.get_max(df, key)
                            rr_std = self.get_std(df, key)

                            key = 'core_temperature'
                            t_mean = self.get_mean(df, key)
                            t_min = self.get_min(df, key)
                            t_max = self.get_max(df, key)
                            t_std = self.get_std(df, key)

                            key = 'heart_rate_quality'
                            hr_q_mean = self.get_mean(df, key)
                            hr_q_min = self.get_min(df, key)
                            hr_q_max = self.get_max(df, key)
                            hr_q_std = self.get_std(df, key)

                            key = 'heart_rate_variability_quality'
                            hrv_q_mean = self.get_mean(df, key)
                            hrv_q_min = self.get_min(df, key)
                            hrv_q_max = self.get_max(df, key)
                            hrv_q_std = self.get_std(df, key)

                            key = 'core_temperature_quality'
                            t_q_mean = self.get_mean(df, key)
                            t_q_min = self.get_min(df, key)
                            t_q_max = self.get_max(df, key)
                            t_q_std = self.get_std(df, key)

                            key = 'oxygen_saturation_quality'
                            spo2_q_mean = self.get_mean(df, key)
                            spo2_q_min = self.get_min(df, key)
                            spo2_q_max = self.get_max(df, key)
                            spo2_q_std = self.get_std(df, key)

                            key = 'respiration_rate_quality'
                            rr_q_mean = self.get_mean(df, key)
                            rr_q_min = self.get_min(df, key)
                            rr_q_max = self.get_max(df, key)
                            rr_q_std = self.get_std(df, key)

                            key = 'activity_classification_quality'
                            ac_q_mean = self.get_mean(df, key)
                            ac_q_min = self.get_min(df, key)
                            ac_q_max = self.get_max(df, key)
                            ac_q_std = self.get_std(df, key)

                            key = 'energy_quality'
                            e_q_mean = self.get_mean(df, key)
                            e_q_min = self.get_min(df, key)
                            e_q_max = self.get_max(df, key)
                            e_q_std = self.get_std(df, key)

                        else:
                            ts_min = ts_max = valid_hours = hours = 0
                            hr_mean = hr_std = hr_min = hr_max = 0
                            hrv_mean = hrv_std = hrv_min = hrv_max = 0
                            spo2_mean = spo2_std = spo2_min = spo2_max = 0
                            rr_mean = rr_std = rr_min = rr_max = 0
                            t_mean = t_std = t_min = t_max = 0
                            hr_q_mean = hr_q_std = hr_q_min = hr_q_max = 0
                            hrv_q_mean = hrv_q_std = hrv_q_min = hrv_q_max = 0
                            spo2_q_mean = spo2_q_std = spo2_q_min = spo2_q_max = 0
                            rr_q_mean = rr_q_std = rr_q_min = rr_q_max = 0
                            t_q_mean = t_q_std = t_q_min = t_q_max = 0
                            ac_q_mean = ac_q_std = ac_q_min = ac_q_max = 0
                            e_q_mean = e_q_std = e_q_min = e_q_max = 0

                        csv_writer.writerow([id, "", line_count, valid_hours, ts_min, ts_max, hours,
                                             hr_mean, hr_std, hr_min, hr_max,
                                             hrv_mean, hrv_std, hrv_min, hrv_max,
                                             spo2_mean, spo2_std, spo2_min, spo2_max,
                                             rr_mean, rr_std, rr_min, rr_max,
                                             t_mean, t_std, t_min, t_max,
                                             hr_q_mean, hr_q_std, hr_q_min, hr_q_max,
                                             hrv_q_mean, hrv_q_std, hrv_q_min, hrv_q_max,
                                             spo2_q_mean, spo2_q_std, spo2_q_min, spo2_q_max,
                                             rr_q_mean, rr_q_std, rr_q_min, rr_q_max,
                                             t_q_mean, t_q_std, t_q_min, t_q_max,
                                             ac_q_mean, ac_q_std, ac_q_min, ac_q_max,
                                             e_q_mean, e_q_std, e_q_min, e_q_max])

                print("num files: ", len(files_sorted))


        def run_mixed_raw_vital(self, dir_name, id_range, in_file_suffix, out_file_name):
            csv_out_file = os.path.join(os.path.join(dir_name, os.pardir), out_file_name)
            if os.path.exists(csv_out_file):
                os.remove(csv_out_file)

            with open(csv_out_file, mode='a+') as out_file:
                csv_writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(['Id', 'Available', '#ValidRows', '#ValidHours',
                                     'timestamp_min', 'timestamp_max', 'hours',
                                     'HR_mean', 'HR_std', 'HR_min', 'HR_max',
                                     'HRV_mean', 'HRV_std', 'HRV_min', 'HRV_max',
                                     'Spo2_mean', 'Spo2_std', 'Spo2_min', 'Spo2_max',
                                     'RR_mean', 'RR_std', 'RR_min', 'RR_max',
                                     'T_mean', 'T_std', 'T_min', 'T_max',
                                     'HR_q_mean', 'HR_q_std', 'HR_q_min', 'HR_q_max',
                                     'Spo2_q_mean', 'Spo2_q_std', 'Spo2_q_min', 'Spo2_q_max',
                                     'AC_q_mean', 'AC_q_std', 'AC_q_min', 'AC_q_max'
                                     ])

                files_sorted = natsort.natsorted(os.listdir(dir_name))

                for count in range(id_range):
                    id = str(count + 1).zfill(3)

                    found = False
                    for i, filename in enumerate(files_sorted):
                        if filename.__contains__(id):
                            found = True

                    if not (found):
                        print("file not found with id: ", id)
                        csv_writer.writerow([id, 'na', '', '', '', '', '', '', ''])

                    else:
                        filename = id + in_file_suffix
                        print("processing file: ", filename, " ...")
                        file_path = os.path.join(dir_name, filename)

                        with open(file_path) as csv_file:
                            csv_reader = csv.reader(csv_file, delimiter=';')  # TODO: use df from below
                            line_count = 0
                            for row in csv_reader:
                                if line_count == 0:
                                    line_count += 1
                                    continue;
                                if int(row[0]) > 0:
                                    line_count += 1

                        if line_count > 0:
                            line_count -= 1  # -1 to subtract header row

                            valid_hours = line_count / 3600.0

                            df = self.loader.load_everion_patient_data(dir_name, filename, ';')

                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                            ts_max = max(df['timestamp'])
                            ts_min = min(df['timestamp'])
                            hours = pd.Timedelta(ts_max - ts_min).seconds / 3600.0
                            hours += pd.Timedelta(ts_max - ts_min).days * 24

                            key = 'HR'
                            hr_mean = self.get_mean(df, key)
                            hr_min = self.get_min(df, key)
                            hr_max = self.get_max(df, key)
                            hr_std = self.get_std(df, key)

                            key = 'HRV'
                            hrv_mean = self.get_mean(df, key)
                            hrv_min = self.get_min(df, key)
                            hrv_max = self.get_max(df, key)
                            hrv_std = self.get_std(df, key)

                            key = 'SPo2'
                            spo2_mean = self.get_mean(df, key)
                            spo2_min = self.get_min(df, key)
                            spo2_max = self.get_max(df, key)
                            spo2_std = self.get_std(df, key)

                            key = 'RespRate'
                            rr_mean = self.get_mean(df, key)
                            rr_min = self.get_min(df, key)
                            rr_max = self.get_max(df, key)
                            rr_std = self.get_std(df, key)

                            key = 'objtemp'
                            t_mean = self.get_mean(df, key)
                            t_min = self.get_min(df, key)
                            t_max = self.get_max(df, key)
                            t_std = self.get_std(df, key)

                            key = 'HRQ'
                            hr_q_mean = self.get_mean(df, key)
                            hr_q_min = self.get_min(df, key)
                            hr_q_max = self.get_max(df, key)
                            hr_q_std = self.get_std(df, key)

                            key = 'SPO2Q'
                            spo2_q_mean = self.get_mean(df, key)
                            spo2_q_min = self.get_min(df, key)
                            spo2_q_max = self.get_max(df, key)
                            spo2_q_std = self.get_std(df, key)

                            key = 'QualityClassification'
                            ac_q_mean = self.get_mean(df, key)
                            ac_q_min = self.get_min(df, key)
                            ac_q_max = self.get_max(df, key)
                            ac_q_std = self.get_std(df, key)

                        else:
                            ts_min = ts_max = valid_hours = hours = 0
                            hr_mean = hr_std = hr_min = hr_max = 0
                            hrv_mean = hrv_std = hrv_min = hrv_max = 0
                            spo2_mean = spo2_std = spo2_min = spo2_max = 0
                            rr_mean = rr_std = rr_min = rr_max = 0
                            t_mean = t_std = t_min = t_max = 0
                            hr_q_mean = hr_q_std = hr_q_min = hr_q_max = 0
                            spo2_q_mean = spo2_q_std = spo2_q_min = spo2_q_max = 0
                            ac_q_mean = ac_q_std = ac_q_min = ac_q_max = 0

                        csv_writer.writerow([id, "", line_count, valid_hours, ts_min, ts_max, hours,
                                             hr_mean, hr_std, hr_min, hr_max,
                                             hrv_mean, hrv_std, hrv_min, hrv_max,
                                             spo2_mean, spo2_std, spo2_min, spo2_max,
                                             rr_mean, rr_std, rr_min, rr_max,
                                             t_mean, t_std, t_min, t_max,
                                             hr_q_mean, hr_q_std, hr_q_min, hr_q_max,
                                             spo2_q_mean, spo2_q_std, spo2_q_min, spo2_q_max,
                                             ac_q_mean, ac_q_std, ac_q_min, ac_q_max])

                print("num files: ", len(files_sorted))


        def run_imove(self, dir_name, id_range, in_file_suffix, out_file_name):
            csv_out_file = os.path.join(os.path.join(dir_name, os.pardir), out_file_name)
            if os.path.exists(csv_out_file):
                os.remove(csv_out_file)

            with open(csv_out_file, mode='a+') as out_file:
                csv_writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(['Id', 'Available', '#ValidRows', '#ValidHours',
                                     'timestamp_min', 'timestamp_max', 'hours',
                                     'HR_mean', 'HR_std', 'HR_min', 'HR_max',
                                     'A_mean', 'A_std', 'A_min', 'A_max',
                                     'AC_mean', 'AC_std', 'AC_min', 'AC_max',
                                     'HR_q_mean', 'HR_q_std', 'HR_q_min', 'HR_q_max',
                                     'Spo2_q_mean', 'Spo2_q_std', 'Spo2_q_min', 'Spo2_q_max',
                                     'AC_q_mean', 'AC_q_std', 'AC_q_min', 'AC_q_max'
                                     ])

                files_sorted = natsort.natsorted(os.listdir(dir_name))

                for count in range(id_range):
                    id = str(count + 1).zfill(3)

                    found = False
                    for i, filename in enumerate(files_sorted):
                        if filename.__contains__(id):
                            found = True

                    if not (found):
                        print("file not found with id: ", id)
                        csv_writer.writerow([id, 'na', '', '', '', '', '', '', ''])

                    else:
                        filename = id + 'L' + in_file_suffix
                        print("processing file: ", filename, " ...")
                        file_path = os.path.join(dir_name, filename)
                        if os.path.exists(file_path):
                            self.write_row_one_side(csv_writer, dir_name, file_path, filename, id, 'L')

                        filename = id + 'R' + in_file_suffix
                        print("processing file: ", filename, " ...")
                        file_path = os.path.join(dir_name, filename)
                        if os.path.exists(file_path):
                            self.write_row_one_side(csv_writer, dir_name, file_path, filename, id, 'R')

                print("num files: ", len(files_sorted))

        def write_row_one_side(self, csv_writer, dir_name, file_path, filename, id, side):
            with open(file_path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=';')  # TODO: use df from below
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        line_count += 1
                        continue;
                    if int(row[0]) > 0:
                        line_count += 1
            if line_count > 0:
                line_count -= 1  # -1 to subtract header row

                valid_hours = line_count / 3600.0

                df = self.loader.load_everion_patient_data(dir_name, filename, ';')

                df['timestamp'] = pd.to_datetime(df['timestamp'])
                ts_max = max(df['timestamp'])
                ts_min = min(df['timestamp'])
                hours = pd.Timedelta(ts_max - ts_min).seconds / 3600.0
                hours += pd.Timedelta(ts_max - ts_min).days * 24

                key = 'HR'
                hr_mean = self.get_mean(df, key)
                hr_min = self.get_min(df, key)
                hr_max = self.get_max(df, key)
                hr_std = self.get_std(df, key)

                key = 'Activity'
                a_mean = self.get_mean(df, key)
                a_min = self.get_min(df, key)
                a_max = self.get_max(df, key)
                a_std = self.get_std(df, key)

                key = 'Classification'
                ac_mean = self.get_mean(df, key)
                ac_min = self.get_min(df, key)
                ac_max = self.get_max(df, key)
                ac_std = self.get_std(df, key)

                key = 'HRQ'
                hr_q_mean = self.get_mean(df, key)
                hr_q_min = self.get_min(df, key)
                hr_q_max = self.get_max(df, key)
                hr_q_std = self.get_std(df, key)

                key = 'SPO2Q'
                spo2_q_mean = self.get_mean(df, key)
                spo2_q_min = self.get_min(df, key)
                spo2_q_max = self.get_max(df, key)
                spo2_q_std = self.get_std(df, key)

                key = 'QualityClassification'
                ac_q_mean = self.get_mean(df, key)
                ac_q_min = self.get_min(df, key)
                ac_q_max = self.get_max(df, key)
                ac_q_std = self.get_std(df, key)

            else:
                ts_min = ts_max = valid_hours = hours = 0
                hr_mean = hr_std = hr_min = hr_max = 0
                a_mean = a_std = a_min = a_max = 0
                ac_mean = ac_std = ac_min = ac_max = 0
                hr_q_mean = hr_q_std = hr_q_min = hr_q_max = 0
                spo2_q_mean = spo2_q_std = spo2_q_min = spo2_q_max = 0
                ac_q_mean = ac_q_std = ac_q_min = ac_q_max = 0
            csv_writer.writerow([id+side, "", line_count, valid_hours, ts_min, ts_max, hours,
                                 hr_mean, hr_std, hr_min, hr_max,
                                 a_mean, a_std, a_min, a_max,
                                 ac_mean, ac_std, ac_min, ac_max,
                                 hr_q_mean, hr_q_std, hr_q_min, hr_q_max,
                                 spo2_q_mean, spo2_q_std, spo2_q_min, spo2_q_max,
                                 ac_q_mean, ac_q_std, ac_q_min, ac_q_max])

        def get_std(self, df, key):
            return df[key].describe().loc[['std']][0]

        def get_max(self, df, key):
            return df[key].describe().loc[['max']][0]

        def get_min(self, df, key):
            return df[key].describe().loc[['min']][0]

        def get_mean(self, df, key):
            return df[key].describe().loc[['mean']][0]


