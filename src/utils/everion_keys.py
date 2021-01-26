
"""
All vital keys (without signal quality).
"""
ALL_VITAL = [
    "activity_classification",
    "activity_score",
    "barometer_pressure",
    "blood_pulse_wave",
    "core_temperature",
    "energy",
    "gsr_electrode",
    "health_score",
    "heart_rate",
    "heart_rate_variability",
    "motion_activity",
    "number_of_steps",
    "oxygen_saturation",
    "perfusion_index",
    "relax_stress_intensity_score",
    "respiration_rate",
    "richness_score",
    "sleep_quality_index_score",
    "temperature_barometer",
    "temperature_local",
    "temperature_object",
    "training_effect_score",
]


"""
Subset of vital parameters considered important.
"""
MAJOR_VITAL = [
    "heart_rate",
    "oxygen_saturation",
    "heart_rate_variability",
    "respiration_rate",
    "core_temperature"
]
assert set(MAJOR_VITAL) <= set(ALL_VITAL)


"""
Subset of vital parameters considered important for iMove project.
"""
MAJOR_IMOVE = [
    "HR", "Activity", "Classification", "BloodPressure"
]


"""
Subset of vital parameters considered important (short names).
Refers to code table: raw_data_data/usb/imove/data/iMove_00_data_info.txt
"""
MAJOR_MIXED_VITAL_RAW = [
    "HR", "HRV", "SPo2", "objtemp", "RespRate"
]


"""
Short names of vital parameters (with signal qualities).
"""
SHORT_NAMES_VITAL = {
    "activity_classification": "AC",
    "activity_classification_quality": "ACQ",
    "activity_score": "AS",
    "barometer_pressure": "PB",
    "blood_pulse_wave": "BPW",
    "core_temperature": "Temp",
    "core_temperature_quality": "CTempQ",
    "energy": "E",
    "energy_quality": "EQ",
    "gsr_electrode": "GSR",
    "health_score": "Health",
    "heart_rate": "HR",
    "heart_rate_quality": "HRQ",
    "heart_rate_variability": "HRV",
    "heart_rate_variability_quality": "HRVQ",
    "motion_activity": "Motion",
    "number_of_steps": "Steps",
    "oxygen_saturation": "SPO2",
    "oxygen_saturation_quality": "SPO2Q",
    "perfusion_index": "PI",
    "relax_stress_intensity_score": "RStressI",
    "respiration_rate": "RR",
    "respiration_rate_quality": "RRQ",
    "richness_score": "RS",
    "sleep_quality_index_score": "Sleep",
    "temperature_barometer": "TBaro",
    "temperature_local": "TmpL",
    "temperature_object": "TObj",
    "training_effect_score": "Training",
}
assert set(ALL_VITAL) <= set(SHORT_NAMES_VITAL)


"""
"""
SHORT_NAMES_MIXED_VITAL_RAW = {
    "HR": "HR",
    "SPo2": "SPO2",
    "BloodPerfusion": "BPerf",
    "Activity": "A",
    "Classification": "AC",
    "QualityClassification": "QAC",
    "HRV": "HRV",
    "RespRate": "RR",
    "Energy": "E",
    "localtemp": "TmpL",
    "HRQ": "HRQ",
    "SPO2Q": "SPO2Q",
    "BloodPressure": "BP",
    "steps": "Steps",
    "objtemp": "Temp",
    "baromtemp": "TBaro",
    "phase": "phase",
    "pressure": "pressure",
}




"""
Units for the vital parameters.
TODO: Unclear what the keys are.
TODO: Unused variable. Remove?
"""
TAG_UNITS_VITAL = {
    "6": "bpm",
    "7": "%",
    "8": "1",
    "9": "1",
    "11": "ms",
    "12": "bpm",
    "13": "cal/s",
    "15": "degC",
    "19": "degC",
    "20": "mbar",
    "21": "kOhm",
    "22": "%",
    "23": "%",
    "24": "%",
    "25": "%",
    "26": "%",
    "66": "%",
    "68": "%",
    "69": "%",
    "70": "1",
    "71": "1",
    "72": "%",
    "73": "%",
    "74": "%",
    "75": "%",
    "76": "%",
    "118": "degC",
    "119": "degC"
}


"""
TODO: Unused variable. Remove?
"""
TAG_NAMES_VITAL = {
    "6": "heart_rate",
    "7": "oxygen_saturation",
    "8": "perfusion_index",
    "9": "motion_activity",
    "10": "activity_classification",
    "11": "heart_rate_variability",
    "12": "respiration_rate",
    "13": "energy",
    "15": "core_temperature",
    "19": "temperature_local",
    "20": "barometer_pressure",
    "21": "gsr_electrode",
    "22": "health_score",
    "23": "relax_stress_intensity_score",
    "24": "sleep_quality_index_score",
    "25": "training_effect_score",
    "26": "activity_score",
    "66": "richness_score",
    "68": "heart_rate_quality",
    "69": "oxygen_saturation_quality",
    "70": "blood_pulse_wave",
    "71": "number_of_steps",
    "72": "activity_classification_quality",
    "73": "energy_quality",
    "74": "heart_rate_variability_quality",
    "75": "respiration_rate_quality",
    "76": "core_temperature_quality",
    "118": "temperature_object",
    "119": "temperature_barometer"
}


"""
TODO: Unused variable. Remove?
"""
TAG_NAMES_MIXED_VITAL_RAW = {
    "2.1":  "HR",
    "4.1":  "HRQ",
    "6.1":  "SPo2",
    "8.1":  "SPO2Q",
    "10.1": "BloodPressure",
    "14.1": "BloodPerfusion",
    "18.1": "Activity",
    "20.1": "Classification",
    "22.1": "QualityClassification",
    "24.1": "steps",
    "26.1": "Energy",
    "28.1": "RespRate",
    "29.1": "HRV",
    "42.1": "phase",
    "44.1": "phase",
    "46.1": "localtemp",
    "48.1": "objtemp",
    "50.1": "baromtemp",
    "52.1": "pressure",
}


"""
TODO: Unused variable. Remove?
"""
ALL_SIGNALS_MIXED_VITAL_RAW = [
    "HR", "HRQ", "SPo2", "SPO2Q", "BloodPressure",
    "BloodPerfusion", "Activity", "Classification",
    "QualityClassification", "steps", "Energy", "RespRate",
    "HRV", "phase", "phase", "localtemp", "objtemp", "baromtemp", "pressure"
]


"""
TODO: Unused variable. Remove?
"""
MEDIUM_MIXED_VITAL_RAW = [
    "HR", "HRV", "SPo2", "BloodPressure", "Activity",
    "Classification", "steps", "Energy", "RespRate"
]


"""
TODO: Unused variable. Remove?
"""
ACTIVITY_CLASS = {
    "0" : "undefined",
    "1" : "resting",
    "2" : "walking_flat",
    "3" : "running_flat",
    "4" : "biking_flat",
    "5" : "walking_up",
    "6" : "running_up",
    "7" : "biking_up",
    "8" : "rowing",
    "9" : "other",
    "10" : "biking",
    "11" : "running",
    "12" : "walking",
    "13" : "walking_down",
    "14" : "running_down",
    "15" : "biking_down",
    "16" : "sitting",
    "17" : "standing",
    "18" : "driving_car",
    "19" : "driving_public",
    "20" : "sleeping",
    "21" : "awake",
    "22" : "ctrl_rest_med_ee",
    "23" : "relax_payin",
    "24" : "sleep_payin",
    "25" : "exercise_payin",
    "26" : "move_payin"
}
