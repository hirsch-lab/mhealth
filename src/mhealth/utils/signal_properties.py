import numpy as np
from matplotlib.colors import LinearSegmentedColormap


class SignalProperties:
    num_discrete_scales = 7
    colors_d = ['#0571b0', '#92c5de', '#f7f7f7', '#f4a582', '#ca0020']  # from colorbrewer.org
    #colors_d = [  '#2c7bb6', '#abd9e9', '#ffffbf', '#fdae61', '#d7191c'] # from colorbrewer.org
    colors_d2 = ['#0571b0', '#b2df8a', '#ca0020']  # mix from colorbrewer.org
    colors_d3 = ['#2b83ba', '#66c2a5', '#abdda4', '#ffffbf', '#d7191c']  # mix from colorbrewer.org

    diverging = LinearSegmentedColormap.from_list(name='vitals_diverging', colors=colors_d3)

    colors_s = ['#cb181d', '#fb6a4a', '#fcae91', '#fee5d9', '#f7f7f7']  # from colorbrewer.org
    #colors_s = ['#bd0026', '#f03b20', '#fd8d3c', '#fecc5c', '#ffffb2']  # from colorbrewer.org
    colors_s2 = ['#cb181d', '#b2df8a']  # from colorbrewer.org
    sequential = LinearSegmentedColormap.from_list(name='vitals_sequential', colors=colors_s2)

    colors_s3 = ['#0571b0', '#b2df8a']  # from colorbrewer.org
    sequential2 = LinearSegmentedColormap.from_list(name='vitals_sequential', colors=colors_s3)

    vital = {
        # according to USB: RR: 10-15
        'RR': {'min_scale': np.nan, 'max_scale': np.nan, 'expected_min': 10, 'expected_max': 15, 'color_map': diverging},
        # according to USB: 37.5
        'Temp': {'min_scale': np.nan, 'max_scale': np.nan, 'expected_min': np.nan, 'expected_max': np.nan, 'color_map': diverging},
        # according to USB: 95-100%
        'SpO2': {'min_scale': 80, 'max_scale': 100, 'expected_min': 95, 'expected_max': 100, 'color_map': sequential2},
        # according to USB: HR : 60-80
        'HR': {'min_scale': np.nan, 'max_scale': np.nan, 'expected_min': 60, 'expected_max': 80, 'color_map': diverging},
        # according to USB: HRV: RMSSD altersabhängig ca. 25 bei 50 jährigen und 20 bei 80 jährigen
        'HRV': {'min_scale': np.nan, 'max_scale': np.nan, 'expected_min': np.nan, 'expected_max': np.nan, 'color_map': sequential2},
        'A': {'min_scale': np.nan, 'max_scale': np.nan, 'expected_min': np.nan, 'expected_max': np.nan,
               'color_map': diverging},
        'AC': {'min_scale': np.nan, 'max_scale': np.nan, 'expected_min': np.nan, 'expected_max': np.nan,
               'color_map': sequential2},
        'BP': {'min_scale': np.nan, 'max_scale': np.nan, 'expected_min': np.nan, 'expected_max': np.nan,
               'color_map': sequential2},
    }
