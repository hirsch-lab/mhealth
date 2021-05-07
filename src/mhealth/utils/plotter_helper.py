import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from .signal_properties import SignalProperties
from .file_helper import ensure_counted_path


from typing import Union, Tuple, Optional
PathLike = Union[str, Path]
OptionalFigure = Optional[mpl.figure.Figure]


def save_figure(path: PathLike="./plot.pdf",
                fig: OptionalFigure=None,
                override: bool=True,
                skip_first: bool=True,
                **kwargs) -> Path:
    """
    Save current figure to path: <outdir>/<ext>/<basename>.<ext>
    If fig is None, the current figure is used. **kwargs is used
    synonymously for the settings dict.

    Arguments:
        path:       Path where the image should be saved to. The file
                    extension determines how the plot is printed.
                    Parent directories are created if not existent.
        override:   If False and if the target file already exists,
                    path = ensure_counted_path(path)
        skip_first: Applies if override=True. Controls if the
                    first file is postfixed by a count or not.
        **kwargs:   Additional keyword arguments are forwarded to
                    plt.savefig
    """
    dpi = kwargs.pop("dpi", None)
    bbox_inches = kwargs.pop("bbox_inches", "tight")
    transparent = kwargs.pop("transparent", False)
    if fig is not None:
        plt.figure(fig.number)
    path = ensure_counted_path(path=path,
                               skip_first=skip_first,
                               enabled=not override)
    plt.savefig(path,
                transparent=transparent,
                bbox_inches="tight",
                dpi=dpi,
                **kwargs)
    return path


def setup_plotting():
    sns.set_theme(style="ticks", palette="pastel")
    # Make text in PDFs editable in Adobe Illustrator
    import matplotlib as mpl
    mpl.rcParams["pdf.fonttype"] = 42


class PlotterHelper:

    @staticmethod
    def get_min_scale(key, signal):
        min_scale = SignalProperties.vital[key]['min_scale']
        if np.isnan(min_scale):
            min_scale = signal.min()
        return min_scale

    @staticmethod
    def get_max_scale(key, signal):
        max_scale = SignalProperties.vital[key]['max_scale']
        if np.isnan(max_scale):
            max_scale = signal.max()
        return max_scale

    @staticmethod
    def get_admitted_days(admitted_date, discharge_date):
        hours = pd.Timedelta(discharge_date - admitted_date).seconds / 3600.0
        hours += pd.Timedelta(discharge_date - admitted_date).days * 24
        days = np.round(hours / 24, decimals=1)
        return days

    @staticmethod
    def get_step_size(len_index):
        step = 1
        if len_index >= 24:
            step = 6
        if len_index >= 6 * 24:
            step = 12
        if len_index >= 10 * 24:
            step = 24
        return step

    @staticmethod
    def get_hourly_x_tick_labels(multi_index, offset, step):
        multi_index_sub = multi_index[offset::step]
        if len(multi_index_sub) <= 0:
            multi_index_sub = multi_index

        m = multi_index_sub.to_frame()
        x_tick_labels_d = m.apply(lambda row: PlotterHelper.row_to_datetime(row), axis=1)

        return x_tick_labels_d

    @staticmethod
    def row_to_datetime(row):
        d = datetime(row['year'], row['month'], row['day'], row['hour'])
        if row['hour'] == 0:
            return d.strftime("%d.%m. %H:00")
        return d.strftime("%H:00")

    @staticmethod
    def get_plot_title(patient_id, admitted_date, discharge_date):
        return 'Patient: ' + patient_id + ' (' + str(
            PlotterHelper.get_admitted_days(admitted_date, discharge_date)) + ' data days from ' + str(
            admitted_date) + ' to ' + str(discharge_date) + ')'

    @staticmethod
    def get_dummy_plot_title(admitted_date, discharge_date):
        return 'Patient: xxx (admitted ' + str(
            PlotterHelper.get_admitted_days(admitted_date, discharge_date)) + ' days)'


    @staticmethod
    def custom_subplots(axn, custom_plot_fct, df_hm, font_size, x, x_daily_lines, x_ticks):
        counter = 0
        for (key, signal) in df_hm.iteritems():
            if signal.isna().all():
                continue
            ax = axn.flat[counter]
            custom_plot_fct(ax, font_size, key, signal, x)

            ax.set_ylabel(key, fontsize=font_size)
            ax.set_xticks(x_ticks)

            ax.vlines(x_ticks, *ax.get_ylim(), color='k', linestyles='--', linewidth=0.25)
            ax.vlines(x_daily_lines, *ax.get_ylim(), color='k', linestyles='--', linewidth=0.75)

            counter += 1

    @staticmethod
    def save_custom_plots(out_file_path, df, df_hm, patient_id, custom_subplots, custom_plot_fct, hspace, fig_width,
                          fig_height):
        num_subplots = 0
        for (key, signal) in df_hm.iteritems():
            if signal.isna().all():
                continue
            num_subplots += 1
        if num_subplots <= 0:
            return

        fig, axn = plt.subplots(num_subplots, 1, sharex=True, sharey=False, figsize=(fig_width, fig_height))
        plt.subplots_adjust(hspace=hspace)
        font_size = 10

        multi_index = df_hm.index
        index = df_hm.reset_index(level=['year', 'month', 'day']).index
        if len(index) < 1:
            return

        step = PlotterHelper.get_step_size(len(index))
        offset = step - (index[0] % step)
        x_tick_labels = PlotterHelper.get_hourly_x_tick_labels(multi_index, offset, step)
        x_ticks = np.arange(offset, len(index), step)
        if len(x_ticks) <= 0:
         x_ticks = np.arange(0.5, len(index), 1)
        x = np.arange(0, len(index), 1)

        offset_lines = 24 - (index[0] % 24)
        x_daily_lines = np.arange(offset_lines, len(index), 24)

        custom_subplots(axn, custom_plot_fct, df_hm, font_size, x, x_daily_lines, x_ticks)

        fig.suptitle(PlotterHelper.get_plot_title(patient_id, df['timestamp'].dt.tz_convert('Europe/Zurich').min(),
                                                  df['timestamp'].dt.tz_convert('Europe/Zurich').max()),
                     fontsize=font_size)

        plt.xticks(x_ticks, x_tick_labels, rotation=90, fontsize=font_size)
        plt.xlabel("Time [h]", fontsize=font_size)

        plt.savefig(out_file_path, bbox_inches='tight')
        plt.close(fig)


    def get_bar_color(df, color1, color2):
        bar = np.where(df.values > 0, color1, color2).T
        return bar
