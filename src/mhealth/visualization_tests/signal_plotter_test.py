import os
import glob
import shutil
import unittest

from ..utils import testing
from ..utils import everion_keys
from ..utils.file_helper import FileHelper
from ..visualization.signal_plotter import SignalPlotter

_MHEALTH_DATA = os.getenv('MHEALTH_DATA', '../../resources')
_MHEALTH_OUT_DIR = os.path.join(_MHEALTH_DATA, 'output')


class SignalPlotterTest(testing.TestCase):
    in_dir = f'{_MHEALTH_DATA}/vital_signals/'
    out_dir = _MHEALTH_OUT_DIR
    plotter = SignalPlotter()


    def test_plot_one_signal(self):
        out_dir = FileHelper.get_out_dir(in_dir=self.in_dir,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_per-signal')
        if out_dir.is_dir():
            shutil.rmtree(out_dir)

        self.plotter.plot_signal(in_dir=self.in_dir,
                                 out_dir=out_dir,
                                 signal_name='heart_rate')
        files = list(out_dir.glob('**/*.png'))
        self.assertEqual(1, len(files))


    def test_plot_all_signals(self):
        out_dir = FileHelper.get_out_dir(in_dir=self.in_dir,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_per-signal')
        if out_dir.is_dir():
            shutil.rmtree(out_dir)

        signals = {'heart_rate', 'heart_rate_variability',
                   'oxygen_saturation', 'core_temperature',
                   'respiration_rate'}
        for signal in signals:
            print('processing ', signal, ' ...')
            self.plotter.plot_signal(in_dir=self.in_dir,
                                     out_dir=out_dir,
                                     signal_name=signal)
        files = list(out_dir.glob('**/*.png'))
        self.assertEqual(5, len(files))


    @testing.skip_because_is_runner
    def test_plot_signals_vital(self):
        dir_name = ''
        out_dir = FileHelper.get_out_dir(in_dir=dir_name,
                                         out_dir=self.out_dir,
                                         out_dir_suffix='_signals')
        signals = everion_keys.MAJOR_VITAL
        for signal in signals:
            print('processing ', signal, ' ...')
            self.plotter.plot_signal(in_dir=dir_name,
                                     out_dir=out_dir,
                                     signal_name=signal)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
