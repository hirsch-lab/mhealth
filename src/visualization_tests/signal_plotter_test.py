import glob
import os
import shutil
import unittest

from utils.everion_keys import EverionKeys
from visualization.signal_plotter import SignalPlotter
from utils.file_helper import FileHelper


class SignalPlotterTest(unittest.TestCase):
        in_dir = '../resources/vital_signals/'
        plotter = SignalPlotter()
        out_dir = FileHelper.get_out_dir(in_dir, '_per-signal')

        def test_plot_one_signal(self):
            if os.path.exists(self.out_dir):
                shutil.rmtree(self.out_dir)

            self.plotter.plot_signal(self.in_dir, self.out_dir, 'heart_rate')
            files = glob.glob(os.path.join(os.path.join(self.out_dir, '**'), '*.png'), recursive=True)
            self.assertEqual(1, len(files))

        def test_plot_all_signals(self):
            if os.path.exists(self.out_dir):
                shutil.rmtree(self.out_dir)

            signals = {'heart_rate', 'heart_rate_variability', 'oxygen_saturation', 'core_temperature', 'respiration_rate'}
            for signal in signals:
                print('processing ', signal, ' ...')
                self.plotter.plot_signal(self.in_dir, self.out_dir, signal)
            files = glob.glob(os.path.join(os.path.join(self.out_dir, '**'), '*.png'), recursive=True)
            self.assertEqual(5, len(files))

        @unittest.SkipTest
        def test_plot_signals_vital(self):
            dir_name = ''
            out_dir = FileHelper.get_out_dir(dir_name, '_signals')

            signals = EverionKeys.major_vital
            for signal in signals:
                print('processing ', signal, ' ...')
                self.plotter.plot_signal(dir_name, out_dir, signal)
            self.assertTrue(True)




if __name__ == '__main__':
    unittest.main()
