import os
from pathlib import Path


class FileHelper:

    @staticmethod
    def get_out_dir(in_dir, out_dir_suffix):
        out_dir_name = Path(in_dir).name + out_dir_suffix
        out_dir = os.path.join(os.path.join(in_dir, os.pardir), out_dir_name)

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        return out_dir

