from pathlib import Path


class FileHelper:

    @staticmethod
    def get_out_dir(in_dir, out_dir, out_dir_suffix=""):
        """
        Construct and create output directory:
            out_dir = out_dir / (in_dir.name+out_dir_suffix)
        """
        in_dir = Path(in_dir)
        out_dir = Path(out_dir)
        out_dir_name = in_dir.name + out_dir_suffix
        out_dir = out_dir / out_dir_name

        if not out_dir.is_dir():
            out_dir.mkdir(parents=True)

        return out_dir

