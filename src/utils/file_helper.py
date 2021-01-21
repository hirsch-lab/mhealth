from pathlib import Path


class FileHelper:

    @staticmethod
    def get_out_dir(in_dir=None, out_dir=None, out_dir_suffix=""):
        """
        Construct and create output directory.
        If out_dir is None: out_dir = in_dir.
        If all arguments are provided, ensure directory with pattern:
            ret = out_dir / (in_dir.name+out_dir_suffix)
        To ensure a directory:
            get_out_dir(out_dir=path)
        To ensure a directory with suffix
            get_out_dir(out_dir=path, out_dir_suffix="suffix")
        """
        assert in_dir is not None or out_dir is not None
        in_dir = Path(out_dir) if in_dir is None else Path(in_dir)
        out_dir = Path(in_dir) if out_dir is None else Path(out_dir)
        out_dir_name = in_dir.name + out_dir_suffix
        out_dir = out_dir / out_dir_name

        if not out_dir.is_dir():
            out_dir.mkdir(parents=True)

        return out_dir

