import pandas as pd
from pathlib import Path

from typing import Union, Tuple, Optional
PathLike = Union[str, Path]
PandasData = Union[pd.DataFrame, pd.Series]


def ensure_dir(path: PathLike,
               exist_ok: bool=True) -> bool:
    """
    Path.mkdir() usually raises if the folder cannot be created.
    """
    path = Path(path)
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=exist_ok)
    return path.is_dir()


def strip_path_annotation(path: PathLike,
                          ext: str) -> Tuple[Path, Optional[str]]:
    """
    Purpose: Strip extra postfix info from path. Required mainly for
             the .h5 target, where the groups can be added to the path:
                path/to/file.h5/group/object
    Examples:
        f("./file.csv", ".csv")                -> ./file.csv
        f("./files.csv/file.csv", ".csv")      -> ./files.csv/file.csv
        f("./file.h5/group/object", ".h5")     -> ./file.h5
        f("./.h5/file.h5/group/object", ".h5") -> ./.h5/file.h5
    """
    path = Path(path)
    parts = list(path.parts)
    annot = []
    while bool(parts) and not parts[-1].endswith(ext):
        annot.append(parts.pop(-1))
    annot = "/".join(reversed(annot)) if annot else None
    if parts:
        return Path(*parts), annot
    else:
        # Return original by default.
        return path, annot


def write_csv(df: PandasData,
              path: PathLike,
              sep: str=",",
              **kwargs) -> bool:
    path = Path(path)
    if ensure_dir(path.parent):
        df.to_csv(path, sep=sep, **kwargs)
    return path.is_file()


def write_hdf(df: PandasData,
              path: PathLike,
              key: Optional[str]=None,
              format: str="table",
              mode: str="a",
              **kwargs) -> bool:
    """
    Convention: out_path = "path/to/file.h5/sub/path"
                is equivalent to
                out_path = "path/to/file.h5"
                key = "sub/path" if key is None else key

    See also my notes here for some understanding:
        https://stackoverflow.com/a/67066662/3388962
    """
    path, annot = strip_path_annotation(path=path, ext=".h5")
    msg = "Inconsistent specification of storage key: %s, %s"
    assert not (annot and key) or annot==key, msg % (key, annot)
    key = key if key else annot
    if ensure_dir(path.parent):
        df.to_hdf(path, key=key, mode=mode, format=format, **kwargs)
    return path.is_file()


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
        in_dir = Path("") if in_dir is None else Path(in_dir)
        out_dir = Path(in_dir) if out_dir is None else Path(out_dir)
        out_dir_name = in_dir.name + out_dir_suffix
        out_dir = out_dir / out_dir_name

        if not out_dir.is_dir():
            out_dir.mkdir(parents=True)

        return out_dir

