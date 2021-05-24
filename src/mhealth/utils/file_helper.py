import re
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


def ensure_counted_path(path: PathLike,
                        fmt: str="-%03d",
                        skip_first: bool=False,
                        start: int=1,
                        step: int=1,
                        ensure_parent: bool=True,
                        enabled: bool=True) -> Path:
    """
    Append a formatted count to a file name to ensure that the path will not
    collide with an existing file or folder.

    Arguments:
        path:           Target path for file or folder
        fmt:            Format for indexer
        skip_first:     If True, path is returned unmodified if the item
                        does not exist
        start:          Count of the first item
        step:           Step by which the current count is incremented
        ensure_parent:  Ensure parent directory
        enabled:        If False, this function simplifies to ensure_dir()

    Examples:
        - File already exists in folder:
            ensure_counted_path("./file.txt")    => file-001.txt
        - File already exists 10x in folder:
            ensure_counted_path("./file.txt")    => file-010.txt
        - File does not exist yet
            ensure_counted_path("./file.txt", first=False) => file.txt
            ensure_counted_path("./file.txt", first=True)  => file-001.txt
    """
    def _construct_path(path, fmt, count):
        if path.exists() and count is None and skip_first:
            count = start
        if count is None:
            count = None if skip_first else start
        if count is None:
            return path
        else:
            count = max(count, start)
            return path.parent / (path.stem + fmt%count + path.suffix)

    # Check arguments.
    assert isinstance(fmt, str)
    assert isinstance(skip_first, bool)
    assert isinstance(step, int) and step > 0
    assert isinstance(start, int) and start >= 0
    assert isinstance(enabled, bool)
    # Match patterns such as the following ones:
    #   "%d", "%10d", "-prefix-%03d-suffix", "%04d-suffix-%s"
    match = re.match(".*(%0?[0-9]*d).*", fmt)
    if not match:
        raise ValueError("Invalid format specifier: %s" % fmt)

    path = Path(path)
    parent = path.parent
    if ensure_parent:
        ensure_dir(parent)
    if not enabled:
        return path
    if not parent.exists():
        return _construct_path(path, fmt, count=None)

    # Extract the counts of existing items      # Example:
    # that match with the current one.          # fmt = "-pre-%03d"
    pattern = match.group(1)                    # pattern = "%03d"
    fmt_regex = fmt.replace(pattern,"([0-9]*)") # fmt_regex = "-pre-([0-9]*)"

    stem = path.stem                            # stem = "file"
    suffix = path.suffix                        # suffix = ".txt"
    regex = re.compile("%s%s$" %                # regex = "file-pre-([0-9]*)"
                       (stem, fmt_regex))

    matches = [regex.match(f.stem) for f in parent.glob(stem+"*"+suffix)]
    matches = [m.group(1) for m in matches if m]
    counts = [int(m) for m in matches if m]
    new_count = max(counts)+step if counts else None
    ensured_path = _construct_path(path, fmt, count=new_count)
    assert not ensured_path.exists()
    return ensured_path


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
              key: Optional[str]=None,
              key_sep: str="-",
              exist_ok: bool=True,
              return_path: bool=False,
              **kwargs) -> Union[bool, Path, None]:
    path = Path(path)
    if key is not None:
        path = path.parent / (path.stem + key_sep + key + path.suffix)
    if not exist_ok and path.is_file():
        msg = "File already exists: %s" % path
        raise FileExistsError(msg)
    if ensure_dir(path.parent):
        df.to_csv(path, sep=sep, **kwargs)
    if return_path:
        return path if path.is_file() else None
    else:
        return path.is_file()


def read_csv(path: PathLike,
             sep: str=",",
             key: Optional[str]=None,
             key_sep: str="-",
             infer_key: bool=False,
             **kwargs) -> pd.DataFrame:
    path = Path(path)
    ret_key = False
    if key is not None:
        path = path.parent / (path.stem + key_sep + key + path.suffix)
    elif infer_key:
        parent = path.parent
        name = path.stem
        ret = re.match("^(.*)"+re.escape(key_sep)+"(.*)$", name)
        if not ret:
            msg = "Cannot infer key from filepath (separator: '%s'): %s"
            raise ValueError(msg % (key_sep, path))
        key = ret.group(2)
        ret_key = True
    df = pd.read_csv(path, sep=sep, **kwargs)
    return (df, key) if ret_key else df


def write_hdf(df: PandasData,
              path: PathLike,
              key: Optional[str]=None,
              format: str="table",
              mode: str="a",
              exist_ok: bool=True,
              return_path: bool=False,
              **kwargs) -> Union[bool, Path, None]:
    """
    Convention: path = "path/to/file.h5/sub/path"
                is equivalent to
                path = "path/to/file.h5"
                key = "sub/path" if key is None else key

    See also my notes here for some understanding:
        https://stackoverflow.com/a/67066662/3388962
    """
    path, annot = strip_path_annotation(path=path, ext=".h5")
    if not exist_ok and path.is_file():
        msg = "File already exists: %s" % path
        raise FileExistsError(msg)
    msg = "Inconsistent specification of storage key: %s, %s"
    assert not (annot and key) or annot==key, msg % (key, annot)
    key = key if key else annot
    if ensure_dir(path.parent):
        df.to_hdf(path, key=key, mode=mode, format=format, **kwargs)
    if return_path:
        return path if path.is_file() else None
    else:
        return path.is_file()


def read_hdf(path: PathLike,
             key: Optional[str]=None,
             format: str="table",
             mode: str="r",
             **kwargs) -> pd.DataFrame:
    """
    Convention: path = "path/to/file.h5/sub/path"
                is equivalent to
                out_path = "path/to/file.h5"
                key = "sub/path" if key is None else key
    """
    path, annot = strip_path_annotation(path=path, ext=".h5")
    msg = "Inconsistent specification of storage key: %s, %s"
    assert not (annot and key) or annot==key, msg % (key, annot)
    key = key if key else annot
    return pd.read_hdf(path, key=key, mode=mode, format=format, **kwargs)


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

