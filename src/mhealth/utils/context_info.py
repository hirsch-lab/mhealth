

import os
import sys
import git                  # package: GitPython
import psutil
import inspect
import datetime
import platform             # system information
import multiprocessing      # cpu_count()
from pathlib import Path
import importlib.metadata as import_meta

from .file_helper import ensure_dir, ensure_counted_path

from typing import Union, Dict, Tuple, Optional, Callable, Any, get_args
PathLike = Union[str, Path]
OptionalPath = Optional[PathLike]
DumpFunction = Callable[[PathLike], Any]
ExtraContextDict = Dict[str, DumpFunction]


# Template text (not in a separate file to avoid an additional dependency):
INFO_TEMPLATE = \
    "Context information\n" +                    \
    "===================\n" +                    \
    "Author:    <AUTHOR>\n" +                    \
    "Date:      <DATE>\n" +                      \
    "Git:       <GIT-HASH>\n\n" +                \
    "----------------------------\n" +           \
    "This file is auto-generated!\n" +           \
    "----------------------------\n\n" +         \
    "System:\n" +                                \
    "-------\n" +                                \
    "       OS: <OS>\n" +                        \
    "     Arch: <ARCH>\n" +                      \
    "    Cores: <CORES>\n" +                     \
    "     Node: <NODE>\n" +                      \
    "     User: <USER>\n" +                      \
    "   Python: <PYTHON>\n"                      \
    "    NumPy: <NUMPY>\n"                       \
    "   Pandas: <PANDAS>\n\n"                      \
    "Console:\n" +                               \
    "--------\n" +                               \
    "<COMMAND>\n\n" +                            \
    "Notes:\n" +                                 \
    "------\n" +                                 \
    "<NOTES>"


def infer_script_name(stack_depth: Optional[int]=2) -> str:
    stack_depth = -1 if stack_depth is None else stack_depth
    caller = inspect.getframeinfo(inspect.stack()[stack_depth][0])
    app_id = Path(caller.filename).stem
    return app_id


def get_git_repo(path_to_repo: OptionalPath=None) -> Optional[git.Repo]:
    """
    Return a git.Repo object. If path_to_repo is None,
    check the environment variable MHEALTH_REPO.
    """
    if not path_to_repo:
        path_to_repo = os.getenv("MHEALTH_ROOT")
    if not path_to_repo:
        candidate = Path(__file__).parent.parent.parent.parent
        if (candidate / ".git").exists:
            path_to_repo = candidate
    if path_to_repo is None:
        print("WARN: The environment variable MHEALTH_ROOT is not set.")
        print("WARN: Some features may not be working properly.")
        return None
    path_to_repo = Path(path_to_repo)
    if path_to_repo.is_file():
        path_to_repo = path_to_repo.parent
    if not path_to_repo.is_dir():
        print("WARN: This is not a valid path: %s" % path_to_repo)
        return None
    try:
        repo = git.Repo(path_to_repo)
    except git.exc.InvalidGitRepositoryError:
        print("WARN: This is not a valid repository: %s" % path_to_repo)
        repo = None
    return repo


def get_git_hash(path_or_repo: Union[OptionalPath, git.Repo]=None,
                 with_repo_name: bool=False) -> str:
    repo = None
    if not isinstance(path_or_repo, git.Repo):
        repo = get_git_repo(path_or_repo)
    else:
        repo = path_or_repo
    if not repo:
        return "<N/A>"
    git_hash = repo.head.object.hexsha[0:8]
    if with_repo_name:
        repo_name = Path(repo.git_dir).parent.stem
        git_hash += " (%s.git)" % repo_name
    return git_hash


def get_module_version(module: str) -> str:
    try:
        return import_meta.version(module)
    except import_meta.PackageNotFoundError:
        return "N/A"


def dump_context(out_dir: PathLike) -> None:
    info = ContextInfo()
    info.dump(out_dir=out_dir)


class ContextInfo:
    """
    Warning: using ContextInfo across multiple threads or processes is unsafe!
    """
    overwrite: bool = False
    sub_dir: str = "_context"  # where the dump goes

    def __init__(self, path_to_repo: str=None):
        self.extra_context: ExtraContextDict = {}
        self.system: Dict[str, Any] = {}
        self.info = INFO_TEMPLATE

        self.repo = get_git_repo(path_to_repo)
        self.system["os"] = self.get_operating_system()
        self.system["arch"] = platform.architecture()[0]
        self.system["cores"] = multiprocessing.cpu_count()
        self.system["node"] = platform.node()
        self.system["user"] = psutil.Process().username()
        self.system["python"] = sys.version.split("\n")[0]
        self.system["numpy"] = get_module_version("numpy")
        self.system["pandas"] = get_module_version("pandas")
        self.time = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")

    @staticmethod
    def get_operating_system(short: bool=False) -> str:
        """
        https://stackoverflow.com/a/10091465/3388962
        https://github.com/easybuilders/easybuild/wiki/OS_flavor_name_version
        """
        os_type = platform.system()
        os_type = os_type.replace("Darwin", "MacOS")
        os_name = platform.platform()
        return os_type if short else "%s (%s)" % (os_type, os_name)

    def _fill_info_tag(self,
                       tag: str,
                       info: Any,
                       indent: Optional[int]=None) -> None:
        info = str(info)
        if indent is not None:
            ind = " "*indent
            info = info.replace("\n", "\n"+ind)
            info = ind + info
        self.info = self.info.replace(tag, info)

    def _fill_template(self, notes: Optional[str]=None) -> None:
        self.info = INFO_TEMPLATE
        if not self.repo:
            print("ERROR: Cannot extract repo info from invalid repo.")
        git_hash = get_git_hash(path_or_repo=self.repo,
                                with_repo_name=True)
        author = " ".join(map(str.capitalize, self.system["user"].split()))
        self._fill_info_tag("<AUTHOR>", author)
        self._fill_info_tag("<DATE>", self.time)
        self._fill_info_tag("<GIT-HASH>", git_hash)
        self._fill_info_tag("<COMMAND>", " ".join(sys.argv))
        self._fill_info_tag("<OS>", self.system["os"])
        self._fill_info_tag("<ARCH>", self.system["arch"])
        self._fill_info_tag("<CORES>", self.system["cores"])
        self._fill_info_tag("<NODE>", self.system["node"])
        self._fill_info_tag("<USER>", self.system["user"])
        self._fill_info_tag("<PYTHON>", self.system["python"])
        self._fill_info_tag("<NUMPY>", self.system["numpy"])
        self._fill_info_tag("<PANDAS>", self.system["pandas"])
        if notes is not None:
            self._fill_info_tag("<NOTES>", notes)


    def _ensure_filename(self, path: PathLike) -> Path:
        return ensure_counted_path(path=path, fmt="_%03d",
                                   enabled=not self.overwrite)

    def _dump_extra_context(self, out_dir: PathLike) -> None:
        out_dir = Path(out_dir)
        for filename, dump_fct in self.extra_context.items():
            filepath = self._ensure_filename(out_dir/filename)
            try:
                dump_fct(filepath)
            except Exception as e:
                print("WARN: Failed to dump item '%s'" % filename)
                print("WARN: The error message: %s" % e)

    @staticmethod
    def _ensure_app_id(app_id: str=None) -> str:
        # Construct app_id
        app_id = app_id if app_id else infer_script_name(stack_depth=None)
        app_id = str(app_id) if app_id else ""
        app_id = app_id.lower()
        app_id = app_id.replace(" ", "_")
        return app_id

    def add_context(self, filename: str,
                    dump_fct: DumpFunction) -> None:
        # Add additional material to dump. The mechanism is very generic,
        # but requires the caller to know how to dump the new item.
        #
        # Arguments:
        #   filename:   target filename (not path) of the the extra material
        #               the filepath will be constructed when calling
        #               ContextInfo.dump(out_dir)
        #   dump_fct:   a unary function: f(filepath)
        if filename in self.extra_context:
            print("WARN: Overriding existing context: %s" % filename)
        self.extra_context[filename] = dump_fct

    def print(self) -> None:
        self._fill_template(notes=None)
        print(self.info)

    @staticmethod
    def context_dir(out_dir: PathLike,
                    app_id: Optional[str]=None) -> Path:
        out_dir = Path(out_dir)
        app_id = ContextInfo._ensure_app_id(app_id)
        out_dir = out_dir / ContextInfo.sub_dir / app_id
        return out_dir

    def dump(self, out_dir: PathLike,
             notes: Optional[str]=None,
             app_id: Optional[str]=None) -> None:
        out_dir = self.context_dir(out_dir=out_dir, app_id=app_id)
        self._fill_template(notes=notes)
        if not ensure_dir(out_dir):
            print("ERROR: Failed to create output directory: %s" % out_dir)
            return
        try:
            info_file = self._ensure_filename(out_dir / "info.txt")
            diff_file = self._ensure_filename(out_dir / "local.diff")
            if self.repo:
                with open(diff_file,"wb") as fidb:
                    t = self.repo.head.commit.tree
                    fidb.write(self.repo.git.diff(t).encode("utf-8").strip())
            with open(info_file, "w") as fid:
                fid.write(self.info)
        except Exception as ex:
            print("ERROR: Failed to dump context info.")
            print(ex)
            return
        self._dump_extra_context(out_dir)
