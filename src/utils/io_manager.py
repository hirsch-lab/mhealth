import re
import warnings
from pathlib import Path

def extract_infos(filename, patterns=None, transformers=None):
    """
    patterns:       A dict of the form {key: pattern}. A collection of
                    regular expressions to extract information from the
                    filename: {key: re.match(pattern, filename)}
    transformers:   A dict of the form {key: trafo}. A collection of
                    transformers with signature trafo(ret), where ret is
                    the object returned by re.match(). The default
                    transformer for pattern is to return the first
                    capture group:
                        def trafo(ret): return ret.group(1)

    The info will be returned as a dict {key: info}

    Example:
        extract_initials = lambda ret: (ret.group(1)+ret.group(2)).upper()
        infos = extract_infos("walt_disney",
                              patterns={"surname": ".*_(.*)",
                                        "initials": "(.).*_(.).*"},
                              transformers={"initials": extract_initials})
        # infos = {'surname': 'disney', 'initials': 'WD'}
    """
    infos = {}
    if patterns is None:
        patterns = {}
    if transformers is None:
        transformers = {}
    def default_trafo(ret): return ret.group(1)
    for key, pattern in patterns.items():
        ret = re.match(pattern, filename)
        msg = "File pattern does not match filename: %s (%s)"
        assert ret is not None, (msg % (pattern, key))
        infos[key] = ret
        transformers.setdefault(key, default_trafo)
    msg = ("Can only transform information that was previously extracted. "
           "No pattern specified for trafo key '%s'")
    for key, trafo in transformers.items():
        assert key in patterns, msg % key
        infos[key] = trafo(infos[key])
    return infos




class IOManager:
    """
    Manager for the abstraction of IO operations.



    Concepts:
        - Manage output creation for different targets with one interface
        - Targets are typically files of a particular type (e.g., .csv, .h5)
        - Information extraction patterns, filename templates and write
          methods can be injected.
        - Information can be extracted from the filename via regular
          expressions and transformers (info_patterns, info_transformers).
          This information then can be used to construct the filenames of
          the output (target_names).
        - Additional context information can be passed through the the
          set_current() method. All additional kwargs will be appended
          to the info-dictionary {key: info}
        - The principal routines are available only after setting the
          current file. Either set IOManager.set_current() or use the
          context manager returned by IOManager.current(). A file is
          named "current file" after it has been "captured" using one
          of those methods.
        - Target keys typically are file extensions (starting with a dot).
          Target keys are used for some default ops to infer the target
          file extension. It is possible to override this behavior by
          setting target_names.

    Example:
        def process_files(files, out_dir, **kwargs):
            iom = IOManager(out_dir=out_dir,
                            save_as=".csv",
                            skip_existing=False)
            for filepath in files:
                with iom.current(filepath):
                    if iom.skip_existing():
                        continue
                    path = iom.path
                    #
                    # Processing...
                    # data = process_data(path)
                    #
                    iom.write_data(data=data)

    See io_manager_test.py for more examples.

    Backlog:
        - Introduce concepts from logging where different sources (loggers)
          and appenders can be linked together flexibly.
    """
    class _IOManagerContext:
        """
        Dummy context manager so that the current file is valid only
        within a with statement:

        iom = IOManager(...)
        with iom.current(filepath):
            print(iom.path)
            print(iom.name)
        """
        def __init__(self, manager, filepath, *args, **kwargs):
            self._manager = manager
            self._current = filepath
            self._args = args
            self._kwargs = kwargs
        def __enter__(self):
            self._manager.set_current(filepath=self._current,
                                      *self._args, **self._kwargs)
            return self._manager
        def __exit__(self, exc_type, exc_val, exc_tb):
            self._manager.reset_current()
            return

    def __init__(self, *args, **kwargs):
        """
        See IOManager.init()
        """
        self.init(*args, **kwargs)

    def init(self,
             out_dir="./output",
             targets=None,
             info_patterns=None,
             info_transformers=None,
             target_writers=None,
             target_names=None,
             skip_existing=False,
             dry_run=False):
        """
        Initialize the IOManager. It resets all states.

        Arguments:
            out_dir:        Path to output directory. This represents the
                            global setting for the out_dir. Some class
                            methods permit to override this setting.
            targets:        Global flag to enable one or multiple
                            output files/targets. If None, all targets
                            introduced through target_writers will be used.
            info_patterns:  A collection of regular expressions to extract
                            information from the name of the current file.
                            The function expects a dict {key: pattern} or
                            None. If None, no information is extracted from
                            the filename.
            info_transformers: An optional collection of transformers that
                            converts for the current file the output ret of
                            the regular expression matching into the actual info:
                                ret = re.match(pattern, filename)
                                info = trafo(ret)
                            Pass a dict {key: trafo}, where key is the same
                            key as already used to specify the regular
                            expression pattern (info_patterns), and trafo is a
                            function object that accepts one argument ret:
                                def trafo(ret: re.Match) -> str
                            The default transformer is equivalent to
                                def default_trafo(ret) -> str:
                                    return ret.group(1)
                            It returns the first capture group of the regular
                            expression.
            target_writers: A collection of function objects. It is a dict
                                {target: func}
                            where func must have the following signature:
                                def func(data, path: str, **kwargs) -> bool
                            Here, data can be anything that will be passed
                            to method write_data(data, **kwargs).
                            A writer must exist for all targets requested
                            through setting targets (see above). If targets
                            is None, all target_writers will be active.
            target_names:   An optional collection of string templates to
                            construct the name of the output files.
                            The argument target_names is expected to be a dict
                                {target: template}
                            Here, target should match with a key of
                            target_writers, and template is a string that can
                            contain named format variables. See the keys of
                            property info for a list of valid names:
                                iom.info.keys()
                            For instance the following a valid expression if
                            info contains keys "pat_id" and "side":
                                template = "{pat_id:04d}-{side}.ext"
                            By default, the target name is constructed as
                                ext = target
                                template = "{name}{ext}"
                            where ext is the name of the target. Hence, use
                            extensions as target names (e.g., ".csv", ".h5"),
                            unless the target_names are set explicitly.
                skip_existing: Triggers method skip_existing()
                dry_run:    Disable any output generation
        """
        def _ensure_dict(x, default):
            if issubclass(type(x), dict):
                return x
            elif x is None:
                return default
            else:
                msg = "Expecting a dictionary."
                assert False, msg

        self._out_dir = Path(out_dir)
        # Info related
        self._info_patterns = info_patterns
        self._info_transformers = info_transformers
        # Target related
        self._targets = [targets] if isinstance(targets, str) else targets
        self._target_writers = _ensure_dict(target_writers, default={})
        self._target_names = _ensure_dict(target_names, default={})
        self._targets = (self._targets if self._targets
                         else list(self._target_writers.keys()))
        # Options
        self._skip_existing = skip_existing
        self._dry_run = dry_run
        # Context containers
        self.reset_current()

    def current(self, filepath, **kwargs):
        """
        Create context manager that calls set_current() and reset_current()
        when entering and exiting a context.
        """
        return self._IOManagerContext(manager=self,
                                      filepath=filepath,
                                      **kwargs)

    def get_info(self, key, default="N/A"):
        return self._info.get(key, default)

    def set_info(self, **kwargs):
        self._info.update(kwargs)

    @property
    def path(self): return self.get_info(key="path")
    @property
    def name(self): return self.get_info(key="name")
    @property
    def info(self):
        msg = "No current file is set!"
        assert bool(self._info), msg
        return self._info

    def get_out_path(self, target, out_dir=None):
        """
        out_dir overrides global setting self._out_dir
        """
        out_dir = self._out_dir if out_dir is None else out_dir
        out_dir = Path(out_dir)
        msg = "No filename specification available for target '%s'."
        assert target in self._info_targets, msg % target
        return out_dir / self._info_targets[target]["name"].format(**self.info)

    def get_out_paths(self, targets=None, out_dir=None):
        """
        targets overrides global setting self._targets
        out_dir overrides global setting self._out_dir
        """
        targets = self._targets if targets is None else targets
        return {t: self.get_out_path(t, out_dir=out_dir) for t in targets}

    def set_current(self, filepath,
                    **kwargs):
        filepath = Path(filepath)
        self.reset_current()
        # Info extraction
        self._info = {}
        infos = extract_infos(filename=filepath.name,
                              patterns=self._info_patterns,
                              transformers=self._info_transformers)
        self._info["path"] = filepath
        #self._info["name"] = filepath.name
        self._info["name"] = filepath.stem
        self._info.update(kwargs)
        self._info.update(infos)

        # Output configuration
        # Always a dict: {ext: info}
        self._info_targets = {}
        for target in self._targets:
            # Default name: name of current file + ext
            default_name = "{name}"+target
            self._info_targets[target] = {
                "name": self._target_names.get(target, default_name),
                "writer": self._target_writers.get(target, None)
            }

    def reset_current(self):
        self._info = {}
        self._info_targets = {}

    def skip_existing(self, targets=None, out_dir=None):
        if not self._skip_existing:
            return False
        if not self._targets:
            return False
        out_paths = self.get_out_paths(targets=targets, out_dir=out_dir)
        exist = [path.is_file() for path in out_paths.values()]
        return all(exist)

    def write_data(self, data, targets=None, out_dir=None, **kwargs):
        out_paths = self.get_out_paths(targets=targets, out_dir=out_dir)
        msg = "Cannot process output for requested target: %s. Skipping..."
        status = True
        rets = {}
        for target, out_path in out_paths.items():
            if target not in self._info_targets:
                warnings.warn(msg % target, RuntimeWarning)
                continue
            info = self._info_targets[target]
            func = info["writer"]
            if self._dry_run:
                print("Writing: target=%s, path=%s" % (target, out_path))
                status &= True
                rets[target] = out_path
            if func:
                # if not out_path.parent.is_dir():
                #     out_path.parent.mkdir(parents=True)
                if not self._out_dir.is_dir():
                    self._out_dir.mkdir(parents=True)
                ret = func(data, out_path=out_path, **kwargs)
                rets[target] = ret
                status &= bool(ret)
            else:
                warnings.warn(msg % target, RuntimeWarning)
                raise RuntimeError(msg % target)
                continue
        return rets
