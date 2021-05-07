import logging
import warnings
import progressbar as pg
from functools import partial
from typing import Optional, Callable, Any

PrintFunctor = Callable[[Any], Any]

DEFAULT_FMT = "%(levelname)s: %(message)s"

def setup_logging(verbosity: int=1,
                  fmt: str=DEFAULT_FMT,
                  logger: Optional[logging.Logger]=None) -> None:
    def _set_level_strings():
        logging.TRACE = logging.DEBUG - 1
        logging.DRYRUN = 1000    # Custom log level.
        logging.STATUS = 1001    # Custom log level.
        logging.addLevelName(logging.TRACE,   "TRACE")
        logging.addLevelName(logging.DEBUG,   "DEBUG")
        logging.addLevelName(logging.INFO,    "INFO")
        logging.addLevelName(logging.WARNING, "WARN")
        logging.addLevelName(logging.ERROR,   "ERROR")
        logging.addLevelName(logging.CRITICAL,"CRIT")
        logging.addLevelName(logging.DRYRUN,  "DRY")
        logging.addLevelName(logging.STATUS,  "STATUS")

        logging.trace = partial(logging.log, logging.TRACE)
        logging.dryrun = partial(logging.log, logging.DRYRUN)
        logging.status = partial(logging.log, logging.STATUS)

    _set_level_strings()

    level: Optional[int] = logging.WARN
    if verbosity >= 3:
        level = logging.TRACE  # type: ignore
    elif verbosity == 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    elif verbosity == 0:
        level = logging.WARN
    elif verbosity < 0:
        level = None

    if level is not None:
        logger = logger if logger else logging.getLogger()
        logger.setLevel(level)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt=fmt))
        logger.addHandler(handler)


def print_title(title: str,
                width: Optional[int]=60,
                printer: PrintFunctor=print) -> None:
    if width is None:
        width = len(title)+2
    printer("")
    printer("#"*width)
    printer("# "+title)
    printer("#"*width)
    printer("")


def print_subtitle(title: str,
                   width: Optional[int]=60,
                   printer: PrintFunctor=print) -> None:
    if width is None:
        width = len(title)+2
    printer("")
    printer(title)
    printer("#"*width)
    printer("")


def create_progress_bar(size: Optional[int]=None,
                        label: str="Processing...",
                        threaded: bool=False,
                        enabled: bool=True,
                        width: int=100,
                        **kwargs) -> pg.ProgressBar:
    widgets = []
    if label:
        widgets.append(pg.FormatLabel("%-5s:" % label))
        widgets.append(" ")
    if size is not None and size>0:
        digits = 3
        fmt_counter = f"%(value){digits}d/{size:{digits}d}"
        widgets.append(pg.Bar())
        widgets.append(" ")
        widgets.append(pg.Counter(fmt_counter))
        widgets.append(" (")
        widgets.append(pg.Percentage())
        widgets.append(")")
    else:
        widgets.append(pg.BouncingBar())
    ProgressBarType: pg.ProgressBar = pg.ProgressBar if enabled else pg.NullBar
    if threaded and enabled:
        from threading import Timer
        class RepeatTimer(Timer):
            def run(self):
                while not self.finished.wait(self.interval):
                    self.function(*self.args, **self.kwargs)
        class ThreadedProgressBar(ProgressBarType):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.timer = RepeatTimer(interval=0.05,
                                         function=self.update)
                self.timer.setDaemon(True)
            def run(self):
                while not self.finished.wait(self.interval):
                    self.function(*self.args, **self.kwargs)
            def start(self, *args, **kwargs):
                ret = super().start(*args, **kwargs)
                self.timer.start()
                return ret
            def finish(self, *args, **kwargs):
                self.timer.cancel()
                return super().finish(*args, **kwargs)
        ProgressBarType = ThreadedProgressBar

    progress = ProgressBarType(max_value=size,
                               widgets=widgets,
                               redirect_stdout=True,
                               poll_interval=0.02,
                               term_width=width,
                               **kwargs)
    return progress


def catch_warnings(ws, warning_to_catch=None,
                   message_to_catch=None, enabled=True):
    """
    Print warnings as usual, except the ones in warning_to_catch.

    To use within a warnings.catch_warnings() context:
        with warnings.catch_warnings(record=True) as ws:
            operations(...)
            # ...
            catch_warnings(ws,
                           warning_to_catch=pd.errors.DtypeWarning,
                           message_to_catch="have mixed types",
                           enabled=(mode=="raw"))
    """
    caught_warnings = 0
    if warning_to_catch is None:
        warning_to_catch = []
    if message_to_catch is None:
        message_to_catch = []
    elif isinstance(message_to_catch, str):
        message_to_catch = [message_to_catch]
    for w in ws:
        if (not issubclass(w.category, warning_to_catch) or
            not any(m in str(w.message) for m in message_to_catch) or
            not enabled):
            warnings.warn_explicit(message=w.message,
                                   category=w.category,
                                   filename=w.filename,
                                   lineno=w.lineno)
        else:
            caught_warnings += 1
    return bool(caught_warnings)
