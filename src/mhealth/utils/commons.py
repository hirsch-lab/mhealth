import progressbar as pg
from typing import Optional

def print_title(title: str,
                width: Optional[int]=60) -> None:
    if width is None:
        width = len(title)+2
    print("")
    print("#"*width)
    print("# "+title)
    print("#"*width)
    print("")


def print_subtitle(title: str,
                   width: Optional[int]=60) -> None:
    if width is None:
        width = len(title)+2
    print("")
    print(title)
    print("#"*width)
    print("")


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
