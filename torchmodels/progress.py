from datetime import datetime, timedelta
from sys import stdout
from timeit import default_timer
from typing import Tuple, List
import webbrowser

from torch.utils.tensorboard import SummaryWriter
from tensorboard.program import TensorBoard

from metrics import MetricsContainer

BLUE = 34
PURPLE = 35


def wrap_color(msg: str, color_code: int):
    """
    Wrap a message with a ANSI color code.
    :param msg: message
    :param color_code: ANSI color code
    :return: wrapped message
    """
    return f"\u001B[{color_code}m{msg}\u001B[0m"


def wrap_info(msg: str, ts: bool = True) -> str:
    """
    Wrap a message in blue color and optionally prepend a timestamp.
    :param msg: message
    :param ts: whether to include a timestamp
    :return: wrapped message
    """
    ts_fmt = f"[{datetime.now()}] " if ts else ""
    return wrap_color(ts_fmt + msg, BLUE)


def launch_tensorboard(log_path: str, open_browser: bool = True) -> str:
    """
    Launch tensorboard at given log path.
    :param log_path: log path
    :param open_browser: whether to open tensorboard url in default browser
    :return: tensorboard url
    """
    tb = TensorBoard()
    tb.configure((None, "--logdir", log_path))
    url = tb.launch()
    if open_browser:
        webbrowser.open(url)
    return url


class StopWatch:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.is_running = False

    def start(self):
        should_start = not self.is_running
        self.is_running = True
        if should_start:
            self.start_time = default_timer()

    def stop(self):
        if self.is_running:
            self.end_time = default_timer()
        self.is_running = False

    def get_elapsed(self) -> float:
        return default_timer() - self.start_time

    def get_stats(self, steps: int) -> Tuple[int, float]:
        if self.is_running:
            elapsed = self.get_elapsed()
        else:
            elapsed = self.total_duration
        elapsed = int(round(elapsed))
        seconds_per_step = elapsed / steps
        return elapsed, seconds_per_step

    @property
    def total_duration(self) -> float:
        return self.end_time - self.start_time


class ProgressLogger:
    def __init__(self, log_path: str, total_epochs: int, modes: List[Tuple[str, int]], tensorboard: bool = True,
                 file=stdout):
        self._log_path = log_path
        self._launch_tensorboard = tensorboard
        self._total_epochs = total_epochs
        self._total_epochs_length = len(str(total_epochs))
        self._current_epoch = None
        self._epoch_fmt = None
        self._session_timer = StopWatch()
        self._modes_names = [m[0].capitalize() for m in modes]
        self._modes_total = [m[1] for m in modes]
        self._modes_steps = [0] * len(modes)
        self._modes_timer = []
        self._modes_fmt = [""] * len(modes)
        for _ in modes:
            self._modes_timer.append(StopWatch())
        self._file = file
        self._summary = None

    def _get_update_format(self, msg: str = ""):
        return f"\r{self._epoch_fmt} - " + msg + " - ".join(filter(None, self._modes_fmt))

    def get_summary(self) -> SummaryWriter:
        return self._summary

    def begin_session(self, session_type: str):
        """
        Print session start info and start session timer.
        """
        if self._session_timer.is_running:
            return

        print(wrap_info(f"### START SESSION '{session_type.upper()}' ###", False), file=self._file)
        print(wrap_info(f"{self._total_epochs} epochs remaining."), file=self._file)
        self._summary = SummaryWriter(self._log_path)

        if self._launch_tensorboard:
            url = launch_tensorboard(self._log_path)
            print(wrap_info(f"TensorBoard launched: {url}."), file=self._file)

        self._session_timer.start()

    def end_session(self):
        """
        Print the running time of the session.
        """
        if not self._session_timer.is_running:
            return

        self._session_timer.stop()
        self._summary.close()
        print(wrap_info(f"Session finished in {timedelta(seconds=self._session_timer.total_duration)}."),
              file=self._file)

    def begin_epoch(self, epoch: int):
        self._current_epoch = epoch
        self._epoch_fmt = f"Epoch {epoch + 1:{self._total_epochs_length}}/{self._total_epochs}"
        self._modes_steps = [0] * len(self._modes_total)
        self._modes_fmt = [""] * len(self._modes_total)
        print(self._epoch_fmt, end="", file=self._file)

    def end_epoch(self, metrics: MetricsContainer):
        assert self._current_epoch >= 0
        for timer in self._modes_timer:
            timer.stop()
        total_elapsed_time = timedelta(seconds=sum(timer.total_duration for timer in self._modes_timer))

        for mode in range(len(self._modes_total)):
            if self._modes_steps[mode] < self._modes_total[mode]:
                self._modes_fmt[mode] = ""
                continue
            elapsed, seconds_per_step = self._modes_timer[mode].get_stats(self._modes_total[mode])
            self._modes_fmt[mode] = f"{self._modes_names[mode]}: {elapsed}s {seconds_per_step:.3f}s/step"

        fmt = self._get_update_format(str(total_elapsed_time) + " - ")
        fmt += " - " + metrics.format_all()
        print(wrap_color(fmt, PURPLE), file=self._file)

        # Write metrics to tensorboard
        for metric in metrics.get_metrics():
            self._summary.add_scalar(metric.name, metric.value, self._current_epoch)
        for group_name, metric_list in metrics.get_related_metrics().items():
            self._summary.add_scalars("Summary/" + group_name, {
                metrics[m].name: metrics[m].value for m in metric_list
            }, self._current_epoch)

    def update_epoch_mode(self, mode: int, n: int = 1, metrics: str = None):
        self._modes_steps[mode] += n
        elapsed, seconds_per_step = self._modes_timer[mode].get_stats(self._modes_steps[mode])
        eta = int((self._modes_total[mode] - self._modes_steps[mode]) * seconds_per_step)

        self._modes_fmt[mode] = f"{self._modes_names[mode]}: {self._modes_steps[mode]}/{self._modes_total[mode]}" \
                                f" {elapsed}s (ETA {eta}s) {seconds_per_step:.3f}s/step"

        fmt = self._get_update_format()
        if metrics is not None:
            fmt += " - " + metrics
        print(fmt, end="", file=self._file)

    def begin_epoch_mode(self, mode: int):
        for i in range(len(self._modes_total)):
            timer = self._modes_timer[i]
            if timer.is_running:
                timer.stop()
                elapsed, seconds_per_step = self._modes_timer[i].get_stats(self._modes_total[i])
                self._modes_fmt[i] = f"{self._modes_names[i]}: {self._modes_steps[i]}/" \
                                     f"{self._modes_total[i]} {elapsed}s {seconds_per_step:.3f}s/step"
                self._modes_fmt[i] = wrap_color(self._modes_fmt[i], PURPLE)
        self._modes_timer[mode].start()
