import os
from datetime import datetime, timedelta
from sys import stdout
from timeit import default_timer
from typing import Tuple, List

import torch
from tensorboard.program import TensorBoard
from torch.utils.tensorboard import SummaryWriter

from metrics import MetricsContainer


class AnsiColors:
    BLACK = 30
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    MAGENTA = 35
    CYAN = 36
    WHITE = 37
    RESET = 39


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
    return wrap_color(ts_fmt + msg, AnsiColors.BLUE)


def launch_tensorboard(log_path: str) -> str:
    """
    Launch tensorboard at given log path.
    :param log_path: log path
    :return: tensorboard url
    """
    tb = TensorBoard()
    tb.configure((None, "--logdir", log_path))
    url = tb.launch()
    return url


class StopWatch:
    """
    Measure duration of any operation.
    """

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
        seconds_per_step = elapsed / steps
        return int(round(elapsed)), seconds_per_step

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
        print(wrap_color(fmt, AnsiColors.MAGENTA), file=self._file)

        # Write metrics to tensorboard
        metrics.to_summary(self._summary, self._current_epoch + 1)

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
                self._modes_fmt[i] = wrap_color(self._modes_fmt[i], AnsiColors.MAGENTA)
        self._modes_timer[mode].start()


class CheckpointManager:
    def __init__(self, checkpoint_path: str, state_dict_objects: dict, num_checkpoints: int):
        self.checkpoint_path = checkpoint_path
        self.state_dict_objects = state_dict_objects
        self.num_checkpoints = num_checkpoints

    def save_checkpoint(self, epoch: int, val_acc: float, **additional_objects: dict):
        os.makedirs(self.checkpoint_path, exist_ok=True)

        state_dicts = {k: v.state_dict() for k, v in self.state_dict_objects.items()}
        state_dicts = {**state_dicts, **additional_objects, "epoch": epoch}

        existing_checkpoints = [file.name for file in os.scandir(self.checkpoint_path) if
                                file.is_file() and file.name.startswith("checkpoint_") and file.name.endswith(".pt")]

        if len(existing_checkpoints) >= self.num_checkpoints:
            cp_acc = [float(file[file.rindex("_") + 1:file.rindex(".pt")]) for file in existing_checkpoints]
            cp_acc_sorted = sorted(cp_acc, reverse=True)
            checkpoint_to_remove = existing_checkpoints[cp_acc.index(cp_acc_sorted[-1])]
            os.remove(os.path.join(self.checkpoint_path, checkpoint_to_remove))

        out_file = os.path.join(self.checkpoint_path, f"checkpoint_{epoch}_{val_acc:.4}.pt")
        torch.save(state_dicts, out_file)

    def save_weights(self, model: torch.nn.Module, file_name_prefix: str = ""):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        if file_name_prefix:
            file_name_prefix += "_"
        out_file = os.path.join(self.checkpoint_path, f"{file_name_prefix}weights.pt")
        torch.save(model.state_dict(), out_file)

    def load_weights(self, model: torch.nn.Module, file_name_prefix: str = ""):
        if file_name_prefix:
            file_name_prefix += "_"
        in_file = os.path.join(self.checkpoint_path, f"{file_name_prefix}weights.pt")
        cp = torch.load(in_file)
        model.load_state_dict(cp)

    def load_best(self, apply: bool = True) -> dict:
        return self._load(CheckpointManager._get_acc, apply)

    def load_latest(self, apply: bool = True) -> dict:
        return self._load(CheckpointManager._get_epoch, apply)

    def _apply_checkpoint(self, cp: dict):
        for object_name, obj in self.state_dict_objects.items():
            if object_name in cp:
                obj.load_state_dict(cp[object_name])

    def _load(self, score_fn, apply: bool) -> dict:
        files = (f.name for f in os.scandir(self.checkpoint_path) if f.is_file() and f.name.endswith(".pt"))
        best_score = 0
        best_score_file = None
        for file in files:
            score = score_fn(file)
            if score >= best_score:
                best_score = score
                best_score_file = file

        if best_score_file is None:
            raise ValueError(f"No checkpoint found in '{self.checkpoint_path}'.")
        cp = torch.load(os.path.join(self.checkpoint_path, best_score_file))

        if apply:
            self._apply_checkpoint(cp)
        return cp

    @staticmethod
    def _get_epoch(file_name: str) -> int:
        return int(file_name[file_name.index("_") + 1:file_name.rindex("_")])

    @staticmethod
    def _get_acc(file_name: str) -> float:
        return float(file_name[file_name.rindex("_") + 1:-3])
