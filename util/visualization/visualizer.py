import abc

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button


class Visualizer:
    def __init__(self):
        pass

    def init(self, subplot: plt.Axes, data_sequence, **kwargs):
        pass

    @abc.abstractmethod
    def show(self, subplot: plt.Axes, data: np.ndarray, **kwargs):
        pass


class SynchronizedVisualizer:
    def __init__(self, sync_modality: str):
        self.sync_modality = sync_modality

    def update(self):
        pass


class Controller:
    def __init__(self, next_cb, save_previous_states=False):
        self._next_cb = next_cb
        self._save_previous_states = save_previous_states
        self._index = 0
        self._fig = None
        self._btn_next = None

    def _redraw(self):
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def _next(self, event):
        self._next_cb(event)
        self._index += 1
        self._redraw()

    def _prev(self, event):
        self._index -= 1
        self._redraw()

    def create_at(self, fig: plt.Figure):
        fig.subplots_adjust(bottom=0.2)
        next_btn_pos = plt.axes([0.81, 0.05, 0.1, 0.075])
        self._fig = fig
        self._btn_next = Button(next_btn_pos, "next")
        self._btn_next.on_clicked(self._next)


# from https://stackoverflow.com/a/19248731
def set_3d_aspect_ratio_equal(ax):
    extents = np.array([getattr(ax, f"get_{dim}lim")() for dim in "xyz"])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, "xyz"):
        getattr(ax, f"set_{dim}lim")(ctr - r, ctr + r)
