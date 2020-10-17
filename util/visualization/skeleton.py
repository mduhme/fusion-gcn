import matplotlib.pyplot as plt
from typing import Optional
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Path3DCollection

from util.visualization.visualizer import Visualizer, set_3d_aspect_ratio_equal


class SkeletonVisualizer(Visualizer):
    def __init__(self):
        super().__init__()
        self.min_xyz = None
        self.max_xyz = None
        self.graph = None
        self.joints = None
        self._scatter: Optional[Path3DCollection] = None
        self._lines: Optional[Line3DCollection] = None

    def init(self, subplot: plt.Axes, data_sequence, **kwargs):
        self.min_xyz = np.min(data_sequence, axis=(0, 1))
        self.max_xyz = np.max(data_sequence, axis=(0, 1))
        self.graph = kwargs.get("graph", None)
        self.joints = kwargs.get("joints", None)
        if kwargs.get("hide_tick_labels", False):
            subplot.set_xticklabels([])
            subplot.set_yticklabels([])
            subplot.set_zticklabels([])

    def show(self, subplot: plt.Axes, data: np.ndarray, **kwargs):
        # data shape should be (num_joints, 3)
        # swap y and z for matplotlib
        dc = data.copy()
        dc[:, 2], dc[:, 1] = data[:, 1], data[:, 2]
        data = dc

        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        if self._scatter is None:
            self._scatter = subplot.scatter(x, y, z)
        else:
            # Update 3D points
            self._scatter.set_offsets(data[:, :2])
            if hasattr(self._scatter, "_offsets3d"):
                self._scatter._offsets3d = x, y, z
                self._scatter.stale = True
            else:
                self._scatter.set_alpha(1.)
                self._scatter.set_3d_properties(z, "z")

        if self.graph is not None:
            # plot lines between scatter plot dots
            edges = np.asarray(self.graph.edges)
            p = data[edges[:, 0]]
            q = data[edges[:, 1]]

            ls = np.hstack([p, q])
            ls = ls.reshape((-1, 2, 3))
            if self._lines is None or self._lines not in subplot.collections:
                self._lines = Line3DCollection(ls, linewidths=2, colors="dimgray")
                subplot.add_collection(self._lines)
            else:
                self._lines.set_segments(ls)

        set_3d_aspect_ratio_equal(subplot)
