"""Utility module.

We separate utility functions that require ROS into a separate module to avoid ROS as a
dependency for sim-only scripts.
"""

from lsy_drone_racing.utils.trajectory_plot import save_top_view_trajectory
from lsy_drone_racing.utils.utils import load_config, load_controller

__all__ = ["load_config", "load_controller", "save_top_view_trajectory"]
