# src/core/__init__.py

from .camera import Camera
from .utils import (
    find_color_in_img_mean,
    find_color_coordinates_hsv,
    live_plot_bnc_result,
    get_rotation_matrix_y
)

__version__ = '1.0.0'

"""
Core components for the BNC Tracker system.

This package contains fundamental classes and utilities for camera handling,
color detection, and coordinate transformations used throughout the system.
"""