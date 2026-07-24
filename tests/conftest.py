"""Test path setup for the editable-style External Project layout."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "source" / "anymal_locomotion"
ROS2_PACKAGE_ROOT = PROJECT_ROOT / "deployment" / "ros2_ws" / "src" / "anymal_locomotion_ros2"

sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(ROS2_PACKAGE_ROOT))
