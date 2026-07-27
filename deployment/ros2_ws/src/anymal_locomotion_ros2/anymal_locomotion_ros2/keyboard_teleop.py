"""Backward-compatible entry point for the official ROS 2 keyboard teleop."""

from __future__ import annotations

from collections.abc import Sequence

from teleop_twist_keyboard import main as teleop_main


def main(args: Sequence[str] | None = None) -> None:
    """Run the official terminal teleop used by the complete bringup launch."""
    if args is not None:
        raise ValueError("teleop_twist_keyboard reads ROS arguments from sys.argv")
    teleop_main()


if __name__ == "__main__":
    main()
