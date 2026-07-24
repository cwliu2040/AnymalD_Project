"""Tk keyboard window that publishes a press-and-hold ROS 2 Twist command."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Sequence

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

from anymal_locomotion_ros2.teleop_core import TeleopLimits, TeleopState


class KeyboardTeleopNode(Node):
    """ROS publisher and configurable command limits."""

    def __init__(self) -> None:
        super().__init__("anymal_keyboard_teleop")
        self.declare_parameter("command_topic", "/cmd_vel")
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("forward_speed", 0.5)
        self.declare_parameter("lateral_speed", 0.3)
        self.declare_parameter("yaw_speed", 0.5)

        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        if publish_rate_hz < 10.0:
            raise ValueError("publish_rate_hz must be at least 10 Hz")
        self.publish_period_ms = max(1, round(1000.0 / publish_rate_hz))
        self.state = TeleopState(
            limits=TeleopLimits(
                forward_speed=float(self.get_parameter("forward_speed").value),
                lateral_speed=float(self.get_parameter("lateral_speed").value),
                yaw_speed=float(self.get_parameter("yaw_speed").value),
            )
        )
        self.publisher = self.create_publisher(
            Twist,
            str(self.get_parameter("command_topic").value),
            10,
        )

    def publish_command(self) -> tuple[float, float, float]:
        vx, vy, wz = (float(value) for value in self.state.command())
        message = Twist()
        message.linear.x = vx
        message.linear.y = vy
        message.angular.z = wz
        self.publisher.publish(message)
        return vx, vy, wz


class KeyboardTeleopWindow:
    """Small focus-aware window with real key-release events."""

    _RELEASE_DELAY_MS = 40

    def __init__(self, node: KeyboardTeleopNode) -> None:
        self.node = node
        self.root = tk.Tk()
        self.root.title("ANYmal-D ROS 2 鍵盤控制")
        self.root.geometry("560x260")
        self.root.minsize(520, 240)
        self._release_jobs: dict[str, str] = {}

        instructions = (
            "請保持此視窗焦點並按住按鍵\n\n"
            "W / S：前進 / 後退       A / D：左轉 / 右轉\n"
            "Q / E：向左 / 向右側移   Space：立即停止\n"
            "+ / -：提高 / 降低速度倍率"
        )
        tk.Label(
            self.root,
            text=instructions,
            font=("Sans", 13),
            justify="left",
            padx=20,
            pady=15,
        ).pack(fill="x")
        self.status = tk.StringVar()
        tk.Label(
            self.root,
            textvariable=self.status,
            font=("Monospace", 12, "bold"),
            fg="#164a86",
            pady=10,
        ).pack(fill="x")
        self.focus_status = tk.StringVar(value="控制已啟用")
        tk.Label(self.root, textvariable=self.focus_status, font=("Sans", 11)).pack(fill="x")

        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)
        self.root.bind("<FocusIn>", self._on_focus_in)
        self.root.bind("<FocusOut>", self._on_focus_out)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(self.node.publish_period_ms, self._tick)
        self.root.focus_force()

    @staticmethod
    def _normalized_key(event: tk.Event) -> str:
        keysym = str(event.keysym).lower()
        if keysym in ("plus", "equal", "kp_add"):
            return "+"
        if keysym in ("minus", "underscore", "kp_subtract"):
            return "-"
        if keysym == "space":
            return "space"
        return keysym

    def _cancel_release(self, key: str) -> None:
        job = self._release_jobs.pop(key, None)
        if job is not None:
            self.root.after_cancel(job)

    def _on_key_press(self, event: tk.Event) -> None:
        key = self._normalized_key(event)
        self._cancel_release(key)
        if key == "+":
            self.node.state.increase_speed()
        elif key == "-":
            self.node.state.decrease_speed()
        else:
            self.node.state.press(key)

    def _on_key_release(self, event: tk.Event) -> None:
        key = self._normalized_key(event)
        if key not in self.node.state.MOTION_KEYS:
            return
        self._cancel_release(key)
        self._release_jobs[key] = self.root.after(
            self._RELEASE_DELAY_MS,
            lambda released_key=key: self._finish_release(released_key),
        )

    def _finish_release(self, key: str) -> None:
        self._release_jobs.pop(key, None)
        self.node.state.release(key)

    def _on_focus_in(self, _event: tk.Event) -> None:
        self.focus_status.set("控制已啟用")

    def _on_focus_out(self, _event: tk.Event) -> None:
        self.node.state.stop()
        self.focus_status.set("視窗失去焦點：已停止")

    def _tick(self) -> None:
        if not rclpy.ok():
            self.root.destroy()
            return
        rclpy.spin_once(self.node, timeout_sec=0.0)
        vx, vy, wz = self.node.publish_command()
        self.status.set(
            f"倍率 {self.node.state.speed_scale:.2f} | "
            f"vx={vx:+.2f} m/s  vy={vy:+.2f} m/s  wz={wz:+.2f} rad/s"
        )
        self.root.after(self.node.publish_period_ms, self._tick)

    def _on_close(self) -> None:
        self.node.state.stop()
        self.node.publish_command()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main(args: Sequence[str] | None = None) -> None:
    rclpy.init(args=args)
    node: KeyboardTeleopNode | None = None
    try:
        node = KeyboardTeleopNode()
        window = KeyboardTeleopWindow(node)
        node.get_logger().info("Publishing press-and-hold commands on /cmd_vel")
        window.run()
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
