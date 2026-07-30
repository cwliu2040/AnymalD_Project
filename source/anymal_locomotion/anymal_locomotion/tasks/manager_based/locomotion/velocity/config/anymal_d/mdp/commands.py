"""Project-owned velocity command distributions for robust locomotion."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.commands import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from isaaclab.utils import configclass


class EdgeBiasedVelocityCommand(UniformVelocityCommand):
    """Mix the full command envelope with deployment-critical edge cases."""

    cfg: EdgeBiasedVelocityCommandCfg

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        if not 0.0 <= self.cfg.edge_probability <= 1.0:
            raise ValueError("edge_probability must be in [0, 1]")

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.numel() == 0 or self.cfg.edge_probability == 0.0:
            return
        edge_mask = (
            torch.rand(ids.numel(), device=self.device)
            < self.cfg.edge_probability
        )
        edge_ids = ids[edge_mask]
        if edge_ids.numel() == 0:
            return
        modes = torch.randint(
            low=0,
            high=7,
            size=(edge_ids.numel(),),
            device=self.device,
        )
        self.is_standing_env[edge_ids] = False
        for mode in range(7):
            selected = edge_ids[modes == mode]
            if selected.numel() == 0:
                continue
            random_values = torch.empty(
                (selected.numel(), 3),
                device=self.device,
            )
            if mode == 0:
                ranges = ((2.5, 3.0), (-0.15, 0.15), (-0.15, 0.15))
            elif mode == 1:
                ranges = ((-2.0, -1.5), (-0.15, 0.15), (-0.15, 0.15))
            elif mode == 2:
                ranges = ((-0.2, 0.2), (1.2, 1.5), (-0.15, 0.15))
            elif mode == 3:
                ranges = ((-0.2, 0.2), (-1.5, -1.2), (-0.15, 0.15))
            elif mode == 4:
                ranges = ((-0.15, 0.15), (-0.15, 0.15), (1.5, 2.0))
            elif mode == 5:
                ranges = ((-0.15, 0.15), (-0.15, 0.15), (-2.0, -1.5))
            else:
                ranges = ((0.5, 3.0), (-0.2, 0.2), (-1.0, 1.0))
            for component, limits in enumerate(ranges):
                random_values[:, component].uniform_(*limits)
            self.vel_command_b[selected] = random_values


@configclass
class EdgeBiasedVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the full-envelope plus edge-case mixture."""

    class_type: type = EdgeBiasedVelocityCommand
    edge_probability: float = 0.6


class RecoveryV05VelocityCommand(UniformVelocityCommand):
    """Sample sustained high-speed turns and explicit transitions to zero."""

    cfg: RecoveryV05VelocityCommandCfg

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._warehouse_mode = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._warehouse_phase = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self._refinery_replay_mode = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._refinery_replay_phase = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self._turning_regression_mode = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._turning_regression_phase = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self._turning_regression_profile = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        for name, probability in (
            ("high_combined_probability", cfg.high_combined_probability),
            ("high_combined_stop_probability", cfg.high_combined_stop_probability),
            (
                "high_combined_straight_probability",
                cfg.high_combined_straight_probability,
            ),
            (
                "warehouse_sequence_probability",
                cfg.warehouse_sequence_probability,
            ),
            (
                "refinery_replay_probability",
                cfg.refinery_replay_probability,
            ),
            (
                "turning_regression_probability",
                cfg.turning_regression_probability,
            ),
            (
                "low_yaw_profile_probability",
                cfg.low_yaw_profile_probability,
            ),
            (
                "low_curve_profile_probability",
                cfg.low_curve_profile_probability,
            ),
            (
                "high_curve_profile_probability",
                cfg.high_curve_profile_probability,
            ),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if (
            cfg.high_combined_stop_probability
            + cfg.high_combined_straight_probability
            > 1.0
        ):
                raise ValueError(
                    "high-combined transition probabilities must sum to <= 1"
                )
        if (
            cfg.warehouse_sequence_probability
            + cfg.refinery_replay_probability
            + cfg.turning_regression_probability
            > 1.0
        ):
            raise ValueError(
                "dedicated replay probabilities must sum to <= 1"
            )
        if (
            cfg.low_yaw_profile_probability
            + cfg.low_curve_profile_probability
            + cfg.high_curve_profile_probability
            > 1.0
        ):
            raise ValueError(
                "targeted turning profile probabilities must sum to <= 1"
            )

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.numel() == 0:
            return
        initial = self.command_counter[ids] == 0
        initial_ids = ids[initial]
        if initial_ids.numel() > 0:
            replay_draw = torch.rand(
                initial_ids.numel(),
                device=self.device,
            )
            self._warehouse_mode[initial_ids] = (
                replay_draw < self.cfg.warehouse_sequence_probability
            )
            refinery_lower = self.cfg.warehouse_sequence_probability
            refinery_upper = (
                refinery_lower + self.cfg.refinery_replay_probability
            )
            self._refinery_replay_mode[initial_ids] = (
                replay_draw >= refinery_lower
            ) & (replay_draw < refinery_upper)
            self._turning_regression_mode[initial_ids] = (
                replay_draw >= refinery_upper
            ) & (
                replay_draw
                < (
                    refinery_upper
                    + self.cfg.turning_regression_probability
                )
            )
            self._warehouse_phase[initial_ids] = 0
            self._refinery_replay_phase[initial_ids] = 0
            self._turning_regression_phase[initial_ids] = 0
            self._turning_regression_profile[initial_ids] = torch.randint(
                0,
                12,
                (initial_ids.numel(),),
                device=self.device,
            )
            profile_draw = torch.rand(
                initial_ids.numel(),
                device=self.device,
            )
            low_yaw = profile_draw < self.cfg.low_yaw_profile_probability
            self._turning_regression_profile[initial_ids[low_yaw]] = (
                3
                * torch.randint(
                    0,
                    2,
                    (int(low_yaw.sum().item()),),
                    device=self.device,
                )
            )
            low_curve = (
                profile_draw >= self.cfg.low_yaw_profile_probability
            ) & (
                profile_draw
                < (
                    self.cfg.low_yaw_profile_probability
                    + self.cfg.low_curve_profile_probability
                )
            )
            self._turning_regression_profile[initial_ids[low_curve]] = (
                6
                + torch.randint(
                    0,
                    2,
                    (int(low_curve.sum().item()),),
                    device=self.device,
                )
            )
            high_curve = (
                profile_draw
                >= (
                    self.cfg.low_yaw_profile_probability
                    + self.cfg.low_curve_profile_probability
                )
            ) & (
                profile_draw
                < (
                    self.cfg.low_yaw_profile_probability
                    + self.cfg.low_curve_profile_probability
                    + self.cfg.high_curve_profile_probability
                )
            )
            self._turning_regression_profile[initial_ids[high_curve]] = (
                10
                + torch.randint(
                    0,
                    2,
                    (int(high_curve.sum().item()),),
                    device=self.device,
                )
            )

        warehouse_ids = ids[self._warehouse_mode[ids]]
        continuing_warehouse_ids = warehouse_ids[
            self.command_counter[warehouse_ids] > 0
        ]
        if continuing_warehouse_ids.numel() > 0:
            self._warehouse_phase[continuing_warehouse_ids] = (
                self._warehouse_phase[continuing_warehouse_ids] + 1
            ) % 9

        # Replay the deployment failure history in a dedicated subset:
        # stand -> left burst -> long straight -> short right -> straight ->
        # short right -> straight -> sustained left -> stand/recovery.
        warehouse_durations_s = (
            5.00,
            1.44,
            9.56,
            0.18,
            5.94,
            0.18,
            3.76,
            2.96,
            10.00,
        )
        warehouse_yaw_commands = (
            0.0,
            2.0,
            0.0,
            -2.0,
            0.0,
            -2.0,
            0.0,
            2.0,
            0.0,
        )
        for phase, (duration_s, yaw_command) in enumerate(
            zip(warehouse_durations_s, warehouse_yaw_commands)
        ):
            selected = warehouse_ids[
                self._warehouse_phase[warehouse_ids] == phase
            ]
            if selected.numel() == 0:
                continue
            self.vel_command_b[selected] = 0.0
            self.time_left[selected] = duration_s
            standing = phase in (0, 8)
            self.is_standing_env[selected] = standing
            if not standing:
                self.vel_command_b[selected, 0] = 2.3
                self.vel_command_b[selected, 2] = yaw_command

        refinery_ids = ids[self._refinery_replay_mode[ids]]
        continuing_refinery_ids = refinery_ids[
            self.command_counter[refinery_ids] > 0
        ]
        # Exact effective-command history from the formal refinery_fix_01 run.
        # The zero phases at indices 15 and 21 are watchdog-effective gaps in
        # that trace; the final three phases are the reproduced failure tail.
        refinery_sequence = (
            (1.46, (0.0, 0.0, 0.0)),
            (0.60, (0.5, 0.0, 0.0)),
            (0.40, (1.898749, 0.0, 0.0)),
            (0.44, (1.898749, 0.0, 0.817349)),
            (1.02, (1.898749, 0.0, 0.0)),
            (0.18, (1.898749, 0.0, 0.817349)),
            (2.80, (1.898749, 0.0, 0.0)),
            (0.12, (1.898749, 0.0, 0.817349)),
            (4.18, (1.898749, 0.0, 0.0)),
            (0.18, (1.898749, 0.0, 0.817349)),
            (0.50, (1.898749, 0.0, 0.0)),
            (0.52, (1.898749, 0.0, 0.817349)),
            (0.28, (1.898749, 0.0, 0.0)),
            (0.22, (1.898749, 0.0, 0.817349)),
            (2.62, (1.898749, 0.0, 0.0)),
            (0.50, (0.0, 0.0, 0.0)),
            (2.32, (1.898749, 0.0, 0.0)),
            (0.10, (1.898749, 0.0, -0.817349)),
            (0.70, (1.898749, 0.0, 0.0)),
            (0.16, (1.898749, 0.0, -0.817349)),
            (3.80, (1.898749, 0.0, 0.0)),
            (1.38, (0.0, 0.0, 0.0)),
            (1.38, (0.0, 0.0, 0.817349)),
            (3.72, (1.898749, 0.0, 0.0)),
            (12.52, (0.0, 0.0, 0.0)),
        )
        if continuing_refinery_ids.numel() > 0:
            self._refinery_replay_phase[continuing_refinery_ids] = (
                self._refinery_replay_phase[continuing_refinery_ids] + 1
            ) % len(refinery_sequence)
        for phase, (duration_s, command) in enumerate(refinery_sequence):
            selected = refinery_ids[
                self._refinery_replay_phase[refinery_ids] == phase
            ]
            if selected.numel() == 0:
                continue
            self.vel_command_b[selected] = torch.tensor(
                command,
                device=self.device,
            )
            self.is_standing_env[selected] = command == (0.0, 0.0, 0.0)
            self.time_left[selected] = duration_s

        turning_ids = ids[self._turning_regression_mode[ids]]
        continuing_turning_ids = turning_ids[
            self.command_counter[turning_ids] > 0
        ]
        if continuing_turning_ids.numel() > 0:
            self._turning_regression_phase[continuing_turning_ids] = (
                1 - self._turning_regression_phase[continuing_turning_ids]
            )
            active_again = continuing_turning_ids[
                self._turning_regression_phase[continuing_turning_ids] == 0
            ]
            if active_again.numel() > 0:
                self._turning_regression_profile[active_again] = torch.randint(
                    0,
                    12,
                    (active_again.numel(),),
                    device=self.device,
                )
                profile_draw = torch.rand(
                    active_again.numel(),
                    device=self.device,
                )
                low_yaw = (
                    profile_draw < self.cfg.low_yaw_profile_probability
                )
                self._turning_regression_profile[active_again[low_yaw]] = (
                    3
                    * torch.randint(
                        0,
                        2,
                        (int(low_yaw.sum().item()),),
                        device=self.device,
                    )
                )
                low_curve = (
                    profile_draw >= self.cfg.low_yaw_profile_probability
                ) & (
                    profile_draw
                    < (
                        self.cfg.low_yaw_profile_probability
                        + self.cfg.low_curve_profile_probability
                    )
                )
                self._turning_regression_profile[
                    active_again[low_curve]
                ] = (
                    6
                    + torch.randint(
                        0,
                        2,
                        (int(low_curve.sum().item()),),
                        device=self.device,
                    )
                )
                high_curve = (
                    profile_draw
                    >= (
                        self.cfg.low_yaw_profile_probability
                        + self.cfg.low_curve_profile_probability
                    )
                ) & (
                    profile_draw
                    < (
                        self.cfg.low_yaw_profile_probability
                        + self.cfg.low_curve_profile_probability
                        + self.cfg.high_curve_profile_probability
                    )
                )
                self._turning_regression_profile[
                    active_again[high_curve]
                ] = (
                    10
                    + torch.randint(
                        0,
                        2,
                        (int(high_curve.sum().item()),),
                        device=self.device,
                    )
                )

        turning_profiles = (
            (0.0, 0.0, 0.5),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 2.0),
            (0.0, 0.0, -0.5),
            (0.0, 0.0, -1.0),
            (0.0, 0.0, -2.0),
            (0.5, 0.0, 0.5),
            (0.5, 0.0, -0.5),
            (1.5, 0.0, 1.0),
            (1.5, 0.0, -1.0),
            (3.0, 0.0, 0.5),
            (3.0, 0.0, -0.5),
        )
        for profile, command in enumerate(turning_profiles):
            selected = turning_ids[
                (self._turning_regression_phase[turning_ids] == 0)
                & (self._turning_regression_profile[turning_ids] == profile)
            ]
            if selected.numel() == 0:
                continue
            self.vel_command_b[selected] = torch.tensor(
                command,
                device=self.device,
            )
            self.is_standing_env[selected] = False
            self.time_left[selected] = 10.0
        turning_stand_ids = turning_ids[
            self._turning_regression_phase[turning_ids] == 1
        ]
        if turning_stand_ids.numel() > 0:
            self.vel_command_b[turning_stand_ids] = 0.0
            self.is_standing_env[turning_stand_ids] = True
            self.time_left[turning_stand_ids] = 2.0

        ids = ids[
            ~(
                self._warehouse_mode[ids]
                | self._refinery_replay_mode[ids]
                | self._turning_regression_mode[ids]
            )
        ]
        if ids.numel() == 0:
            return
        previous = self.vel_command_b[ids].clone()
        previous_high_combined = (
            (previous[:, 0] >= 2.0)
            & (torch.abs(previous[:, 2]) >= 1.5)
        )
        previous_high_straight = (
            (previous[:, 0] >= 2.0)
            & (torch.abs(previous[:, 2]) <= 0.3)
        )
        super()._resample_command(ids)

        transition = torch.rand(ids.numel(), device=self.device)
        stop_mask = (
            previous_high_combined
            & (
                transition
                < self.cfg.high_combined_stop_probability
            )
        )
        combined_to_straight = previous_high_combined & (
            transition
            >= self.cfg.high_combined_stop_probability
        ) & (
            transition
            < (
                self.cfg.high_combined_stop_probability
                + self.cfg.high_combined_straight_probability
            )
        )
        combined_to_reverse = (
            previous_high_combined
            & ~stop_mask
            & ~combined_to_straight
        )
        straight_to_stop = previous_high_straight & (transition < 0.15)
        straight_to_burst = previous_high_straight & (
            transition >= 0.15
        ) & (transition < 0.65)
        stop_mask |= straight_to_stop
        stop_ids = ids[stop_mask]
        if stop_ids.numel() > 0:
            self.vel_command_b[stop_ids] = 0.0
            self.is_standing_env[stop_ids] = True
            self.time_left[stop_ids].uniform_(1.0, 5.0)

        straight_ids = ids[combined_to_straight]
        if straight_ids.numel() > 0:
            straight_commands = torch.empty(
                (straight_ids.numel(), 3),
                device=self.device,
            )
            straight_commands[:, 0].uniform_(2.2, 2.6)
            straight_commands[:, 1].uniform_(-0.1, 0.1)
            straight_commands[:, 2].uniform_(-0.15, 0.15)
            self.vel_command_b[straight_ids] = straight_commands
            self.is_standing_env[straight_ids] = False
            self.time_left[straight_ids].uniform_(4.0, 10.0)

        reverse_ids = ids[combined_to_reverse]
        if reverse_ids.numel() > 0:
            reverse_commands = torch.empty(
                (reverse_ids.numel(), 3),
                device=self.device,
            )
            reverse_commands[:, 0].uniform_(2.2, 2.6)
            reverse_commands[:, 1].uniform_(-0.1, 0.1)
            reverse_commands[:, 2].uniform_(1.8, 2.0)
            reverse_commands[:, 2] *= -torch.sign(
                previous[combined_to_reverse, 2]
            )
            self.vel_command_b[reverse_ids] = reverse_commands
            self.is_standing_env[reverse_ids] = False
            self.time_left[reverse_ids].uniform_(0.1, 0.5)

        burst_ids = ids[straight_to_burst]
        if burst_ids.numel() > 0:
            burst_commands = torch.empty(
                (burst_ids.numel(), 3),
                device=self.device,
            )
            burst_commands[:, 0].uniform_(2.2, 2.6)
            burst_commands[:, 1].uniform_(-0.1, 0.1)
            burst_commands[:, 2].uniform_(1.8, 2.0)
            burst_commands[:, 2] *= torch.where(
                torch.rand(burst_ids.numel(), device=self.device) < 0.5,
                -1.0,
                1.0,
            )
            self.vel_command_b[burst_ids] = burst_commands
            self.is_standing_env[burst_ids] = False
            self.time_left[burst_ids].uniform_(0.1, 0.5)

        eligible = ~(
            stop_mask
            | combined_to_straight
            | combined_to_reverse
            | straight_to_burst
            | previous_high_straight
        )
        high_mask = eligible & (
            torch.rand(ids.numel(), device=self.device)
            < self.cfg.high_combined_probability
        )
        high_ids = ids[high_mask]
        if high_ids.numel() == 0:
            return
        high_commands = torch.empty(
            (high_ids.numel(), 3),
            device=self.device,
        )
        high_commands[:, 0].uniform_(2.2, 2.6)
        high_commands[:, 1].uniform_(-0.1, 0.1)
        high_commands[:, 2].uniform_(1.8, 2.0)
        turn_sign = torch.where(
            torch.rand(high_ids.numel(), device=self.device) < 0.5,
            -1.0,
            1.0,
        )
        high_commands[:, 2] *= turn_sign
        self.vel_command_b[high_ids] = high_commands
        self.is_standing_env[high_ids] = False
        short_burst = torch.rand(
            high_ids.numel(), device=self.device
        ) < 0.5
        if torch.any(short_burst):
            self.time_left[high_ids[short_burst]].uniform_(0.8, 1.5)
        if torch.any(~short_burst):
            self.time_left[high_ids[~short_burst]].uniform_(6.0, 12.0)


@configclass
class RecoveryV05VelocityCommandCfg(UniformVelocityCommandCfg):
    """High-combined and stop-recovery mixture for Recovery v0.5."""

    class_type: type = RecoveryV05VelocityCommand
    high_combined_probability: float = 0.65
    high_combined_stop_probability: float = 0.35
    high_combined_straight_probability: float = 0.40
    warehouse_sequence_probability: float = 0.35
    refinery_replay_probability: float = 0.15
    turning_regression_probability: float = 0.40
    low_yaw_profile_probability: float = 0.25
    low_curve_profile_probability: float = 0.25
    high_curve_profile_probability: float = 0.25
