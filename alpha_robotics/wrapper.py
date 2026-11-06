"""
Universal ChaosWrapper for Gymnasium-compatible robotics environments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import gymnasium as gym
import numpy as np
import yaml

from alpha_robotics.faults import (
    BaseFault,
    LatencyFault,
    FrameDropFault,
    ObservationNoiseFault,
    ActionNoiseFault,
    ActionDelayFault,
    GripSlipFault,
    ReachLimitFault,
    FAULT_REGISTRY,
)


class ChaosWrapper(gym.Wrapper):
    """
    Gymnasium Wrapper that systematically intercepts observations and actions
    to inject calibrated interface faults (latency, noise, frame drops, grip slip).

    Example:
        ```python
        import gymnasium as gym
        from alpha_robotics import ChaosWrapper

        env = gym.make("FetchPickAndPlace-v2")
        env = ChaosWrapper(env, faults={"obs_latency_steps": 3, "grip_slip_prob": 0.1})
        obs, info = env.reset()
        ```
    """

    def __init__(
        self,
        env: gym.Env,
        faults: dict[str, Any] | str | Path | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(env)
        self.master_seed = seed
        self._rng = np.random.default_rng(seed)

        # Separate fault pipelines
        self.obs_faults: list[BaseFault] = []
        self.action_faults: list[BaseFault] = []
        self._active_fault_config: dict[str, Any] = {}

        if faults is not None:
            self.configure_faults(faults)

        # Telemetry & episode tracking
        self.episode_count = 0
        self.step_count = 0
        self.total_episodes_reward: list[float] = []
        self.episodes_success: list[bool] = []
        self.current_episode_reward = 0.0
        self.current_episode_steps = 0

    def configure_faults(self, faults: dict[str, Any] | str | Path) -> None:
        """Parse fault specification from dict or YAML file."""
        if isinstance(faults, (str, Path)):
            with open(faults, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            fault_dict = loaded.get("faults", loaded)
        else:
            fault_dict = dict(faults)

        self._active_fault_config = dict(fault_dict)
        self.obs_faults.clear()
        self.action_faults.clear()

        # Observation faults
        if "obs_latency_steps" in fault_dict and fault_dict["obs_latency_steps"] > 0:
            self.obs_faults.append(LatencyFault(fault_dict["obs_latency_steps"]))
        if "frame_drop_prob" in fault_dict and fault_dict["frame_drop_prob"] > 0.0:
            self.obs_faults.append(FrameDropFault(fault_dict["frame_drop_prob"]))
        if "obs_noise_std" in fault_dict and fault_dict["obs_noise_std"] > 0.0:
            self.obs_faults.append(ObservationNoiseFault(fault_dict["obs_noise_std"]))

        # Action faults
        if "action_delay_steps" in fault_dict and fault_dict["action_delay_steps"] > 0:
            self.action_faults.append(ActionDelayFault(fault_dict["action_delay_steps"]))
        if "action_noise_std" in fault_dict and fault_dict["action_noise_std"] > 0.0:
            self.action_faults.append(ActionNoiseFault(fault_dict["action_noise_std"]))
        if "grip_slip_prob" in fault_dict and fault_dict["grip_slip_prob"] > 0.0:
            gripper_dim = fault_dict.get("gripper_dim", -1)
            self.action_faults.append(GripSlipFault(fault_dict["grip_slip_prob"], gripper_dim=gripper_dim))
        if "reach_clamp_fraction" in fault_dict and fault_dict["reach_clamp_fraction"] < 1.0:
            self.action_faults.append(ReachLimitFault(fault_dict["reach_clamp_fraction"]))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset environment and all active fault states."""
        effective_seed = seed if seed is not None else (self.master_seed + self.episode_count)
        self._rng = np.random.default_rng(effective_seed)

        for fault in self.obs_faults:
            fault.reset()
        for fault in self.action_faults:
            fault.reset()

        self.current_episode_reward = 0.0
        self.current_episode_steps = 0

        obs, info = self.env.reset(seed=effective_seed, options=options)

        # Apply observation faults to initial state
        for fault in self.obs_faults:
            obs = fault.apply(obs, self._rng)

        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """
        Step with fault injection:
        1. Commanded action -> Action Faults -> Executed action
        2. Environment step
        3. Raw observation -> Observation Faults -> Delivered observation
        """
        self.step_count += 1
        self.current_episode_steps += 1

        # 1. Apply action faults
        perturbed_action = action
        for fault in self.action_faults:
            perturbed_action = fault.apply(perturbed_action, self._rng)

        # Ensure action adheres to environment action space if standard
        if hasattr(self.env, "action_space") and self.env.action_space is not None:
            if isinstance(self.env.action_space, gym.spaces.Discrete):
                if isinstance(perturbed_action, (int, float, np.number)):
                    perturbed_action = int(np.clip(round(float(perturbed_action)), 0, self.env.action_space.n - 1))
            elif isinstance(self.env.action_space, gym.spaces.Box):
                if isinstance(perturbed_action, np.ndarray):
                    perturbed_action = np.clip(perturbed_action, self.env.action_space.low, self.env.action_space.high)

        # 2. Step underlying simulator
        obs, reward, terminated, truncated, info = self.env.step(perturbed_action)
        self.current_episode_reward += float(reward)

        # 3. Apply observation faults
        for fault in self.obs_faults:
            obs = fault.apply(obs, self._rng)

        # 4. Check terminal condition & record episode stats
        if terminated or truncated:
            self.episode_count += 1
            self.total_episodes_reward.append(self.current_episode_reward)

            # Determine task success
            if "success" in info:
                success = bool(info["success"])
            elif "is_success" in info:
                success = bool(info["is_success"])
            elif truncated and not terminated:
                # Survived full episode horizon without premature failure
                success = True
            elif self.current_episode_steps >= 195 and "CartPole" in str(type(self.env.unwrapped)) and not terminated:
                success = True
            else:
                success = False

            self.episodes_success.append(success)
            info["alpha_episode_success"] = success
            info["alpha_episode_reward"] = self.current_episode_reward
            info["alpha_episode_steps"] = self.current_episode_steps

        return obs, reward, terminated, truncated, info

    @property
    def telemetry(self) -> dict[str, Any]:
        """Summary metrics across completed episodes."""
        total = len(self.episodes_success)
        success_count = sum(self.episodes_success)
        success_rate = (success_count / total) if total > 0 else 0.0
        mean_reward = (sum(self.total_episodes_reward) / total) if total > 0 else 0.0

        return {
            "episodes_total": total,
            "episodes_success": success_count,
            "success_rate": success_rate,
            "mean_reward": mean_reward,
            "active_fault_config": dict(self._active_fault_config),
        }
