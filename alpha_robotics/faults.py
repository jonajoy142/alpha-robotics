"""
Fault injection primitives for robot policy observations and actions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Any
import numpy as np


class BaseFault(ABC):
    """Abstract base class for all fault primitives."""

    @abstractmethod
    def apply(self, data: Any, rng: np.random.Generator) -> Any:
        """Apply the fault to observation or action data."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal fault state (e.g., ring buffers) at episode boundary."""
        pass


class LatencyFault(BaseFault):
    """
    Simulates observation delay using a FIFO ring buffer.
    Returns observations from `delay_steps` in the past.
    """

    def __init__(self, delay_steps: int = 0) -> None:
        self.delay_steps = max(0, int(delay_steps))
        self._buffer: deque[Any] = deque(maxlen=self.delay_steps + 1)

    def apply(self, data: Any, rng: np.random.Generator) -> Any:
        if self.delay_steps == 0:
            return data

        # Deep copy array or dict to prevent mutation
        stored = _copy_data(data)
        self._buffer.append(stored)

        if len(self._buffer) <= self.delay_steps:
            # While buffer is filling up, return the earliest observation
            return _copy_data(self._buffer[0])
        return _copy_data(self._buffer[0])

    def reset(self) -> None:
        self._buffer.clear()


class FrameDropFault(BaseFault):
    """
    Simulates dropped camera frames or missing sensor packets.
    With probability `drop_prob`, returns the previous observation.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        self.drop_prob = float(np.clip(drop_prob, 0.0, 1.0))
        self._last_data: Any = None

    def apply(self, data: Any, rng: np.random.Generator) -> Any:
        if self.drop_prob <= 0.0 or self._last_data is None:
            self._last_data = _copy_data(data)
            return data

        if rng.uniform(0.0, 1.0) < self.drop_prob:
            # Drop frame: return stale cached observation
            return _copy_data(self._last_data)

        self._last_data = _copy_data(data)
        return data

    def reset(self) -> None:
        self._last_data = None


class ObservationNoiseFault(BaseFault):
    """
    Adds zero-mean Gaussian noise to numerical observations.
    Works on continuous numpy arrays and dictionaries of arrays.
    """

    def __init__(self, noise_std: float = 0.0) -> None:
        self.noise_std = max(0.0, float(noise_std))

    def apply(self, data: Any, rng: np.random.Generator) -> Any:
        if self.noise_std <= 0.0:
            return data
        return _add_gaussian_noise(data, self.noise_std, rng)

    def reset(self) -> None:
        pass


class ActionNoiseFault(BaseFault):
    """
    Adds zero-mean Gaussian perturbation to commanded robot actions.
    Simulates motor torque ripple, servo tracking jitter, and gear backlash.
    Works for both continuous tensor actions and discrete integer actions.
    """

    def __init__(self, noise_std: float = 0.0) -> None:
        self.noise_std = max(0.0, float(noise_std))

    def apply(self, data: Any, rng: np.random.Generator) -> Any:
        if self.noise_std <= 0.0:
            return data

        # Preserve discrete action integer type if integer scalar
        if isinstance(data, (int, np.integer)) and not isinstance(data, bool):
            noise = float(rng.normal(0.0, self.noise_std))
            return int(round(data + noise))

        return _add_gaussian_noise(data, self.noise_std, rng)

    def reset(self) -> None:
        pass


class ActionDelayFault(BaseFault):
    """
    Simulates communication or motor command latency between policy and actuators.
    """

    def __init__(self, delay_steps: int = 0) -> None:
        self.delay_steps = max(0, int(delay_steps))
        self._buffer: deque[Any] = deque(maxlen=self.delay_steps + 1)

    def apply(self, data: Any, rng: np.random.Generator) -> Any:
        if self.delay_steps == 0:
            return data

        stored = _copy_data(data)
        self._buffer.append(stored)

        if len(self._buffer) <= self.delay_steps:
            return _copy_data(self._buffer[0])
        return _copy_data(self._buffer[0])

    def reset(self) -> None:
        self._buffer.clear()


class GripSlipFault(BaseFault):
    """
    Simulates mechanical gripper slip, object drop, or suction loss.
    With probability `slip_prob`, zeroes out the gripper control dimension
    (by default, the final dimension of the action vector).
    """

    def __init__(self, slip_prob: float = 0.0, gripper_dim: int = -1) -> None:
        self.slip_prob = float(np.clip(slip_prob, 0.0, 1.0))
        self.gripper_dim = gripper_dim

    def apply(self, data: Any, rng: np.random.Generator) -> Any:
        if self.slip_prob <= 0.0:
            return data

        if rng.uniform(0.0, 1.0) < self.slip_prob:
            copied = _copy_data(data)
            if isinstance(copied, np.ndarray) and copied.size > 0:
                copied[self.gripper_dim] = 0.0
                return copied
            if isinstance(copied, list) and len(copied) > 0:
                copied[self.gripper_dim] = 0.0
                return copied
        return data

    def reset(self) -> None:
        pass


class ReachLimitFault(BaseFault):
    """
    Simulates kinematically unreachable targets, arm joint limits,
    or artificial workspace boundary clamps.
    Clips action components to a fraction of their commanded range.
    """

    def __init__(self, clamp_fraction: float = 1.0) -> None:
        self.clamp_fraction = float(np.clip(clamp_fraction, 0.05, 1.0))

    def apply(self, data: Any, rng: np.random.Generator) -> Any:
        if self.clamp_fraction >= 1.0:
            return data

        if isinstance(data, np.ndarray):
            return data * self.clamp_fraction
        if isinstance(data, (list, tuple)):
            return [x * self.clamp_fraction for x in data]
        return data

    def reset(self) -> None:
        pass


# Helper utilities
def _copy_data(data: Any) -> Any:
    if isinstance(data, np.ndarray):
        return np.copy(data)
    if isinstance(data, dict):
        return {k: _copy_data(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_copy_data(x) for x in data]
    return data


def _add_gaussian_noise(data: Any, std: float, rng: np.random.Generator) -> Any:
    if isinstance(data, np.ndarray):
        if np.issubdtype(data.dtype, np.floating) or np.issubdtype(data.dtype, np.integer):
            noise = rng.normal(0.0, std, size=data.shape).astype(data.dtype)
            return data + noise
        return data
    if isinstance(data, dict):
        return {k: _add_gaussian_noise(v, std, rng) for k, v in data.items()}
    if isinstance(data, list):
        arr = np.array(data, dtype=float)
        arr = arr + rng.normal(0.0, std, size=arr.shape)
        return arr.tolist()
    if isinstance(data, (int, float)):
        return data + float(rng.normal(0.0, std))
    return data


# Canonical registry mapping user configuration keys to Fault classes
FAULT_REGISTRY: dict[str, type[BaseFault]] = {
    "obs_latency_steps": LatencyFault,
    "frame_drop_prob": FrameDropFault,
    "obs_noise_std": ObservationNoiseFault,
    "action_noise_std": ActionNoiseFault,
    "action_delay_steps": ActionDelayFault,
    "grip_slip_prob": GripSlipFault,
    "reach_clamp_fraction": ReachLimitFault,
}
