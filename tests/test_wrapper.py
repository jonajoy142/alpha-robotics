"""
Unit tests for ChaosWrapper.
"""

import gymnasium as gym
import numpy as np
import pytest
from alpha_robotics.wrapper import ChaosWrapper


def test_wrapper_baseline_no_faults():
    base_env = gym.make("CartPole-v1")
    wrapped = ChaosWrapper(base_env, faults={}, seed=42)

    obs, info = wrapped.reset(seed=42)
    assert obs is not None
    assert isinstance(obs, np.ndarray)

    for _ in range(10):
        obs, reward, terminated, truncated, info = wrapped.step(0)
        if terminated or truncated:
            break

    wrapped.close()
    assert wrapped.step_count > 0


def test_wrapper_latency_injection():
    base_env = gym.make("CartPole-v1")
    wrapped = ChaosWrapper(base_env, faults={"obs_latency_steps": 3}, seed=42)

    obs0, _ = wrapped.reset(seed=42)
    observations = [obs0]

    for _ in range(5):
        obs, _, term, trunc, _ = wrapped.step(1)
        observations.append(obs)
        if term or trunc:
            break

    # With 3 steps latency, observations[1] should equal observations[0]
    assert np.allclose(observations[0], observations[1])
    wrapped.close()


def test_wrapper_deterministic_reproducibility():
    """Verify that same seed generates strictly identical episode trajectories."""
    def run_trajectory(seed):
        env = gym.make("CartPole-v1")
        wrapped = ChaosWrapper(
            env,
            faults={"obs_noise_std": 0.05},
            seed=seed,
        )
        obs, _ = wrapped.reset(seed=seed)
        traj = [obs]
        for _ in range(15):
            obs, reward, term, trunc, _ = wrapped.step(0)
            traj.append(obs)
            if term or trunc:
                break
        wrapped.close()
        return traj

    traj1 = run_trajectory(42)
    traj2 = run_trajectory(42)

    assert len(traj1) == len(traj2)
    for o1, o2 in zip(traj1, traj2):
        assert np.allclose(o1, o2)
