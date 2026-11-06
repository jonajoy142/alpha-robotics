"""
Unit tests for SweepRunner.
"""

import gymnasium as gym
from alpha_robotics.runner import SweepRunner


def test_sweep_runner_marginal():
    def simple_policy(obs):
        return 0

    sweep_config = {
        "obs_latency_steps": [0, 2],
        "obs_noise_std": [0.0, 0.05],
    }

    runner = SweepRunner(
        env_fn=lambda: gym.make("CartPole-v1"),
        policy_fn=simple_policy,
        sweep=sweep_config,
        episodes_per_cell=2,
        seed=123,
        max_steps_per_episode=20,
    )

    records = runner.run()

    # Marginal sweep generates: baseline (1) + latency non-zero (1) + noise non-zero (1) = 3 cells * 2 episodes = 6
    assert len(records) == 6
    assert "cell_id" in records[0]
    assert "success" in records[0]
    assert "reward" in records[0]
    assert "steps" in records[0]


def test_sweep_runner_deterministic():
    def simple_policy(obs):
        return 1

    sweep_config = {"obs_noise_std": [0.0, 0.1]}

    r1 = SweepRunner(
        env_fn=lambda: gym.make("CartPole-v1"),
        policy_fn=simple_policy,
        sweep=sweep_config,
        episodes_per_cell=3,
        seed=999,
        max_steps_per_episode=15,
    ).run()

    r2 = SweepRunner(
        env_fn=lambda: gym.make("CartPole-v1"),
        policy_fn=simple_policy,
        sweep=sweep_config,
        episodes_per_cell=3,
        seed=999,
        max_steps_per_episode=15,
    ).run()

    for rec1, rec2 in zip(r1, r2):
        assert rec1["reward"] == rec2["reward"]
        assert rec1["steps"] == rec2["steps"]
        assert rec1["success"] == rec2["success"]
