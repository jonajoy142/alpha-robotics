"""
Automated SweepRunner for executing systematic reliability stress batteries.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Callable
import yaml

from alpha_robotics.wrapper import ChaosWrapper


class SweepRunner:
    """
    Executes a structured stress matrix across policy checkpoints and simulation environments.

    Example:
        ```python
        runner = SweepRunner(
            env_fn=lambda: gym.make("CartPole-v1"),
            policy_fn=my_policy,
            sweep={
                "obs_latency_steps": [0, 1, 2, 4, 8],
                "obs_noise_std": [0.0, 0.05, 0.1, 0.2],
            },
            episodes_per_cell=5,
            seed=42,
        )
        results = runner.run()
        ```
    """

    def __init__(
        self,
        env_fn: Callable[[], Any],
        policy_fn: Callable[[Any], Any],
        sweep: dict[str, list[Any]] | str | Path,
        episodes_per_cell: int = 10,
        seed: int = 42,
        max_steps_per_episode: int = 1000,
        mode: str = "marginal",
    ) -> None:
        self.env_fn = env_fn
        self.policy_fn = policy_fn
        self.episodes_per_cell = max(1, int(episodes_per_cell))
        self.master_seed = int(seed)
        self.max_steps = int(max_steps_per_episode)
        self.mode = mode.lower()

        # Parse sweep configuration
        if isinstance(sweep, (str, Path)):
            with open(sweep, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            self.sweep_grid = loaded.get("sweep", loaded)
            if "episodes_per_cell" in loaded:
                self.episodes_per_cell = int(loaded["episodes_per_cell"])
        else:
            self.sweep_grid = {k: list(v) for k, v in sweep.items()}

    def _generate_cells(self) -> list[dict[str, Any]]:
        """Generate list of fault configurations to evaluate."""
        cells: list[dict[str, Any]] = []

        if self.mode == "cartesian":
            keys = list(self.sweep_grid.keys())
            values = list(self.sweep_grid.values())
            for combination in itertools.product(*values):
                cells.append(dict(zip(keys, combination)))
        else:
            # Default "marginal" sweep: evaluate each fault dimension independently
            # First, add baseline nominal cell (all faults at zero/neutral)
            baseline = {k: (1.0 if k == "reach_clamp_fraction" else 0) for k in self.sweep_grid}
            cells.append(baseline)

            for key, val_list in self.sweep_grid.items():
                for val in val_list:
                    # Skip duplicate of baseline
                    cfg = dict(baseline)
                    cfg[key] = val
                    if cfg not in cells:
                        cells.append(cfg)

        return cells

    def run(self) -> list[dict[str, Any]]:
        """
        Run the complete stress battery across all generated fault cells.
        Returns a structured list of episode records.
        """
        cells = self._generate_cells()
        records: list[dict[str, Any]] = []

        for cell_idx, fault_config in enumerate(cells):
            base_env = self.env_fn()
            wrapped_env = ChaosWrapper(base_env, faults=fault_config, seed=self.master_seed)

            for ep_idx in range(self.episodes_per_cell):
                if hasattr(self.policy_fn, "reset"):
                    self.policy_fn.reset()
                ep_seed = self.master_seed + cell_idx * 1000 + ep_idx
                obs, info = wrapped_env.reset(seed=ep_seed)

                ep_reward = 0.0
                steps = 0
                terminated = False
                truncated = False

                for _ in range(self.max_steps):
                    action = self.policy_fn(obs)
                    obs, reward, terminated, truncated, step_info = wrapped_env.step(action)
                    ep_reward += float(reward)
                    steps += 1
                    if terminated or truncated:
                        break

                # Determine success
                success = bool(step_info.get("alpha_episode_success", False))

                record = {
                    "cell_id": cell_idx,
                    "episode_id": ep_idx,
                    "seed": ep_seed,
                    "success": success,
                    "reward": ep_reward,
                    "steps": steps,
                    **fault_config,
                }
                records.append(record)

            wrapped_env.close()

        return records

    def to_dataframe(self, records: list[dict[str, Any]]) -> Any:
        """Helper to convert records to pandas DataFrame if pandas is installed."""
        try:
            import pandas as pd
            return pd.DataFrame(records)
        except ImportError:
            return records

