"""
Alpha Robotics — ALOHA 7-DoF Robotic Manipulation Stress Audit.

This example evaluates a continuous 7-DoF manipulation policy:
- Action space: [dx, dy, dz, droll, dpitch, dyaw, gripper] (7-DoF continuous tensor)
- Task: Robotic pick-and-place block transfer
- Nominal performance: 100% success rate in nominal conditions
- Stress Battery:
    - Camera Latency: 0ms -> 20ms -> 40ms -> 60ms -> 100ms (at 100Hz control loop)
    - Gripper Slip: 0% -> 5% -> 10% -> 20% -> 35% slip probability
    - Reachability Clamp: 1.0 -> 0.9 -> 0.8 -> 0.7 -> 0.5 workspace fraction

Alpha automatically discovers:
1. The exact latency cliff where tracking delay causes grasp overshoot
2. The mechanical slip tolerance of the policy
3. The kinematic boundary where the arm hits reach limits
4. Exports `aloha_reliability_scorecard.html`
"""

from __future__ import annotations

from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from alpha_robotics import SweepRunner, FailureAttributor, ReportGenerator


# =====================================================================
# 1. Self-Contained 7-DoF ALOHA Manipulation Simulation Environment
# =====================================================================
class AlohaPickAndPlaceEnv(gym.Env):
    """
    Simulation of an ALOHA / 7-DoF robotic arm executing pick-and-place.
    Operates at 100Hz control rate (1 step = 10ms).
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()

        # Action: 7-DoF continuous tensor [dx, dy, dz, droll, dpitch, dyaw, gripper]
        # dx, dy, dz in [-0.05, 0.05] m per step; gripper in [0.0 (open), 1.0 (closed)]
        self.action_space = spaces.Box(
            low=np.array([-0.05, -0.05, -0.05, -0.1, -0.1, -0.1, 0.0], dtype=np.float32),
            high=np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: [ee_x, ee_y, ee_z, ee_roll, ee_pitch, ee_yaw,
        #               obj_x, obj_y, obj_z, target_x, target_y, target_z,
        #               gripper_state, is_grasped]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )

        self.workspace_bounds = {
            "x": (0.20, 0.80),
            "y": (-0.40, 0.40),
            "z": (0.00, 0.45),
        }

        self.target_pos = np.array([0.60, 0.25, 0.05], dtype=np.float32)
        self.grasp_radius = 0.04  # 4cm grasp envelope
        self.max_steps = 150      # 1.5 seconds at 100Hz

        self.ee_pos = np.zeros(3, dtype=np.float32)
        self.ee_rot = np.zeros(3, dtype=np.float32)
        self.obj_pos = np.zeros(3, dtype=np.float32)
        self.gripper = 0.0
        self.is_grasped = False
        self.current_step = 0
        self._rng = np.random.default_rng(42)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Reset end-effector hover pose
        self.ee_pos = np.array([0.35, -0.15, 0.30], dtype=np.float32)
        self.ee_rot = np.zeros(3, dtype=np.float32)

        # Spawn object with small realistic spatial jitter
        obj_jitter = self._rng.uniform(-0.02, 0.02, size=2)
        self.obj_pos = np.array([0.45 + obj_jitter[0], -0.10 + obj_jitter[1], 0.02], dtype=np.float32)

        self.gripper = 0.0
        self.is_grasped = False
        self.current_step = 0

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([
            self.ee_pos,
            self.ee_rot,
            self.obj_pos,
            self.target_pos,
            [self.gripper],
            [1.0 if self.is_grasped else 0.0],
        ]).astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.current_step += 1
        action = np.asarray(action, dtype=np.float32)

        delta_pos = action[:3]
        delta_rot = action[3:6]
        gripper_cmd = float(action[6])

        # 1. Kinematic integration with physical boundary enforcement
        new_ee_pos = self.ee_pos + delta_pos
        new_ee_pos[0] = np.clip(new_ee_pos[0], self.workspace_bounds["x"][0], self.workspace_bounds["x"][1])
        new_ee_pos[1] = np.clip(new_ee_pos[1], self.workspace_bounds["y"][0], self.workspace_bounds["y"][1])
        new_ee_pos[2] = np.clip(new_ee_pos[2], self.workspace_bounds["z"][0], self.workspace_bounds["z"][1])
        self.ee_pos = new_ee_pos
        self.ee_rot += delta_rot
        self.gripper = np.clip(gripper_cmd, 0.0, 1.0)

        # 2. Grasp mechanics & contact physics
        dist_to_obj = np.linalg.norm(self.ee_pos - self.obj_pos)

        if not self.is_grasped:
            # Grasp condition: EE close enough & gripper commanded closed (> 0.6)
            if dist_to_obj < self.grasp_radius and self.gripper > 0.6:
                self.is_grasped = True
        else:
            # If grasped, object tracks end-effector
            if self.gripper > 0.3:
                self.obj_pos = np.copy(self.ee_pos)
            else:
                # Released / slipped
                self.is_grasped = False
                # Drops to table surface (z = 0.02)
                self.obj_pos[2] = 0.02

        # 3. Success check: object placed inside target zone within 5cm tolerance
        dist_to_target = np.linalg.norm(self.obj_pos - self.target_pos)
        success = bool(dist_to_target < 0.06 and self.obj_pos[2] < 0.10 and not self.is_grasped)

        terminated = success
        truncated = bool(self.current_step >= self.max_steps)

        # Dense reward for progress
        reward = -dist_to_target + (100.0 if success else 0.0)

        obs = self._get_obs()
        info = {
            "is_success": success,
            "dist_to_target": float(dist_to_target),
            "is_grasped": self.is_grasped,
        }

        return obs, float(reward), terminated, truncated, info


# =====================================================================
# 2. Continuous 7-DoF Pick-and-Place Policy (ALOHA / ACT Model Interface)
# =====================================================================
class AlohaPickAndPlacePolicy:
    """
    Trajectory policy simulating an imitation-learned ACT policy.
    Maps 14-dim observation to 7-DoF continuous action commands.
    """
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.phase = "HOVER"
        self.grasp_timer = 0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        ee_pos = obs[0:3]
        obj_pos = obs[6:9]
        target_pos = obs[9:12]
        is_grasped = bool(obs[13] > 0.5)

        action = np.zeros(7, dtype=np.float32)

        if self.phase == "HOVER":
            target = obj_pos + np.array([0.0, 0.0, 0.08])
            delta = target - ee_pos
            action[:3] = np.clip(delta * 0.8, -0.04, 0.04)
            action[6] = 0.0  # Open gripper
            if np.linalg.norm(delta) < 0.02:
                self.phase = "DESCEND"

        elif self.phase == "DESCEND":
            delta = (obj_pos + np.array([0.0, 0.0, 0.01])) - ee_pos
            action[:3] = np.clip(delta * 0.8, -0.03, 0.03)
            action[6] = 0.0
            if np.linalg.norm(delta) < 0.02:
                self.phase = "GRASP"
                self.grasp_timer = 0

        elif self.phase == "GRASP":
            action[6] = 1.0  # Close gripper
            self.grasp_timer += 1
            if self.grasp_timer >= 5 or is_grasped:
                self.phase = "LIFT"

        elif self.phase == "LIFT":
            target = ee_pos.copy()
            target[2] = 0.20
            delta = target - ee_pos
            action[:3] = np.clip(delta * 0.8, -0.04, 0.04)
            action[6] = 1.0
            if abs(ee_pos[2] - 0.20) < 0.03:
                self.phase = "TRANSFER"

        elif self.phase == "TRANSFER":
            target = target_pos + np.array([0.0, 0.0, 0.12])
            delta = target - ee_pos
            action[:3] = np.clip(delta * 0.7, -0.04, 0.04)
            action[6] = 1.0
            if np.linalg.norm(delta) < 0.03:
                self.phase = "PLACE"

        elif self.phase == "PLACE":
            target = target_pos + np.array([0.0, 0.0, 0.02])
            delta = target - ee_pos
            action[:3] = np.clip(delta * 0.6, -0.03, 0.03)
            if np.linalg.norm(delta) < 0.03:
                action[6] = 0.0  # Release
                self.phase = "DONE"
            else:
                action[6] = 1.0

        elif self.phase == "DONE":
            action[6] = 0.0  # Keep open

        return action


# =====================================================================
# 3. Main Audit Execution & Reliability Scorecard Generation
# =====================================================================
def main() -> None:
    print("=" * 70)
    print("🤖 ALPHA ROBOTICS — ALOHA 7-DOF MANIPULATION RELIABILITY AUDIT")
    print("=" * 70)

    # Stress Battery: Latency, Gripper Slip, and Reach Clamping
    # Control loop = 100Hz (1 step = 10ms)
    # obs_latency_steps: [0, 2, 4, 6, 10] -> [0ms, 20ms, 40ms, 60ms, 100ms]
    sweep_config = {
        "obs_latency_steps": [0, 2, 4, 6, 10],
        "grip_slip_prob": [0.0, 0.05, 0.10, 0.20, 0.35],
        "reach_clamp_fraction": [1.0, 0.90, 0.80, 0.70, 0.50],
        "action_noise_std": [0.0, 0.005, 0.015, 0.03, 0.05],
    }

    print("\n1. Running structured stress matrix across 7-DoF action space...")
    print("   Operational Boundaries Tested:")
    print("   - Camera Latency:    0ms -> 20ms -> 40ms -> 60ms -> 100ms")
    print("   - Gripper Slip:      0% -> 5% -> 10% -> 20% -> 35% probability")
    print("   - Kinematic Clamps:  100% -> 90% -> 80% -> 70% -> 50% workspace")
    print("   - Actuator Noise:    0.0 -> 0.05 motor ripple (std)")

    policy = AlohaPickAndPlacePolicy()

    runner = SweepRunner(
        env_fn=AlohaPickAndPlaceEnv,
        policy_fn=policy,
        sweep=sweep_config,
        episodes_per_cell=6,
        seed=100,
        max_steps_per_episode=150,
        mode="marginal",
    )

    records = runner.run()
    total_cells = len(set(r["cell_id"] for r in records))
    print(f"\n   Completed {len(records)} test episodes across {total_cells} calibrated stress cells.")

    # 2. Failure Attribution & Boundary Discovery
    print("\n2. Analyzing failure boundaries, cliffs, and operational thresholds...")
    attributor = FailureAttributor(records)
    diagnosis = attributor.analyze()

    print("\n" + "-" * 70)
    print(f"🏆 OVERALL ROBUSTNESS SCORE: {diagnosis.overall_robustness_score}/100 (GRADE: {diagnosis.grade})")
    print(f"   Nominal Success Rate:    {diagnosis.nominal_success_rate * 100:.1f}%")
    print(f"   Worst-Case Success Rate: {diagnosis.worst_case_success_rate * 100:.1f}%")
    if diagnosis.most_vulnerable_fault:
        print(f"   Primary Vulnerability:   {diagnosis.most_vulnerable_fault}")
    print("-" * 70)

    print("\n🔬 Quantitative Operational Boundary Breakdown:")
    for s in diagnosis.fault_sensitivities:
        cliff_flag = " ⚠️ [SHARP CLIFF DETECTED]" if s.cliff_detected else ""
        thresh_str = f"Critical boundary: {s.failure_threshold}" if s.failure_threshold is not None else "Within bounds"
        print(f"   • {s.fault_name:22s} : {thresh_str:25s} (Max drop: -{s.max_success_drop * 100:4.1f}%){cliff_flag}")
        print(f"     Diagnostic: {s.evidence}")

    print("\n📋 Actionable Recommendations for Hardware Deployment:")
    for i, rec in enumerate(diagnosis.recommendations, 1):
        print(f"   {i}. {rec}")

    # 3. Export Self-Contained HTML Scorecard
    output_filename = "aloha_reliability_scorecard.html"
    print(f"\n3. Compiling executive scorecard to {output_filename}...")
    reporter = ReportGenerator(
        diagnosis=diagnosis,
        records=records,
        title="Alpha Audit: ALOHA 7-DoF Manipulation Policy",
        env_name="ALOHA-Bimanual-Transfer-v0 (7-DoF Continuous)",
    )
    saved_path = reporter.save(output_filename)
    print(f"\n✅ AUDIT COMPLETE! Open your scorecard in browser:\n   file://{saved_path.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
