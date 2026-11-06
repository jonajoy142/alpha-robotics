"""
Alpha Robotics — End-to-End CartPole Stress Audit Example.

This script demonstrates:
1. Wrapping a standard Gymnasium environment
2. Running a systematic stress battery over latency & noise
3. Discovering the exact failure boundary
4. Generating an interactive HTML Reliability Scorecard
"""

import gymnasium as gym
import numpy as np

from alpha_robotics import SweepRunner, FailureAttributor, ReportGenerator


# Calibrated balancing policy for CartPole
def heuristic_policy(obs):
    # obs: [cart_pos, cart_velocity, pole_angle, pole_velocity]
    cart_pos, cart_vel, pole_angle, pole_vel = obs
    # Linear balance controller
    action = 1 if (pole_angle * 1.5 + pole_vel * 0.5 + cart_pos * 0.2 + cart_vel * 0.1) > 0 else 0
    return action


def main():
    print("=" * 60)
    print("ALPHA ROBOTICS — POLICY STRESS AUDIT DEMO")
    print("=" * 60)

    # 1. Define stress battery
    sweep = {
        "obs_latency_steps": [0, 2, 4, 8, 12, 16],
        "obs_noise_std": [0.0, 0.05, 0.1, 0.25, 0.5],
        "action_noise_std": [0.0, 0.05, 0.1, 0.2, 0.4],
    }

    print("\n1. Running stress sweeps on CartPole-v1...")
    runner = SweepRunner(
        env_fn=lambda: gym.make("CartPole-v1"),
        policy_fn=heuristic_policy,
        sweep=sweep,
        episodes_per_cell=5,
        seed=42,
        max_steps_per_episode=500,
    )
    records = runner.run()
    print(f"   Collected {len(records)} test episodes across {len(set(r['cell_id'] for r in records))} cells.")

    # 2. Analyze failure attribution & boundary thresholds
    print("\n2. Analyzing failure boundaries and cliff thresholds...")
    attributor = FailureAttributor(records)
    diagnosis = attributor.analyze()

    print(f"\n   >>> OVERALL ROBUSTNESS SCORE: {diagnosis.overall_robustness_score}/100 (GRADE: {diagnosis.grade}) <<<")
    print(f"   Nominal Success Rate:    {diagnosis.nominal_success_rate * 100:.1f}%")
    print(f"   Worst-Case Success Rate: {diagnosis.worst_case_success_rate * 100:.1f}%")

    print("\n   Vulnerability Breakdown:")
    for s in diagnosis.fault_sensitivities:
        cliff_str = " [CLIFF DETECTED]" if s.cliff_detected else ""
        thresh_str = f"Critical boundary: {s.failure_threshold}" if s.failure_threshold is not None else "Resilient"
        print(f"   - {s.fault_name:20s}: {thresh_str:25s} (Max drop: -{s.max_success_drop * 100:.1f}%){cliff_str}")

    # 3. Export self-contained HTML scorecard
    output_path = "cartpole_reliability_scorecard.html"
    print(f"\n3. Exporting HTML Reliability Report to {output_path}...")
    reporter = ReportGenerator(
        diagnosis=diagnosis,
        records=records,
        title="Alpha Audit: CartPole-v1 Heuristic Policy",
        env_name="CartPole-v1",
    )
    saved = reporter.save(output_path)
    print(f"   Done! Open file in your browser to view interactive curves:\n   file://{saved.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
