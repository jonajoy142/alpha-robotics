"""
Command Line Interface (CLI) for Alpha Robotics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import gymnasium as gym

from alpha_robotics.runner import SweepRunner
from alpha_robotics.attributor import FailureAttributor
from alpha_robotics.report import ReportGenerator


def _default_random_policy(action_space: gym.Space):
    def policy(obs):
        return action_space.sample()
    return policy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="alpha",
        description="Alpha Robotics — Automated Reliability Boundary Discovery & Policy Audit",
    )
    subparsers = parser.add_subparsers(dest="command")

    # `alpha audit`
    audit_parser = subparsers.add_parser("audit", help="Run a systematic stress audit on an environment/policy")
    audit_parser.add_argument("--env", type=str, default="CartPole-v1", help="Gymnasium environment ID (default: CartPole-v1)")
    audit_parser.add_argument("--profile", type=str, default=None, help="Path to custom YAML stress profile")
    audit_parser.add_argument("--episodes", type=int, default=5, help="Episodes per stress cell (default: 5)")
    audit_parser.add_argument("--seed", type=int, default=42, help="Master random seed (default: 42)")
    audit_parser.add_argument("--output", type=str, default="alpha_report.html", help="Output path for HTML report (default: alpha_report.html)")
    audit_parser.add_argument("--fail-under", type=float, default=None, help="Exit code 1 if robustness score is below this threshold (CI gate)")

    args = parser.parse_args(argv)

    if args.command == "audit":
        print(f"🚀 Initializing Alpha Reliability Audit...")
        print(f"   Environment: {args.env}")
        print(f"   Episodes/Cell: {args.episodes} | Seed: {args.seed}")

        # Default profile if not provided
        profile_path = args.profile
        if profile_path is None:
            profile_path = Path(__file__).parent / "profiles" / "standard.yaml"

        probe_env = gym.make(args.env)
        policy_fn = _default_random_policy(probe_env.action_space)
        probe_env.close()

        runner = SweepRunner(
            env_fn=lambda: gym.make(args.env),
            policy_fn=policy_fn,
            sweep=profile_path,
            episodes_per_cell=args.episodes,
            seed=args.seed,
        )

        print("⚡ Executing stress matrix sweeps...")
        records = runner.run()
        print(f"   Completed {len(records)} test episodes across {len(set(r['cell_id'] for r in records))} cells.")

        print("🔍 Computing failure attribution & boundary thresholds...")
        attributor = FailureAttributor(records)
        diagnosis = attributor.analyze()

        print(f"\n📊 Audit Results:")
        print(f"   Overall Score:   {diagnosis.overall_robustness_score}/100 (Grade: {diagnosis.grade})")
        print(f"   Nominal Success: {diagnosis.nominal_success_rate * 100:.1f}%")
        print(f"   Worst-Case:      {diagnosis.worst_case_success_rate * 100:.1f}%")
        if diagnosis.most_vulnerable_fault:
            print(f"   Main Bottleneck: {diagnosis.most_vulnerable_fault}")

        # Generate report
        reporter = ReportGenerator(
            diagnosis=diagnosis,
            records=records,
            title=f"Alpha Audit: {args.env}",
            env_name=args.env,
        )
        saved_file = reporter.save(args.output)
        print(f"\n✅ Interactive report generated: {saved_file.resolve()}")

        # CI failure threshold
        if args.fail_under is not None and diagnosis.overall_robustness_score < args.fail_under:
            print(f"\n❌ FAILED CI GATE: Score {diagnosis.overall_robustness_score} is below threshold {args.fail_under}")
            return 1

        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

