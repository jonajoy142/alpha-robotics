"""
Unit tests for FailureAttributor and boundary discovery.
"""

from alpha_robotics.attributor import FailureAttributor


def test_failure_attributor_cliff_detection():
    # Synthesize records showing a sharp cliff at latency = 4
    records = []
    latencies = [0, 1, 2, 4, 8]
    success_map = {0: 1.0, 1: 1.0, 2: 0.9, 4: 0.2, 8: 0.0}

    for cell_id, lat in enumerate(latencies):
        rate = success_map[lat]
        for ep in range(10):
            # 10 episodes per cell matching the rate
            is_succ = (ep < int(rate * 10))
            records.append({
                "cell_id": cell_id,
                "episode_id": ep,
                "seed": 42 + ep,
                "success": is_succ,
                "reward": 100.0 if is_succ else 10.0,
                "steps": 200 if is_succ else 20,
                "obs_latency_steps": lat,
            })

    attributor = FailureAttributor(records)
    diagnosis = attributor.analyze()

    assert diagnosis.nominal_success_rate == 1.0
    assert len(diagnosis.fault_sensitivities) == 1

    sens = diagnosis.fault_sensitivities[0]
    assert sens.fault_name == "obs_latency_steps"
    assert sens.cliff_detected is True
    # Failure threshold (first point <= 50% of nominal) should be 4
    assert sens.failure_threshold == 4.0
    assert diagnosis.most_vulnerable_fault == "obs_latency_steps"
    assert len(diagnosis.recommendations) > 0


def test_failure_attributor_resilient_policy():
    # Synthesize records where policy never fails
    records = []
    for cell_id, noise in enumerate([0.0, 0.05, 0.1]):
        for ep in range(5):
            records.append({
                "cell_id": cell_id,
                "episode_id": ep,
                "seed": 100 + ep,
                "success": True,
                "reward": 500.0,
                "steps": 500,
                "action_noise_std": noise,
            })

    attributor = FailureAttributor(records)
    diagnosis = attributor.analyze()

    assert diagnosis.nominal_success_rate == 1.0
    assert diagnosis.overall_robustness_score == 100.0
    assert diagnosis.grade == "A"
    assert diagnosis.most_vulnerable_fault is None

