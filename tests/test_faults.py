"""
Unit tests for fault injection primitives.
"""

import numpy as np
import pytest
from alpha_robotics.faults import (
    LatencyFault,
    FrameDropFault,
    ObservationNoiseFault,
    ActionNoiseFault,
    ActionDelayFault,
    GripSlipFault,
    ReachLimitFault,
)


def test_latency_fault_ring_buffer():
    rng = np.random.default_rng(42)
    fault = LatencyFault(delay_steps=2)

    # Step 0: feed 10.0 -> buffer filling, returns earliest (10.0)
    out0 = fault.apply(np.array([10.0]), rng)
    assert np.allclose(out0, [10.0])

    # Step 1: feed 20.0 -> returns 10.0
    out1 = fault.apply(np.array([20.0]), rng)
    assert np.allclose(out1, [10.0])

    # Step 2: feed 30.0 -> returns 10.0 (delayed by 2 steps)
    out2 = fault.apply(np.array([30.0]), rng)
    assert np.allclose(out2, [10.0])

    # Step 3: feed 40.0 -> returns 20.0
    out3 = fault.apply(np.array([40.0]), rng)
    assert np.allclose(out3, [20.0])

    # Reset
    fault.reset()
    out_fresh = fault.apply(np.array([99.0]), rng)
    assert np.allclose(out_fresh, [99.0])


def test_frame_drop_fault():
    rng = np.random.default_rng(42)
    fault = FrameDropFault(drop_prob=1.0)  # always drop after initial frame

    # First frame stored
    f0 = fault.apply(np.array([1.0, 2.0]), rng)
    assert np.allclose(f0, [1.0, 2.0])

    # Second frame dropped -> returns first frame
    f1 = fault.apply(np.array([9.0, 9.0]), rng)
    assert np.allclose(f1, [1.0, 2.0])


def test_observation_noise_fault():
    rng = np.random.default_rng(42)
    fault = ObservationNoiseFault(noise_std=0.1)

    obs = np.zeros(1000)
    noisy = fault.apply(obs, rng)

    assert not np.allclose(obs, noisy)
    # Mean should be close to 0, std close to 0.1
    assert abs(np.mean(noisy)) < 0.05
    assert abs(np.std(noisy) - 0.1) < 0.02


def test_action_noise_fault():
    rng = np.random.default_rng(42)
    fault = ActionNoiseFault(noise_std=0.05)

    act = np.ones(500)
    noisy = fault.apply(act, rng)
    assert not np.allclose(act, noisy)
    assert abs(np.mean(noisy) - 1.0) < 0.05


def test_grip_slip_fault():
    rng = np.random.default_rng(42)
    fault = GripSlipFault(slip_prob=1.0, gripper_dim=-1)

    action = np.array([0.5, -0.2, 0.8, 1.0])
    slipped = fault.apply(action, rng)

    # Gripper dimension (last item) should be zeroed
    assert slipped[-1] == 0.0
    assert slipped[0] == 0.5


def test_reach_limit_fault():
    rng = np.random.default_rng(42)
    fault = ReachLimitFault(clamp_fraction=0.5)

    action = np.array([1.0, -2.0, 4.0])
    clamped = fault.apply(action, rng)

    assert np.allclose(clamped, [0.5, -1.0, 2.0])
