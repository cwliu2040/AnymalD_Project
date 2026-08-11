import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY = PROJECT_ROOT / (
    "docs/validation/slam_confidence_dds_and_fast_live_gate_summary.json"
)


def test_dds_fault_gate_summary_is_complete_and_fail_closed() -> None:
    report = json.loads(SUMMARY.read_text(encoding="utf-8"))
    dds = report["dds_fault_qualification"]

    assert dds["transport"] == "rmw_cyclonedds_cpp"
    assert dds["case_count"] == 12
    assert dds["passed_case_count"] == dds["case_count"]
    assert dds["passed"] is True
    assert dds["stale_high_fail_closed"] is True
    assert dds["watchdog_forced_observation"] == [0.0, 0.0]
    assert len(dds["cases_per_backend"]) == 6


def test_fast_live_gate_summary_preserves_runtime_boundaries() -> None:
    report = json.loads(SUMMARY.read_text(encoding="utf-8"))
    live = report["fastlio2_live_qualification"]
    configuration = live["configuration"]

    assert configuration["deskew"] == "native"
    assert configuration["cube_side_length_m"] == 1000.0
    assert configuration["policy_odometry"] == "/slam/odom"
    assert configuration["ground_truth_runtime_input"] is False
    assert live["passed"] is True
    assert len(live["profiles"]) == 4
    for profile in live["profiles"]:
        assert profile["invalid_after_tracking_count"] == 0
        assert profile["terminated_count"] == 0
        assert profile["truncated_count"] == 0
        assert profile["classification"] == "no_instability"

    totals = live["all_profiles"]
    assert all(value == 0 for value in totals.values())
    assert live["retired_diagnostic_run"]["terminated_count"] == 1
