"""Focused tests for the versioned FABLE v2 lifecycle runners."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import xarray as xr
import yaml

import davinci_monet.tests.synthetic.fable_v2_acceptance as acceptance_module
import davinci_monet.tests.synthetic.fable_v2_calibration_runner as calibration_runner_module
import davinci_monet.tests.synthetic.fable_v2_lifecycle as lifecycle_module
import davinci_monet.tests.synthetic.fable_v2_preflight as preflight_module
import davinci_monet.tests.synthetic.fable_v2_runner as runner
from davinci_monet.tests.synthetic._aerosol_contracts import SyntheticTuningSpec, spec_hash
from davinci_monet.tests.synthetic.fable_acceptance_gate import resource_gate
from davinci_monet.tests.synthetic.fable_v2_acceptance import run_v2_acceptance
from davinci_monet.tests.synthetic.fable_v2_calibration import (
    V2CalibrationCandidate,
    V2NullSeedResult,
    V2RecoverySeedResult,
)
from davinci_monet.tests.synthetic.fable_v2_calibration_record import (
    load_v2_calibration_record,
    select_v2_calibration_policy,
    write_v2_calibration_record,
)
from davinci_monet.tests.synthetic.fable_v2_calibration_runner import run_v2_calibration
from davinci_monet.tests.synthetic.fable_v2_freeze import (
    FrozenFileIdentity,
    V2FileBinding,
    V2Preregistration,
    V2TestEvidence,
    write_v2_preregistration,
)
from davinci_monet.tests.synthetic.fable_v2_lifecycle import prepare_v2_generation
from davinci_monet.tests.synthetic.fable_v2_policy import (
    FableV2Policy,
    v2_calibration_policies,
    v2_fitting_policy_values,
)
from davinci_monet.tests.synthetic.fable_v2_preflight import (
    V2PreflightRecord,
    run_v2_preflight,
    write_v2_preflight_record,
)
from davinci_monet.tests.synthetic.fable_v2_protocol import ACCEPTANCE_SEEDS, seed_roles
from davinci_monet.tests.synthetic.fable_v2_record_io import json_sha256
from davinci_monet.tests.synthetic.fable_v2_runner import V2ScenarioOutcome


def _identity(tmp_path: Path, name: str) -> FrozenFileIdentity:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(name, encoding="ascii")
    return FrozenFileIdentity.capture(path)


def _outcome(
    tmp_path: Path,
    seed: int,
    policy: FableV2Policy,
    *,
    score_kind: str = "recovery",
    nrmse: float = 0.2,
    evaluation_splits: tuple[str, ...] = ("calibration",),
) -> V2ScenarioOutcome:
    identity = _identity(tmp_path, f"evidence-{policy.policy_id}-{seed}-{score_kind}.json")
    if score_kind == "recovery":
        score: dict[str, Any] = {
            "diagnostics": {
                "estimate_vs_unfiltered_in_span_nrmse": 0.5,
                "excluded_fraction": 0.2,
            },
            "metrics": {
                "aod_rmse_ratio": 0.5,
                "field_correlation": 0.95,
                "field_nrmse": nrmse,
                "field_origin_slope": 1.0,
                "full_target_aod_rmse_ratio": 0.7,
            },
            "failures": [],
            "passed": True,
        }
        diagnostic_hash: str | None = "d" * 64
        learned_nrmse: float | None = 0.03
    else:
        score = {
            "metrics": {
                "null_retained_energy_fraction": 0.03,
                "null_significant_fraction": 0.02,
            },
            "passed": True,
        }
        diagnostic_hash = None
        learned_nrmse = None
    scenario_spec = (
        SyntheticTuningSpec.synthetic_osse(seed)
        if score_kind == "recovery"
        else SyntheticTuningSpec.synthetic_osse_null(seed)
    )
    return V2ScenarioOutcome(
        seed=seed,
        scenario="synthetic_osse" if score_kind == "recovery" else "synthetic_osse_null",
        policy=policy,
        score_kind=cast(Any, score_kind),
        evaluation_splits=evaluation_splits if score_kind == "recovery" else (),
        spec_sha256=spec_hash(scenario_spec),
        scenario_manifest=identity,
        fitting_config=identity,
        fitting_manifest=identity,
        evaluation_config=identity if score_kind == "recovery" else None,
        evaluation_manifest=identity if score_kind == "recovery" else None,
        report_sha256="b" * 64 if score_kind == "recovery" else json_sha256(score),
        diagnostic_report_sha256=diagnostic_hash,
        learned_basis_oracle_nrmse=learned_nrmse,
        score=score,
        evidence_passed=True,
        elapsed_seconds=1.0,
        process_peak_rss_bytes=1024,
        resources=resource_gate(1.0, 1024),
    )


def _recovery(
    tmp_path: Path,
    seed: int,
    policy: FableV2Policy,
    *,
    correlation: float = 0.95,
    nrmse: float = 0.2,
    evaluation_splits: tuple[str, ...] = ("calibration",),
) -> V2RecoverySeedResult:
    outcome = _outcome(
        tmp_path,
        seed,
        policy,
        nrmse=nrmse,
        evaluation_splits=evaluation_splits,
    )
    if correlation != 0.95:
        score = dict(outcome.score)
        score["metrics"] = {**score["metrics"], "field_correlation": correlation}
        outcome = replace(outcome, score=score)
    return V2RecoverySeedResult.from_outcome(outcome)


def _null(tmp_path: Path, seed: int, policy: FableV2Policy) -> V2NullSeedResult:
    return V2NullSeedResult.from_outcome(
        _outcome(tmp_path, seed, policy, score_kind="null", evaluation_splits=())
    )


def _preregistration(tmp_path: Path) -> tuple[V2Preregistration, Any]:
    config = tmp_path / "config.yaml"
    config.write_text("analysis: {}\n", encoding="ascii")
    current = V2Preregistration(
        generator_schema_version="test-schema",
        generator_spec_sha256="1" * 64,
        thresholds_sha256="2" * 64,
        code_sha256="3" * 64,
        environment_sha256="4" * 64,
        configs=(V2FileBinding.capture("test", config),),
        tests=(V2TestEvidence("suite", "pytest", "5" * 64),),
    )
    frozen = write_v2_preregistration(tmp_path / "preregistration.json", current)
    return current, frozen


def _calibration_record(tmp_path: Path, current: V2Preregistration, frozen: Any) -> Path:
    del current
    recovery_seeds = seed_roles()["calibration_recovery"]
    null_seeds = seed_roles()["calibration_null"]
    candidates = tuple(
        V2CalibrationCandidate(
            policy,
            tuple(
                _recovery(
                    tmp_path,
                    seed,
                    policy,
                    nrmse=0.20 + index * 0.02,
                )
                for seed in recovery_seeds
            ),
            tuple(_null(tmp_path, seed, policy) for seed in null_seeds),
        )
        for index, policy in enumerate(v2_calibration_policies())
    )
    record = select_v2_calibration_policy(
        candidates,
        frozen,
        _identity(tmp_path, "calibration-attempt.json"),
        (
            _identity(tmp_path, "calibration-recovery-lock.json"),
            _identity(tmp_path, "calibration-null-lock.json"),
        ),
    )
    return write_v2_calibration_record(tmp_path / "calibration.json", record)


def test_v2_renderer_uses_versioned_templates_and_updates_both_evaluators(
    tmp_path: Path,
) -> None:
    spec = SyntheticTuningSpec.synthetic_osse(seed_roles()["development"][0])
    policy = v2_calibration_policies()[1]

    fitting_path, evaluation_path = runner.render_v2_scenario_configs(
        tmp_path, spec, policy, evaluation_splits=("calibration",)
    )

    assert evaluation_path is not None
    fitting = yaml.safe_load(fitting_path.read_text(encoding="utf-8"))
    evaluation = yaml.safe_load(evaluation_path.read_text(encoding="utf-8"))
    expected = policy.normalized()
    expected.pop("policy_id")
    expected.pop("simplicity_rank")
    assert v2_fitting_policy_values(fitting) == expected
    assert fitting["analyses"]["model_daily"]["target_grid"] == 10.0
    assert fitting["analysis"]["end_time"] == "2008-12-31 23:59:59"
    assert "/oracle/" not in fitting_path.read_text(encoding="utf-8")
    assert evaluation["analyses"]["recovery"]["evaluation_splits"] == ["calibration"]
    assert evaluation["analyses"]["v2_diagnostics"]["evaluation_splits"] == ["calibration"]
    assert evaluation["analyses"]["v2_diagnostics"]["reported_common_factor_amplitude"] == 0.025


def test_one_scenario_executes_fit_and_evaluation_through_pipeline_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []
    seeds = seed_roles()["development"]
    lock = prepare_v2_generation(tmp_path / "campaign", "development", seeds)
    spec = SyntheticTuningSpec.synthetic_osse(seeds[0])
    bundle_root = lock.root / "bundles" / f"seed-{spec.master_seed}"
    (bundle_root / "inputs").mkdir(parents=True)
    (bundle_root / "oracle").mkdir()
    input_path = bundle_root / "inputs" / "placeholder.nc"
    oracle_path = bundle_root / "oracle" / "placeholder.nc"
    input_path.write_bytes(b"input")
    oracle_path.write_bytes(b"oracle")
    files = {}
    for relative, path in (
        ("inputs/placeholder.nc", input_path),
        ("oracle/placeholder.nc", oracle_path),
    ):
        files[relative] = {
            "byte_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "scientific_sha256": "a" * 64,
        }
    (bundle_root / "scenario.json").write_text(
        json.dumps(
            {
                "files": files,
                "root_seed": spec.master_seed,
                "scenario": spec.scenario,
                "spec_hash": spec_hash(spec),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="ascii",
    )

    class FakePipelineRunner:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def run_from_config(self, config: str) -> Any:
            path = Path(config)
            calls.append(path)
            root = Path(os.environ["FABLE_SYNTH"])
            if path.name.endswith("eval.yaml"):
                manifest = root / "evaluation" / "manifest.json"
                sources = {
                    "recovery": SimpleNamespace(data=xr.Dataset({"metric": xr.DataArray(1.0)}))
                }
            else:
                manifest = root / "output" / "manifest.json"
                sources = {}
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text('{"status":"completed"}\n', encoding="ascii")
            context = SimpleNamespace(sources=sources, metadata={})
            return SimpleNamespace(success=True, stage_errors={}, context=context)

    diagnostic = xr.Dataset({"learned_basis_oracle_nrmse": xr.DataArray(0.03)})
    diagnostic.attrs.update(diagnostic_only="true", eligible_for_calibration="false")
    report = xr.Dataset({"metric": xr.DataArray(1.0)})
    monkeypatch.setattr(runner, "PipelineRunner", FakePipelineRunner)
    monkeypatch.setattr(runner, "_close_result", lambda result: None)
    monkeypatch.setattr(
        runner,
        "_loaded_finalized_artifact",
        lambda result, analysis, role: diagnostic if analysis == "v2_diagnostics" else report,
    )
    monkeypatch.setattr(runner, "_recovery_artifact", lambda result: {})
    monkeypatch.setattr(runner, "evidence_gate", lambda fitting, evaluation: {"passed": True})
    monkeypatch.setattr(runner, "scientific_dataset_hash", lambda dataset: "f" * 64)
    monkeypatch.setattr(runner, "peak_rss_bytes", lambda: 1024)
    monkeypatch.setattr(runner, "resource_gate", lambda elapsed, rss: {"passed": True})
    monkeypatch.setattr(
        runner,
        "_v2_recovery_gate",
        lambda report: {
            "passed": True,
            "metrics": {},
            "diagnostics": {"estimate_vs_unfiltered_in_span_nrmse": 0.5},
        },
    )

    outcome = runner.run_v2_scenario(
        lock.root / "runs" / v2_calibration_policies()[0].policy_id / f"seed-{spec.master_seed}",
        spec,
        v2_calibration_policies()[0],
        score_kind="recovery",
        lock=lock,
    )

    assert [path.name for path in calls] == ["fable-v2-fit.yaml", "fable-v2-eval.yaml"]
    assert outcome.diagnostic_report_sha256 == "f" * 64
    assert outcome.learned_basis_oracle_nrmse == 0.03
    assert "best_representable_nrmse" not in outcome.score["diagnostics"]


def test_multi_seed_selection_hard_rejects_any_failed_seed_and_round_trips(
    tmp_path: Path,
) -> None:
    recovery_seeds = seed_roles()["calibration_recovery"]
    null_seeds = seed_roles()["calibration_null"]
    first, second = v2_calibration_policies()
    failed_recovery = tuple(
        _recovery(
            tmp_path,
            seed,
            first,
            correlation=0.89 if index == 0 else 0.99,
            nrmse=0.05,
        )
        for index, seed in enumerate(recovery_seeds)
    )
    passing_recovery = tuple(
        _recovery(tmp_path, seed, second, nrmse=0.25) for seed in recovery_seeds
    )
    first_null = tuple(_null(tmp_path, seed, first) for seed in null_seeds)
    second_null = tuple(_null(tmp_path, seed, second) for seed in null_seeds)
    from davinci_monet.tests.synthetic.fable_v2_freeze import (
        FrozenFileIdentity,
        FrozenPreregistration,
    )

    prereg = FrozenPreregistration(FrozenFileIdentity("/tmp/prereg", "d" * 64), "e" * 64)
    record = select_v2_calibration_policy(
        (
            V2CalibrationCandidate(first, failed_recovery, first_null),
            V2CalibrationCandidate(second, passing_recovery, second_null),
        ),
        prereg,
        FrozenFileIdentity("/tmp/attempt", "f" * 64),
        (
            FrozenFileIdentity("/tmp/recovery-lock", "1" * 64),
            FrozenFileIdentity("/tmp/null-lock", "2" * 64),
        ),
    )

    assert record.selected_policy_id == second.policy_id
    path = write_v2_calibration_record(tmp_path / "record.json", record)
    assert load_v2_calibration_record(path) == record
    with pytest.raises(FileExistsError, match="already exists"):
        write_v2_calibration_record(path, record)


def test_dry_runs_lock_fixed_roles_without_generating_acceptance_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current, frozen = _preregistration(tmp_path)
    calibration_path = _calibration_record(tmp_path, current, frozen)
    monkeypatch.setattr(
        calibration_runner_module,
        "eligible_v2_calibration_policies",
        lambda current: v2_calibration_policies(),
    )
    monkeypatch.setattr(lifecycle_module, "_validate_v2_generation_evidence", lambda *args: None)
    monkeypatch.setattr(
        preflight_module,
        "validate_v2_calibration_record",
        lambda record, current, frozen: record,
    )
    monkeypatch.setattr(
        acceptance_module,
        "validate_v2_calibration_record",
        lambda record, current, frozen: record,
    )
    monkeypatch.setattr(
        acceptance_module,
        "validate_v2_preflight_record",
        lambda record, current, frozen, calibration: record,
    )

    def forbidden(*args: object, **kwargs: object) -> Any:
        del args, kwargs
        raise AssertionError("dry-run attempted synthetic generation")

    calibration_plan = run_v2_calibration(
        tmp_path / "calibration-dry",
        tmp_path / "unused.json",
        current,
        frozen,
        dry_run=True,
        scenario_runner=forbidden,
    )
    assert json.loads(calibration_plan.read_text(encoding="ascii"))["seed_roles"] == {
        "calibration_null": list(seed_roles()["calibration_null"]),
        "calibration_recovery": list(seed_roles()["calibration_recovery"]),
    }

    preflight_plan = run_v2_preflight(
        tmp_path / "preflight-dry",
        tmp_path / "unused-preflight.json",
        current,
        frozen,
        calibration_path,
        dry_run=True,
        scenario_runner=forbidden,
    )
    assert (
        json.loads(preflight_plan.read_text(encoding="ascii"))["seed"]
        == seed_roles()["preflight"][0]
    )

    calibration_identity = FrozenFileIdentity.capture(calibration_path)
    policy = load_v2_calibration_record(calibration_path).selected_policy
    result = _recovery(
        tmp_path,
        seed_roles()["preflight"][0],
        policy,
        evaluation_splits=("development_test",),
    )
    preflight_record = V2PreflightRecord(
        frozen,
        _identity(tmp_path, "preflight-attempt.json"),
        _identity(tmp_path, "preflight-lock.json"),
        calibration_identity,
        policy,
        result,
    )
    preflight_path = write_v2_preflight_record(tmp_path / "preflight.json", preflight_record)

    acceptance_plan = run_v2_acceptance(
        tmp_path / "acceptance-dry",
        current,
        frozen,
        calibration_path,
        preflight_path,
        dry_run=True,
        scenario_runner=forbidden,
    )
    assert json.loads(acceptance_plan.read_text(encoding="ascii"))["seeds"] == list(
        ACCEPTANCE_SEEDS
    )


def test_versioned_lifecycle_scripts_expose_no_seed_override() -> None:
    root = Path(__file__).parents[4] / "analyses" / "aerosol-tuning" / "scripts"
    names = (
        "calibrate_v2_synthetic.py",
        "run_v2_preflight.py",
        "run_v2_acceptance.py",
    )
    for name in names:
        text = (root / name).read_text(encoding="utf-8")
        assert "--seed" not in text
