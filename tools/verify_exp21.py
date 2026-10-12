"""Audit experiment 21's downloaded outcomes, plans and executed controls.

Usage: python tools/verify_exp21.py [output-directory] [--write]
No model inference or new research episode is executed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "results/lewm_kaggle/exp21_selection/version2"
SOURCE = ROOT / "kaggle/lewm-cem-selection/selection.py"
SOURCE_SHA = "c68e26ed084d1467c8b448a6fdeebe59190fddb756ac552dffea5b15701997ed"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=2e-6, atol=2e-6)


def verify(data):
    assert sha(SOURCE) == SOURCE_SHA
    code = runpy.run_path(str(SOURCE))
    manifest, inputs = read(data / "run_manifest.json"), read(data / "input_manifest.json")
    assert manifest["script_sha256"] == SOURCE_SHA
    assert manifest["upstream_revision"] == code["CODE_REV"]
    assert manifest["input_hashes"] == code["WEIGHT_HASHES"]
    assert sha(data / "official_eval.py") == code["EVAL_SHA"]
    for name, digest in code["SOURCE_HASHES"].items():
        assert sha(data / "official_library" / name) == digest
    for task in code["TASKS"]:
        assert inputs[task]["hashes"] == code["WEIGHT_HASHES"][task]
        assert inputs[task]["same_encoder_and_projector"]
    for item in read(data / "download_manifest.json")["files"]:
        path = data / item["path"]
        assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"]
    assert not (data / "failure.json").exists()
    smoke = read(data / "runtime_smoke.json")
    assert smoke["pass_"] and all(smoke["diagnostic_cost_path"].values())
    assert smoke["mean_matches_original_exactly"] and smoke["candidates_and_solver_rng_identical"]
    complete, gate = read(data / "complete.json"), read(data / "gate.json")
    assert complete["complete"] and complete["status"] == "evaluated" and complete["evaluations"] == 36
    assert gate["proceed"]

    identity = lambda row: (row["episode_id"], row["start_step"], row["goal_step"])
    normalization, runs, first_plans = {}, [], {}
    for phase in ("gate", "evaluation"):
        for task in code["TASKS"]:
            seeds = (42,) if phase == "gate" else code["SEEDS"]
            selections = ("mean",) if phase == "gate" else code["SELECTIONS"]
            for seed in seeds:
                reference = read(data / "tasks" / f"{task}_seed{seed}.json")
                assert len(reference) == 100
                tasks = reference[:8] if phase == "gate" else reference
                n = len(tasks)
                for model in code["MODELS"]:
                    for selection in selections:
                        folder = data / phase / task / f"{model}_{selection}_seed{seed}"
                        episodes, metrics = read(folder / "episodes.json"), read(folder / "metrics.json")
                        assert len(episodes) == n and read(folder / "tasks.json") == tasks
                        assert [identity(r) for r in episodes] == [identity(r) for r in tasks]
                        assert all(type(r["success"]) is bool and r["goal_step"] == r["start_step"] + 25 for r in episodes)
                        successes = [r["success"] for r in episodes]
                        assert metrics["metrics"]["episode_successes"] == successes
                        assert metrics["successes"] == sum(successes) and metrics["episodes"] == n
                        assert abs(metrics["metrics"]["success_rate"] - 100 * sum(successes) / n) < 1e-9
                        config = read(folder / "solver_and_normalization.json")
                        for key, value in dict(task=task, model=model, selection=selection, seed=seed,
                                               horizon=5, receding_horizon=5, action_block=5, num_samples=300,
                                               iterations=30, elites=30, batch_size=1,
                                               search_candidates_per_decision=9000, diagnostic_candidates_per_decision=4).items():
                            assert config[key] == value
                        stats = (config["action_mean"], config["action_scale"])
                        assert normalization.setdefault(task, stats) == stats
                        offset, scale = np.asarray(stats[0]), np.asarray(stats[1])
                        controls = [[] for _ in range(n)]
                        for line in (folder / "executed_controls.jsonl").read_text().splitlines():
                            record = json.loads(line)
                            assert 0 <= record["env_index"] < n
                            submitted, effective = np.asarray(record["submitted"]), np.asarray(record["effective"])
                            assert submitted.shape == effective.shape == (2,) and np.isfinite(submitted).all()
                            close(effective, np.clip(submitted, -1, 1) if task == "tworoom" else submitted)
                            controls[record["env_index"]].append(submitted)
                        assert all(0 < len(c) <= 50 for c in controls)
                        diagnostics = read(folder / "plan_diagnostics.json")
                        paths = sorted(folder.glob("plans_*.npz"))
                        assert len(paths) == len(diagnostics) == max((len(c) + 24) // 25 for c in controls)
                        planned_rows = 0
                        for index, path in enumerate(paths):
                            with np.load(path, allow_pickle=False) as file:
                                arrays = {key: file[key] for key in file.files}
                            assert all(np.isfinite(value).all() for value in arrays.values())
                            mean, best, elites, costs = (arrays[k] for k in ("mean", "best", "elites", "elite_costs"))
                            active = [i for i, c in enumerate(controls) if len(c) > 25 * index]
                            b = len(active)
                            assert mean.shape == best.shape == (b, 5, 10)
                            assert elites.shape == (b, 30, 5, 10) and costs.shape == (b, 30)
                            assert np.array_equal(best, elites[np.arange(b), costs.argmin(1)])
                            close(mean, elites.mean(1))
                            normalized_effective = []
                            for name in ("mean", "best"):
                                normalized = arrays[name].reshape(b, 25, 2)
                                physical = arrays["physical_" + name].reshape(b, 25, 2)
                                effective = arrays["effective_" + name].reshape(b, 25, 2)
                                close(physical, normalized * scale + offset)
                                close(effective, np.clip(physical, -1, 1) if task == "tworoom" else physical)
                                normalized_effective.append((effective - offset) / scale)
                            diagnostic = diagnostics[index]
                            assert diagnostic["index"] == index and diagnostic["rows"] == b
                            assert diagnostic["search_candidates_per_row"] == 9000 and diagnostic["extra_diagnostic_candidates_per_row"] == 4
                            np.testing.assert_array_equal(arrays["predicted_costs"], diagnostic["predicted_costs"])
                            delta = np.abs(normalized_effective[0] - normalized_effective[1]).max((1, 2))
                            close(delta, diagnostic["effective_plan_max_abs_delta_normalized"])
                            selected = arrays["physical_" + selection].reshape(b, 25, 2)
                            for row, env in enumerate(active):
                                executed = np.asarray(controls[env][25 * index:25 * (index + 1)])
                                close(executed, selected[row, :len(executed)])
                            planned_rows += b
                            if index == 0:
                                first_plans[phase, task, seed, model, selection] = arrays
                                if phase == "gate":
                                    cell = gate["cells"][task][model]
                                    close(delta, cell["first_plan_deltas"])
                                    assert cell["differing_starts"] == sum(delta > code["GATE_THRESHOLD"])
                        runs.append(dict(phase=phase, task=task, seed=seed, model=model, selection=selection,
                                         successes=sum(successes), episodes=n, solve_calls=len(paths), planned_rows=planned_rows,
                                         search_candidates=9000 * planned_rows, diagnostic_candidates=4 * planned_rows,
                                         executed_control_steps=sum(map(len, controls)), elapsed_seconds=metrics["elapsed_seconds"]))
    assert gate["cells"]["tworoom"]["retrained"]["differing_starts"] >= 2
    for task in code["TASKS"]:
        for seed in code["SEEDS"]:
            for model in code["MODELS"]:
                a, b = [first_plans["evaluation", task, seed, model, selection] for selection in ("mean", "best")]
                for name in ("mean", "best", "elites", "elite_costs"):
                    assert np.array_equal(a[name], b[name]), ("Initial search differs", task, seed, model, name)
    recalculated = code["analyze"](data)
    assert recalculated == read(data / "analysis.json")
    assert complete["inference"] == recalculated["inference"] and complete["primary"] == recalculated["primary_material_selection_interaction"]
    assert len(runs) == 42 and sum(r["episodes"] for r in runs if r["phase"] == "evaluation") == 3600
    return dict(pass_=True, script_sha256=SOURCE_SHA, numpy=np.__version__, completed_gate_runs=6, completed_evaluation_runs=36,
                evaluated_outcomes=3600, first_search_elites_and_costs_exactly_equal_across_selectors=True,
                all_selected_plans_and_executed_controls_verified=True, paired_analysis_reproduced=True,
                gate_differing_starts=gate["cells"]["tworoom"]["retrained"]["differing_starts"],
                inference=recalculated["inference"], runs=runs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=DEFAULT)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = verify(args.directory)
    if args.write:
        (args.directory / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2))
