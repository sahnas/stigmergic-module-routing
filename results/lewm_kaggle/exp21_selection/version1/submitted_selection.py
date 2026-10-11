"""Experiment 21: change only CEM's final returned plan, with frozen models.

The upstream solver still generates and scores every candidate. A passive
callback captures its last elite set; the intervention replaces only the
returned actions. No model fit, search clipping or objective is changed.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time

CODE_REV = "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"
EVAL_SHA = "6584f009b40550fa0df2f8dee22b9329fbfacdfaf74464aa4fd105f464795245"
SOURCE_HASHES = {
    "solver/cem.py": "a6662e7f1f2c6ecaf0a8527255d531aa05ed583339aabdf2798d0206382ef51b",
    "policy.py": "4967e7e3d5b20eb7a1d0b00e5d60fd701cce1c750ae7ec4a9b02529b9366db22",
    "envs/two_room/env.py": "5e1d392de5b02472062dbe872aded67fd465fcc8f7eaa1c02a753b2fc31c61f0",
    "envs/pusht/env.py": "3c4df51ed01ed034386c24961caa638f02b5c0e31c88fec727a9f1cd5808e203",
}
WEIGHT_HASHES = {
    "tworoom": {
        "released": "566f223624ea4bfb39dbfe6ae731198dd6ea73b7b8919fed6b1ecafca810f7dd",
        "retrained": "df94cb98551b9ed0b5ca54fb7477b9fd305faf277261fac04f8ba321f440a132",
        "bilinear": "611f430358576046c1b6cbdbd52e94b9e6577e28872b9c02812f64c5d857d23f",
    },
    "pusht": {
        "released": "48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607",
        "retrained": "9224779215c068aa4b7c68260fb0380e4e68958e347ec8d0cb08d1a6a55e5f23",
        "bilinear": "fa06ecffd4f73ccd8ffa64e851dac38e9cf5f2f042b72c5998c4dc9a7ea8283e",
    },
}
CONFIG_HASHES = {
    "tworoom_released": "2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09",
    "tworoom_retrained": "4d7e8cf837ec9898dfb638bb58789901ac476e8b550c51bd9a6da6e9455af91e",
}
TASKS = ("tworoom", "pusht")
MODELS = ("released", "retrained", "bilinear")
SELECTIONS = ("mean", "best")
SEEDS = (43, 44, 45)
GATE_N = 8
GATE_THRESHOLD = 1e-3
OUT = Path("/kaggle/working")
TMP = Path("/kaggle/tmp/exp21")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def log(event, **fields):
    item = dict(event=event, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields)
    line = json.dumps(item, allow_nan=False)
    print(line, flush=True)
    with (OUT / "progress.jsonl").open("a") as f:
        f.write(line + "\n")


def command(args):
    print("$ " + " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True)


def unique_matching(suffix, expected=None):
    matches = sorted(Path("/kaggle/input").rglob(suffix))
    if expected is not None:
        matches = [p for p in matches if sha256(p) == expected]
    assert matches, f"Missing or changed frozen input: {suffix}"
    if len(matches) > 1:
        assert len({sha256(p) for p in matches}) == 1, f"Ambiguous input: {suffix}"
    return matches[0]


def make_bilinear_predictor(weights):
    import torch

    class BilinearPredictor(torch.nn.Module):
        num_frames = 3

        def __init__(self):
            super().__init__()
            self.register_buffer("W", weights)

        def forward(self, emb, act):
            batch, steps, _ = emb.shape
            predictions = []
            for t in range(steps):
                z0 = emb[:, t]
                z1 = emb[:, t - 1] if t >= 1 else z0
                z2 = emb[:, t - 2] if t >= 2 else z1
                a = act[:, t]
                x = torch.cat([z0, z1, z2, a,
                               (z0.unsqueeze(2) * a.unsqueeze(1)).reshape(batch, -1),
                               torch.ones(batch, 1, device=emb.device, dtype=emb.dtype)], 1)
                predictions.append(x @ self.W.to(emb.dtype))
            return torch.stack(predictions, 1)

    return BilinearPredictor()


class FinalElites:
    """Duck-typed passive callback for stable-worldmodel 0.1.1."""
    output_key = "exp21_final_elites"

    def __init__(self, n_steps):
        self.n_steps = n_steps
        self.history = []
        self.reset()

    def reset(self):
        self.means, self.best, self.elites, self.costs = [], [], [], []
        self.history = []

    def start_batch(self):
        pass

    def end_solve(self):
        pass

    def __call__(self, *, step, topk_candidates, topk_vals, mean, **unused):
        if step != self.n_steps - 1:
            return
        import torch
        index = topk_vals.argmin(1)
        chosen = topk_candidates[torch.arange(len(index), device=index.device), index]
        self.means.append(mean.detach().cpu().clone())
        self.best.append(chosen.detach().cpu().clone())
        self.elites.append(topk_candidates.detach().cpu().clone())
        self.costs.append(topk_vals.detach().cpu().clone())


def attach_selection(solver, selection, inspect=None):
    """Call the original solve, then replace only output['actions'] in best mode."""
    import torch
    assert selection in SELECTIONS
    recorder = FinalElites(solver.n_steps)
    solver.callbacks.append(recorder)
    original = solver.solve

    def solve(info_dict, init_action=None):
        output = original(info_dict, init_action=init_action)
        mean = output["actions"]
        captured_mean = torch.cat(recorder.means)
        assert torch.equal(mean, captured_mean), "Original output is not the captured final mean"
        best = torch.cat(recorder.best)
        if inspect is not None:
            inspect(info_dict, mean, best, torch.cat(recorder.elites), torch.cat(recorder.costs))
        if selection == "best":
            output["actions"] = best
        return output

    solver.solve = solve
    return recorder


def effective_plan(task, normalized, scaler):
    """Map a normalized plan to the controls actually applied by the pinned env."""
    import numpy as np
    shape = normalized.shape
    physical = scaler.inverse_transform(normalized.reshape(-1, 2)).reshape(shape)
    effective = np.clip(physical, -1.0, 1.0) if task == "tworoom" else physical.copy()
    re_normalized = scaler.transform(effective.reshape(-1, 2)).reshape(shape)
    return physical, effective, re_normalized


def expanded_info(info, n, device):
    import numpy as np
    import torch
    result = {}
    for key, value in info.items():
        if torch.is_tensor(value):
            result[key] = value.to(device).unsqueeze(1).expand(len(value), n, *value.shape[1:])
        elif isinstance(value, np.ndarray):
            result[key] = np.repeat(value[:, None], n, axis=1)
    return result


def runtime_smoke():
    """Check the hook against the actually installed solver, before research data."""
    import numpy as np
    import torch
    import stable_worldmodel as swm
    from gymnasium.spaces import Box
    from types import SimpleNamespace

    class Cost(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.candidates = []

        def get_cost(self, info, action):
            self.candidates.append(action.detach().clone())
            return ((action - .37) ** 2).sum((-1, -2))

    solvers, costs = [], []
    for selection in (None, "mean", "best"):
        cost = Cost()
        solver = swm.solver.CEMSolver(cost, batch_size=1, num_samples=300, n_steps=3, topk=30, seed=21)
        solver.configure(action_space=Box(-1, 1, shape=(2, 2), dtype=np.float32), n_envs=2,
                         config=SimpleNamespace(horizon=5, action_block=5))
        if selection is not None:
            attach_selection(solver, selection)
        solvers.append(solver)
        costs.append(cost)
    outputs = [s.solve({"pixels": torch.zeros(2, 1, 1)}) for s in solvers]
    assert torch.equal(outputs[0]["actions"], outputs[1]["actions"])
    assert all(torch.equal(x, y) and torch.equal(x, z)
               for x, y, z in zip(*(c.candidates for c in costs)))
    assert len({len(c.candidates) for c in costs}) == 1
    assert all(torch.equal(solvers[0].torch_gen.get_state(), s.torch_gen.get_state()) for s in solvers[1:])
    save_json(OUT / "runtime_smoke.json", dict(pass_=True, torch=torch.__version__,
              mean_matches_original_exactly=True, candidates_and_solver_rng_identical=True))


def worker(task, model_name, selection, seed, phase):
    import numpy as np
    import torch
    import hdf5plugin  # register HDF5 filters before the evaluator imports data
    import stable_worldmodel as swm

    assert task in TASKS and model_name in MODELS and selection in SELECTIONS
    assert (phase == "gate" and seed == 42 and selection == "mean") or (phase == "evaluation" and seed in SEEDS)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inputs = json.loads((OUT / "input_manifest.json").read_text())[task]
    run_dir = OUT / phase / task / f"{model_name}_{selection}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    original_load = swm.wm.utils.load_pretrained

    def load_selected(name, *args, **kwargs):
        assert name == model_name
        folder = inputs["retrained"] if name == "retrained" else inputs["released"]
        model = original_load(folder, *args, **kwargs)
        if name == "bilinear":
            saved = torch.load(inputs["bilinear"], map_location="cpu", weights_only=True)
            assert tuple(saved["W"].shape) == (2507, 192)
            model.predictor = make_bilinear_predictor(saved["W"])
            model.action_encoder = torch.nn.Identity()
            model.pred_proj = torch.nn.Identity()
        return model

    swm.wm.utils.load_pretrained = load_selected
    original_set_policy = swm.World.set_policy

    def set_policy(world, policy):
        solver = policy.solver
        assert solver.__class__.__name__ == "CEMSolver"
        assert (solver.batch_size, solver.num_samples, solver.n_steps, solver.topk, solver.var_scale) == (1, 300, 30, 30, 1.0)
        assert (policy.cfg.horizon, policy.cfg.receding_horizon, policy.cfg.action_block) == (5, 5, 5)
        scaler = policy.process["action"]
        calls = []

        def inspect(info, mean, best, elites, elite_costs):
            mean_np, best_np = mean.numpy(), best.numpy()
            physical_mean, effective_mean, norm_effective_mean = effective_plan(task, mean_np, scaler)
            physical_best, effective_best, norm_effective_best = effective_plan(task, best_np, scaler)
            candidates = np.stack([mean_np, best_np, norm_effective_mean, norm_effective_best], 1)
            local_info = expanded_info(info, 4, solver.device)
            # This is logging only: four extra predictions shared by both modes,
            # after the complete 9,000-candidate search and with a fresh info dict.
            generator_state = solver.torch_gen.get_state().clone()
            with torch.inference_mode():
                cost = solver.model.get_cost(local_info, torch.as_tensor(candidates, device=solver.device, dtype=solver.dtype)).detach().cpu().numpy()
            assert torch.equal(generator_state, solver.torch_gen.get_state())
            index = len(calls)
            calls.append(dict(index=index, rows=len(mean), real_context=int(info["pixels"].shape[1]),
                              effective_plan_max_abs_delta_normalized=np.max(np.abs(norm_effective_mean - norm_effective_best), axis=(1, 2)).tolist(),
                              cost_columns=["mean_unbounded", "best_unbounded", "mean_effective", "best_effective"],
                              predicted_costs=cost.tolist(), search_candidates_per_row=9000,
                              extra_diagnostic_candidates_per_row=4))
            np.savez_compressed(run_dir / f"plans_{index:03d}.npz", mean=mean_np, best=best_np,
                                physical_mean=physical_mean, physical_best=physical_best,
                                effective_mean=effective_mean, effective_best=effective_best,
                                elites=elites.numpy(), elite_costs=elite_costs.numpy(),
                                predicted_costs=cost)
            save_json(run_dir / "plan_diagnostics.json", calls)

        attach_selection(solver, selection, inspect)
        # Capture the controls submitted to the pinned environment, including
        # TwoRoom's internal clamp. PushT has no action clamp in this version.
        for env_index, wrapped_env in enumerate(world.envs.envs):
            original_step = wrapped_env.step

            def step(action, original_step=original_step, env_index=env_index):
                submitted = np.asarray(action, dtype=np.float32)
                effective = np.clip(submitted, -1, 1) if task == "tworoom" else submitted.copy()
                with (run_dir / "executed_controls.jsonl").open("a") as f:
                    f.write(json.dumps(dict(env_index=env_index, submitted=submitted.tolist(),
                                            effective=effective.tolist()), allow_nan=False) + "\n")
                return original_step(action)

            wrapped_env.step = step
        save_json(run_dir / "solver_and_normalization.json", dict(
            action_mean=scaler.mean_.tolist(), action_scale=scaler.scale_.tolist(),
            task=task, model=model_name, selection=selection, seed=seed,
            horizon=5, receding_horizon=5, action_block=5, num_samples=300,
            iterations=30, elites=30, batch_size=1, search_candidates_per_decision=9000,
            diagnostic_candidates_per_decision=4, execution_mapping="clamp [-1,1]" if task == "tworoom" else "identity"))
        return original_set_policy(world, policy)

    swm.World.set_policy = set_policy
    original_evaluate = swm.World.evaluate

    def evaluate(world, *args, **kwargs):
        assert not args and kwargs["goal_offset"] == 25 and kwargs["eval_budget"] == 50
        all_tasks = [dict(episode_id=int(e), start_step=int(s), goal_step=int(s) + 25)
                     for e, s in zip(kwargs["episodes_idx"], kwargs["start_steps"])]
        assert len(all_tasks) == 100
        reference = OUT / "tasks" / f"{task}_seed{seed}.json"
        if reference.exists():
            assert json.loads(reference.read_text()) == all_tasks
        else:
            save_json(reference, all_tasks)
        if phase == "gate":
            kwargs["episodes_idx"] = kwargs["episodes_idx"][:GATE_N]
            kwargs["start_steps"] = kwargs["start_steps"][:GATE_N]
        tasks = all_tasks[:GATE_N] if phase == "gate" else all_tasks
        save_json(run_dir / "tasks.json", tasks)
        kwargs["video"] = run_dir / "videos"
        start = time.monotonic()
        log("evaluation_started", task=task, model=model_name, selection=selection, seed=seed, phase=phase, episodes=len(tasks))
        result = original_evaluate(world, **kwargs)
        successes = np.asarray(result["episode_successes"], dtype=bool)
        assert successes.shape == (len(tasks),)
        assert abs(float(result["success_rate"]) - 100 * successes.mean()) < 1e-8
        rows = [dict(**t, success=bool(success)) for t, success in zip(tasks, successes)]
        save_json(run_dir / "episodes.json", rows)
        save_json(run_dir / "metrics.json", dict(successes=int(successes.sum()), episodes=len(tasks),
                                                  elapsed_seconds=time.monotonic() - start,
                                                  metrics={k: v.tolist() if hasattr(v, "tolist") else v for k, v in result.items()}))
        log("evaluation_complete", task=task, model=model_name, selection=selection, seed=seed, phase=phase,
            successes=int(successes.sum()), episodes=len(tasks), seconds=time.monotonic() - start)
        return result

    swm.World.evaluate = evaluate
    sys.argv = [str(TMP / "le-wm/eval.py"), f"--config-name={task}", f"policy={model_name}",
                f"seed={seed}", "eval.num_eval=100", f"output.filename={run_dir / 'official_results.txt'}"]
    if phase == "gate":
        sys.argv.append(f"world.num_envs={GATE_N}")
    save_json(run_dir / "command.json", sys.argv)
    runpy.run_path(str(TMP / "le-wm/eval.py"), run_name="__main__")


def paired(a, b):
    key = lambda row: (row["episode_id"], row["start_step"], row["goal_step"])
    assert len(a) == len(b) and [key(x) for x in a] == [key(x) for x in b]
    wins = sum(x["success"] and not y["success"] for x, y in zip(a, b))
    losses = sum(y["success"] and not x["success"] for x, y in zip(a, b))
    n = wins + losses
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n) if n else 1.0
    return dict(a_only=wins, b_only=losses, difference_percentage_points=100 * (wins - losses) / len(a),
                mcnemar_exact_two_sided_p=p)


def analyze(root):
    """Deterministic analysis; also usable on a retrieved result directory."""
    import numpy as np
    root = Path(root)
    results = {}
    for task in TASKS:
        per_seed = {}
        for seed in SEEDS:
            def rows(model, selection):
                return json.loads((root / "evaluation" / task / f"{model}_{selection}_seed{seed}" / "episodes.json").read_text())
            all_rows = {(m, s): rows(m, s) for m in MODELS for s in SELECTIONS}
            baseline_gap = paired(all_rows["bilinear", "mean"], all_rows["retrained", "mean"])["difference_percentage_points"]
            selected_gap = paired(all_rows["bilinear", "best"], all_rows["retrained", "best"])["difference_percentage_points"]
            interaction = baseline_gap - selected_gap
            # Keep all four outcomes for each starting task together.
            d = np.array([int(n1["success"]) - int(n0["success"]) - int(b1["success"]) + int(b0["success"])
                          for n1, n0, b1, b0 in zip(all_rows["retrained", "best"], all_rows["retrained", "mean"],
                                                    all_rows["bilinear", "best"], all_rows["bilinear", "mean"])])
            assert abs(100 * d.mean() - interaction) < 1e-8
            rng = np.random.default_rng(21000 + seed)
            bootstrap = 100 * d[rng.integers(0, len(d), size=(20000, len(d)))].mean(1)
            ci = np.quantile(bootstrap, [0.025, 0.975]).tolist()
            per_seed[str(seed)] = dict(
                counts={m: {s: sum(x["success"] for x in all_rows[m, s]) for s in SELECTIONS} for m in MODELS},
                baseline_bilinear_minus_corrected_pp=baseline_gap,
                best_bilinear_minus_corrected_pp=selected_gap,
                reduction_of_bilinear_advantage_pp=interaction,
                interaction_paired_bootstrap_95_ci=ci,
                selection_effects={m: paired(all_rows[m, "best"], all_rows[m, "mean"]) for m in MODELS},
                bilinear_comparisons={s: {m: paired(all_rows["bilinear", s], all_rows[m, s])
                                         for m in ("released", "retrained")} for s in SELECTIONS})
        results[task] = per_seed
    tr = list(results["tworoom"].values())
    reproduced = all(v["baseline_bilinear_minus_corrected_pp"] >= 5 for v in tr)
    criterion = reproduced and all(v["reduction_of_bilinear_advantage_pp"] >= 5 for v in tr)
    return dict(results=results, baseline_gap_reproduced=reproduced,
                primary_material_selection_interaction=criterion,
                inference="baseline_not_reproduced" if not reproduced else "criterion_met" if criterion else "criterion_not_met",
                statistical_scope="Fixed checkpoints, paired evaluation tasks; CIs per seed are descriptive. No independence assumption across seeds or training-seed generalization.")


def bootstrap():
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    os.environ.update(STABLEWM_HOME=str(TMP / "stablewm"), HF_HOME=str(TMP / "hf"),
                      WANDB_MODE="disabled", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
                      SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy", PYTHONUNBUFFERED="1")
    command(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    command(["apt-get", "install", "-y", "-qq", "zstd", "libegl1", "libgl1", "ffmpeg"])
    command([sys.executable, "-m", "pip", "install", "-q", "stable-worldmodel[train,format]==0.1.1",
             "stable-pretraining==0.1.8", "transformers==4.57.6", "hydra-core==1.3.2",
             "pygame==2.6.1", "pymunk==7.2.0", "shapely==2.1.2", "opencv-python-headless==4.13.0.92"])
    (OUT / "environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    upstream = TMP / "le-wm"
    command(["git", "clone", "--quiet", "--no-checkout", "https://github.com/lucas-maes/le-wm.git", upstream])
    command(["git", "-C", upstream, "checkout", "--quiet", "--detach", CODE_REV])
    assert sha256(upstream / "eval.py") == EVAL_SHA
    import stable_worldmodel as swm
    import torch
    package_root = Path(swm.__file__).parent
    for name, expected in SOURCE_HASHES.items():
        assert sha256(package_root / name) == expected, f"Changed pinned source: {name}"
    runtime_smoke()
    shutil.copytree(upstream / "config/eval", OUT / "official_eval_config")
    shutil.copyfile(upstream / "eval.py", OUT / "official_eval.py")
    for name in SOURCE_HASHES:
        target = OUT / "official_library" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package_root / name, target)
    manifest = {}
    for task in TASKS:
        if task == "tworoom":
            rel = unique_matching("models--quentinll--lewm-tworooms/weights.pt", WEIGHT_HASHES[task]["released"])
            rt = unique_matching("tworoom_retrained_v2/weights.pt", WEIGHT_HASHES[task]["retrained"])
            bil = unique_matching("koopman_bilinear.pt", WEIGHT_HASHES[task]["bilinear"])
            assert sha256(rel.parent / "config.json") == CONFIG_HASHES["tworoom_released"]
            assert sha256(rt.parent / "config.json") == CONFIG_HASHES["tworoom_retrained"]
        else:
            rel = unique_matching("pusht_released/weights.pt", WEIGHT_HASHES[task]["released"])
            rt = unique_matching("pusht_retrained_v2/weights.pt", WEIGHT_HASHES[task]["retrained"])
            bil = unique_matching("pusht_bilinear.pt", WEIGHT_HASHES[task]["bilinear"])
            assert (rel.parent / "config.json").read_bytes() == (rt.parent / "config.json").read_bytes()
        a = torch.load(rel, map_location="cpu", weights_only=True)
        b = torch.load(rt, map_location="cpu", weights_only=True)
        keys = [k for k in a if k.startswith(("encoder.", "projector."))]
        assert keys and all(k in b and torch.equal(a[k], b[k]) for k in keys), "Encoder/projector differs"
        del a, b
        manifest[task] = dict(released=str(rel.parent), retrained=str(rt.parent), bilinear=str(bil),
                              hashes=WEIGHT_HASHES[task], same_encoder_and_projector=True,
                              config_hashes={"released": sha256(rel.parent / "config.json"),
                                             "retrained": sha256(rt.parent / "config.json")})
    save_json(OUT / "input_manifest.json", manifest)
    datasets = TMP / "stablewm/datasets"
    datasets.mkdir(parents=True, exist_ok=True)
    tr = unique_matching("tworoom.h5")
    (datasets / "tworoom.h5").symlink_to(tr)
    from huggingface_hub import hf_hub_download
    archive = hf_hub_download("quentinll/lewm-pusht", "pusht_expert_train.h5.zst", repo_type="dataset",
                              revision="655cd446b992", local_dir=TMP / "download")
    command(["zstd", "-d", "-q", archive, "-o", datasets / "pusht_expert_train.h5"])
    Path(archive).unlink()
    save_json(OUT / "run_manifest.json", dict(script_sha256=sha256(__file__), upstream_revision=CODE_REV,
              official_eval_sha256=EVAL_SHA, library_source_hashes=SOURCE_HASHES, input_hashes=WEIGHT_HASHES,
              gate_seed=42, gate_episodes=GATE_N, gate_threshold_normalized=GATE_THRESHOLD,
              evaluation_seeds=list(SEEDS), episodes_per_condition=100, planned_evaluations=36,
              solver_budget_per_decision=9000, diagnostic_candidates_per_decision=4,
              selection_intervention="Only output['actions'] after the last original CEM iteration",
              dataset_bytes={p.name: p.stat().st_size for p in datasets.glob("*.h5")},
              dataset_sha256={p.name: sha256(p) for p in datasets.glob("*.h5")},
              execution_mapping={"tworoom": "internal clamp [-1,1]", "pusht": "identity, no added clamp"}))


def main():
    bootstrap()
    # The gate uses only the first returned plan, before paths diverge; it
    # cannot establish that the selection intervention improves success.
    gates = {}
    for task in TASKS:
        gates[task] = {}
        for model in MODELS:
            command([sys.executable, __file__, "--worker", task, model, "mean", "42", "gate"])
            diagnostics = json.loads((OUT / "gate" / task / f"{model}_mean_seed42/plan_diagnostics.json").read_text())
            delta = diagnostics[0]["effective_plan_max_abs_delta_normalized"]
            assert len(delta) == GATE_N
            gates[task][model] = dict(first_plan_deltas=delta, differing_starts=sum(d > GATE_THRESHOLD for d in delta))
    proceed = gates["tworoom"]["retrained"]["differing_starts"] >= 2
    save_json(OUT / "gate.json", dict(cells=gates, proceed=proceed,
              rule="TwoRoom corrected network: effective plans differ by >1e-3 normalized units on at least 2 of 8 fixed starts"))
    log("gate_complete", proceed=proceed, cells=gates)
    if not proceed:
        save_json(OUT / "complete.json", dict(complete=True, status="stopped_at_feasibility_gate", evaluations=0,
                  inference="No full evaluation; this small gate is not evidence of universal plan equivalence."))
        return
    for task in TASKS:
        for seed in SEEDS:
            for model in MODELS:
                for selection in SELECTIONS:
                    command([sys.executable, __file__, "--worker", task, model, selection, str(seed), "evaluation"])
    result = analyze(OUT)
    save_json(OUT / "analysis.json", result)
    save_json(OUT / "complete.json", dict(complete=True, status="evaluated", evaluations=36,
                                          inference=result["inference"], primary=result["primary_material_selection_interaction"]))
    log("complete", inference=result["inference"], primary=result["primary_material_selection_interaction"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=5, metavar=("TASK", "MODEL", "SELECTION", "SEED", "PHASE"))
    parser.add_argument("--analyze")
    args = parser.parse_args()
    try:
        if args.worker:
            task, model, selection, seed, phase = args.worker
            worker(task, model, selection, int(seed), phase)
        elif args.analyze:
            print(json.dumps(analyze(args.analyze), indent=2))
        else:
            main()
    except BaseException as error:
        if not args.analyze:
            save_json(OUT / "failure.json", dict(type=type(error).__name__, message=str(error),
                                                 worker=args.worker, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        raise
