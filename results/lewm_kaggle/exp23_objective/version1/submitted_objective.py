"""Experiment 23: a shared decoded-position terminal cost on TwoRoom.

Single-file Kaggle entry point. The unchanged experiment-21 helpers are loaded
from their frozen Git commit and verified before import. No research data is
used by --check; the only fitted parameters are a shared affine position probe.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time
from types import MethodType
import urllib.request

REPO_REV = "944e7ef3a61952c0ba01b5406921180fc9a3dc3f"
BASE_SHA = "c68e26ed084d1467c8b448a6fdeebe59190fddb756ac552dffea5b15701997ed"
DATA_SHA = "129a36aa93ea0de488d2bcc876e396de9e3907bf66c6aae6394e542ef6a6d623"
EXTRA_SOURCES = {
    "wm/lewm/lewm.py": "eb259214252af0b01d309044e632cd95ada06e0bb5ef0383102f03699b64a5cb",
    "world/world.py": "39318b81ed151d8556d8540f460a63eedaed0ce4b2211ce0af9f8200e7d83bde",
    "world/env_pool.py": "b6e73253199e1a39f0fcadfad7dbea09270cbd559e366d1bc9bf09c9e7da0606",
    "data/formats/hdf5.py": "afe836ec706acc6c55450f3c936a5898b57ed65a999695af25b4b59cd14d10f3",
    "spaces.py": "73079b2a66dace087abc7c417691e3cc339887866915553e21d0936a362229eb",
}
TASK_HASHES = {
    42: "acd63f952c7fba3cd893bd5db376dac1f6a7136ea6f2077b73a6eb2f030d7ad5",
    43: "6b14b80bd54aedf0c729d5bc01a52d316bbf0d4eb5a752daee9f1e258dde3ec6",
    44: "712dc643852fcf6f9111ec07c47e3c31e17e2c06f252d6f7f639dfb51b12fece",
    45: "b46f363c77b6acdde7004132b427722ecbb506870ae2d31df97d9fe76d0bf5ce",
}
MODELS = ("released", "retrained", "bilinear")
OBJECTIVES = ("latent", "position")
SEEDS = (43, 44, 45)
OUT = Path(os.environ.get("EXP23_OUT", "/kaggle/working"))
TMP = Path(os.environ.get("EXP23_TMP", "/kaggle/tmp/exp23"))
_BASE = None


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_source(path, target, expected):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        url = f"https://raw.githubusercontent.com/sahnas/stigmergic-module-routing/{REPO_REV}/{path}"
        with urllib.request.urlopen(url, timeout=90) as response:
            target.write_bytes(response.read())
    assert digest(target) == expected, f"Changed source: {path}"
    return target


def base():
    global _BASE
    if _BASE is None:
        local = Path(__file__).resolve().parent.parent / "lewm-cem-selection/selection.py"
        source = local if local.is_file() else download_source(
            "kaggle/lewm-cem-selection/selection.py", TMP / "selection.py", BASE_SHA)
        assert digest(source) == BASE_SHA
        spec = importlib.util.spec_from_file_location("exp21_frozen", source)
        _BASE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_BASE)
        _BASE.OUT, _BASE.TMP = OUT, TMP
    return _BASE


def configure_environment():
    os.environ.update(STABLEWM_HOME=str(TMP / "stablewm"), HF_HOME=str(TMP / "hf"),
                      WANDB_MODE="disabled", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
                      SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy", PYTHONUNBUFFERED="1")


def attach_objective(model, objective, weights):
    """Override only criterion; preserve the official get_cost/rollout path."""
    import torch
    assert objective in OBJECTIVES
    assert weights.ndim == 2 and weights.shape[1] == 2
    assert torch.isfinite(weights).all()
    if objective == "latent":
        return  # no wrapper at all around the baseline criterion
    W = weights[:-1].detach().clone()

    def criterion(self, info):
        pred = info["predicted_emb"]
        assert pred.shape[0] == 1, "Pinned official CEM must score one environment at a time"
        goal = info["goal_emb"][..., -1:, :].expand_as(pred)
        delta = pred[..., -1:, :] - goal[..., -1:, :].detach()
        projected = delta @ W.to(device=delta.device, dtype=delta.dtype)
        cost = projected.square().sum(dim=tuple(range(2, pred.ndim)))
        assert torch.isfinite(cost).all(), "Non-finite position costs"
        return cost

    model.criterion = MethodType(criterion, model)


def sample_tasks(episode_ids, step_indices, seed, count=100):
    """Exact official draw, including its len(valid_indices)-1 convention."""
    import numpy as np
    episode_ids, step_indices = np.asarray(episode_ids), np.asarray(step_indices)
    assert episode_ids.shape == step_indices.shape and episode_ids.ndim == 1
    unique, inv = np.unique(episode_ids, return_inverse=True)
    maximum = np.full(len(unique), -1, dtype=np.int64)
    np.maximum.at(maximum, inv, step_indices)
    valid = np.flatnonzero(step_indices <= maximum[inv] - 25)
    drawn = np.random.default_rng(seed).choice(len(valid) - 1, size=count, replace=False)
    rows = np.sort(valid[drawn])
    return [dict(episode_id=int(episode_ids[i]), start_step=int(step_indices[i]),
                 goal_step=int(step_indices[i]) + 25) for i in rows]


def probe_indices(lengths, offsets, excluded):
    import numpy as np
    lengths, offsets = np.asarray(lengths), np.asarray(offsets)
    n = len(lengths)
    heldout = set(np.random.default_rng(0).choice(n, size=n // 10, replace=False).tolist())
    train_eps = [e for e in range(n) if e not in heldout and e not in excluded]
    valid_eps = [e for e in range(n) if e in heldout and e not in excluded]
    assert train_eps and valid_eps
    frames = lambda eps: np.concatenate([np.arange(offsets[e], offsets[e] + lengths[e]) for e in eps])
    train_frames, valid_frames = frames(train_eps), frames(valid_eps)
    selected = np.random.default_rng(1).choice(train_frames, min(300000, len(train_frames)), replace=False)
    return selected, valid_frames, train_eps, valid_eps


def fit_ridge(embeddings, positions, indices, batch_size=8192):
    import numpy as np
    width = embeddings.shape[1] + 1
    gram, cross = np.zeros((width, width), dtype=np.float64), np.zeros((width, 2), dtype=np.float64)
    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]
        x = np.column_stack([np.asarray(embeddings[idx], dtype=np.float64), np.ones(len(idx))])
        y = np.asarray(positions[idx], dtype=np.float64)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        gram += x.T @ x
        cross += x.T @ y
    return np.linalg.solve(gram + 1e-3 * len(indices) * np.eye(width), cross).astype(np.float32)


def probe_metrics(embeddings, positions, indices, weights):
    import numpy as np
    errors, targets = [], []
    for start in range(0, len(indices), 8192):
        idx = indices[start:start + 8192]
        x = np.asarray(embeddings[idx], dtype=np.float32)
        y = np.asarray(positions[idx], dtype=np.float32)
        errors.append(x @ weights[:-1] + weights[-1] - y)
        targets.append(y)
    error, truth = np.concatenate(errors), np.concatenate(targets)
    assert np.isfinite(error).all() and np.isfinite(truth).all()
    variance = truth.var(0)
    assert (variance > 0).all()
    r2 = 1 - np.square(error).mean(0) / variance
    distances = np.linalg.norm(error, axis=1)
    median, p95 = np.quantile(distances, [.5, .95])
    return dict(r2=r2.tolist(), median=float(median), p95=float(p95), frames=len(indices),
                pass_=bool((r2 >= .9).all() and median <= 4 and p95 <= 8))


def pair_rows(a, b):
    assert len(a) == len(b) == 100, "Expected 100 paired outcomes"
    for rows in (a, b):
        assert all(isinstance(row["success"], bool) for row in rows)
    return base().paired(a, b)


def analysis(root, baseline_only=False):
    import numpy as np
    assert not (Path(root) / "failure.json").exists(), "Failed run; do not interpret as a scientific result"
    result = {}
    for seed in SEEDS:
        def read(model, objective):
            return json.loads((Path(root) / "evaluation" / f"{model}_{objective}_seed{seed}" / "episodes.json").read_text())
        b0, n0 = read("bilinear", "latent"), read("retrained", "latent")
        g0 = pair_rows(b0, n0)["difference_percentage_points"]
        row = dict(baseline_gap_pp=g0)
        if not baseline_only:
            data = {(m, o): read(m, o) for m in MODELS for o in OBJECTIVES}
            for rows in data.values():
                pair_rows(b0, rows)
            b1, n1 = data["bilinear", "position"], data["retrained", "position"]
            g1 = pair_rows(b1, n1)["difference_percentage_points"]
            d = np.array([int(x0["success"]) - int(y0["success"]) - int(x1["success"]) + int(y1["success"])
                          for x0, y0, x1, y1 in zip(b0, n0, b1, n1)])
            rng = np.random.default_rng(23000 + seed)
            samples = 100 * d[rng.integers(0, 100, size=(20000, 100))].mean(1)
            interaction = float(100 * d.mean())
            assert abs(interaction - (g0 - g1)) < 1e-8
            row.update(position_gap_pp=g1, interaction_pp=interaction,
                       interaction_bootstrap_95_ci=np.quantile(samples, [.025, .975]).tolist(),
                       counts={m: {o: sum(r["success"] for r in data[m, o]) for o in OBJECTIVES} for m in MODELS},
                       objective_effects={m: pair_rows(data[m, "position"], data[m, "latent"]) for m in MODELS},
                       bilinear_comparisons={o: {m: pair_rows(data["bilinear", o], data[m, o]) for m in ("released", "retrained")} for o in OBJECTIVES})
        result[str(seed)] = row
    reproduced = all(r["baseline_gap_pp"] >= 5 for r in result.values())
    criterion = not baseline_only and reproduced and all(r["interaction_pp"] >= 5 for r in result.values())
    return dict(seeds=result, baseline_reproduced=reproduced, primary_met=criterion,
                status="baseline_only" if baseline_only else "criterion_met" if criterion else "criterion_not_met",
                scope="Fixed fits, reused evaluation seeds; descriptive paired CIs per seed; no training-seed inference.")


class GateComplete(Exception):
    pass


def replay_clone(native, controls):
    """Clone native TwoRoom state; never reset or stop early on termination."""
    import numpy as np
    import torch
    original_pixels = native.render().copy()
    original_state = native.agent_position.clone()
    cloned = copy.deepcopy(native)
    np.testing.assert_array_equal(cloned.render(), original_pixels)
    torch.testing.assert_close(cloned.agent_position, original_state, rtol=0, atol=0)
    for action in controls:
        _, _, _, _, info = cloned.step(action)
    np.testing.assert_array_equal(native.render(), original_pixels)
    torch.testing.assert_close(native.agent_position, original_state, rtol=0, atol=0)
    return cloned.render().copy(), np.asarray(info["state"]).copy()


def gate_diagnostics(world, policy, info, actions, weights, run_dir):
    import numpy as np
    import torch
    assert info["pixels"].shape[1] == 1 and len(actions) == 8
    helper = base()
    sent, applied, normalized = helper.effective_plan("tworoom", actions.numpy(), policy.process["action"])
    solver, model = policy.solver, policy.solver.model
    rng_before = solver.torch_gen.get_state().clone()
    rows, predicted = [], []
    for i in range(8):
        # Row-wise calls preserve the official batch-one goal semantics.
        local = helper.expanded_info({k: v[i:i + 1] for k, v in info.items()}, 1, solver.device)
        plan = torch.as_tensor(normalized[i:i + 1, None], device=solver.device, dtype=solver.dtype)
        with torch.inference_mode():
            scored = model.get_cost(local, plan)
            zhat = local["predicted_emb"][0, 0, -1].detach().cpu().numpy()
        px, true = replay_clone(world.envs.envs[i].unwrapped, sent[i].reshape(25, 2))
        prepared = policy._prepare_info({"pixels": px[None, None]})
        prepared = {k: v.to(solver.device) for k, v in prepared.items()}
        with torch.inference_mode():
            zreal = model.encode(prepared)["emb"][0, 0].detach().cpu().numpy()
        decode = lambda z: z @ weights[:-1] + weights[-1]
        a, b, c = decode(zreal) - true, decode(zhat) - decode(zreal), decode(zhat) - true
        assert all(np.isfinite(x).all() for x in (zhat, zreal, a, b, c))
        np.testing.assert_allclose(a + b, c, rtol=1e-5, atol=1e-4)
        predicted.append(zhat)
        rows.append(dict(row=i, real_context=1, effective_plan_cost=float(scored.item()),
                         true_position=true.tolist(), decoded_observation_error=a.tolist(),
                         decoded_prediction_difference=b.tolist(), composite_error=c.tolist()))
    assert torch.equal(rng_before, solver.torch_gen.get_state())
    np.savez_compressed(run_dir / "first_plans.npz", normalized=actions.numpy(),
                        submitted=sent, effective=applied, predicted_terminal=np.asarray(predicted))
    helper.save_json(run_dir / "gate.json", dict(pass_=True, rows=rows,
                     warning="Composite errors are descriptive; no predictor excluded or probe retuned using them."))


def worker(model_name, objective, seed, phase):
    configure_environment()
    import hdf5plugin  # noqa: F401; register filters before reading HDF5
    import numpy as np
    import torch
    import stable_worldmodel as swm
    assert model_name in MODELS and objective in OBJECTIVES
    assert (phase == "gate" and seed == 42) or (phase == "evaluation" and seed in SEEDS)
    helper = base()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads((OUT / "input_manifest.json").read_text())
    frozen = json.loads((OUT / "decoder_manifest.json").read_text())
    assert digest(OUT / "decoder.npy") == frozen["sha256"] and frozen["metrics"]["pass_"]
    weights = np.load(OUT / "decoder.npy", allow_pickle=False)
    run_dir = OUT / phase / f"{model_name}_{objective}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    original_load = swm.wm.utils.load_pretrained

    def load_selected(name, *args, **kwargs):
        assert name == model_name
        folder = Path(manifest["retrained"] if name == "retrained" else manifest["released"])
        assert digest(folder / "weights.pt") == helper.WEIGHT_HASHES["tworoom"]["retrained" if name == "retrained" else "released"]
        model = original_load(str(folder), *args, **kwargs)
        if name == "bilinear":
            assert digest(manifest["bilinear"]) == helper.WEIGHT_HASHES["tworoom"]["bilinear"]
            saved = torch.load(manifest["bilinear"], map_location="cpu", weights_only=True)
            assert tuple(saved["W"].shape) == (2507, 192)
            model.predictor = helper.make_bilinear_predictor(saved["W"])
            model.action_encoder, model.pred_proj = torch.nn.Identity(), torch.nn.Identity()
        attach_objective(model, objective, torch.from_numpy(weights))
        return model

    swm.wm.utils.load_pretrained = load_selected
    original_set = swm.World.set_policy

    def set_policy(world, policy):
        solver = policy.solver
        assert solver.__class__.__name__ == "CEMSolver"
        assert (solver.batch_size, solver.num_samples, solver.n_steps, solver.topk, solver.var_scale) == (1, 300, 30, 30, 1.)
        assert (policy.cfg.horizon, policy.cfg.receding_horizon, policy.cfg.action_block, policy.cfg.history_len) == (5, 5, 5, 1)
        solve = solver.solve
        decisions = []

        def observed_solve(info, init_action=None):
            if not decisions:
                assert info["pixels"].shape[1] == 1
            output = solve(info, init_action=init_action)
            assert torch.isfinite(output["actions"]).all()
            decisions.append(dict(rows=len(output["actions"]), context=int(info["pixels"].shape[1])))
            helper.save_json(run_dir / "decisions.json", decisions)
            if phase == "gate":
                gate_diagnostics(world, policy, info, output["actions"], weights, run_dir)
                raise GateComplete()
            return output

        solver.solve = observed_solve
        for index, env in enumerate(world.envs.envs):
            step = env.step

            def logged_step(action, step=step, index=index):
                action = np.asarray(action, dtype=np.float32)
                assert np.isfinite(action).all()
                with (run_dir / "controls.jsonl").open("a") as f:
                    f.write(json.dumps(dict(env=index, submitted=action.tolist(), effective=np.clip(action, -1, 1).tolist()), allow_nan=False) + "\n")
                return step(action)

            env.step = logged_step
        helper.save_json(run_dir / "configuration.json", dict(model=model_name, objective=objective, seed=seed,
                         decoder_sha256=frozen["sha256"], action_mean=policy.process["action"].mean_.tolist(),
                         action_scale=policy.process["action"].scale_.tolist(), candidates_per_decision=9000))
        return original_set(world, policy)

    swm.World.set_policy = set_policy
    original_evaluate = swm.World.evaluate

    def evaluate(world, *args, **kwargs):
        assert not args and kwargs["goal_offset"] == 25 and kwargs["eval_budget"] == 50
        tasks = [dict(episode_id=int(e), start_step=int(s), goal_step=int(s) + 25)
                 for e, s in zip(kwargs["episodes_idx"], kwargs["start_steps"])]
        assert tasks == json.loads((OUT / "tasks" / f"tworoom_seed{seed}.json").read_text())
        if phase == "gate":
            tasks = tasks[:8]
            kwargs["episodes_idx"], kwargs["start_steps"] = kwargs["episodes_idx"][:8], kwargs["start_steps"][:8]
        helper.save_json(run_dir / "tasks.json", tasks)
        kwargs["video"] = None
        start = time.monotonic()
        metrics = original_evaluate(world, **kwargs)
        successes = np.asarray(metrics["episode_successes"], dtype=bool)
        assert successes.shape == (len(tasks),)
        assert abs(float(metrics["success_rate"]) - 100 * successes.mean()) < 1e-8
        helper.save_json(run_dir / "episodes.json", [dict(**t, success=bool(s)) for t, s in zip(tasks, successes)])
        helper.save_json(run_dir / "metrics.json", dict(successes=int(successes.sum()), episodes=len(tasks),
                         seconds=time.monotonic() - start,
                         metrics={k: v.tolist() if hasattr(v, "tolist") else v for k, v in metrics.items()}))
        helper.log("evaluation_complete", model=model_name, objective=objective, seed=seed, successes=int(successes.sum()))
        return metrics

    swm.World.evaluate = evaluate
    sys.argv = [str(TMP / "le-wm/eval.py"), "--config-name=tworoom", f"policy={model_name}",
                f"seed={seed}", "eval.num_eval=100", f"output.filename={run_dir / 'official_results.txt'}"]
    if phase == "gate":
        sys.argv.append("world.num_envs=8")
    helper.save_json(run_dir / "command.json", sys.argv)
    try:
        runpy.run_path(str(TMP / "le-wm/eval.py"), run_name="__main__")
    except GateComplete:
        assert phase == "gate" and (run_dir / "gate.json").is_file()


def bootstrap():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "input_manifest.json").exists(), "Use a fresh output directory; no silent reruns"
    TMP.mkdir(parents=True, exist_ok=True)
    configure_environment()
    helper = base()
    helper.command(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    helper.command(["apt-get", "install", "-y", "-qq", "libegl1", "libgl1", "ffmpeg"])
    helper.command([sys.executable, "-m", "pip", "install", "-q", "stable-worldmodel[train,format]==0.1.1",
                    "stable-pretraining==0.1.8", "transformers==4.57.6", "hydra-core==1.3.2",
                    "pygame==2.6.1", "pymunk==7.2.0", "shapely==2.1.2", "opencv-python-headless==4.13.0.92"])
    import torch
    import stable_worldmodel as swm
    torch.set_num_threads(4)
    sources = {k: v for k, v in helper.SOURCE_HASHES.items() if "pusht" not in k} | EXTRA_SOURCES
    root = Path(swm.__file__).parent
    for name, expected in sources.items():
        assert digest(root / name) == expected, f"Changed source: {name}"
        target = OUT / "official_library" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    check = checks(root)
    helper.save_json(OUT / "runtime_checks.json", check)
    (OUT / "environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    helper.command(["git", "clone", "--quiet", "--no-checkout", "https://github.com/lucas-maes/le-wm.git", TMP / "le-wm"])
    helper.command(["git", "-C", TMP / "le-wm", "checkout", "--quiet", "--detach", helper.CODE_REV])
    assert digest(TMP / "le-wm/eval.py") == helper.EVAL_SHA
    shutil.copyfile(TMP / "le-wm/eval.py", OUT / "official_eval.py")
    shutil.copytree(TMP / "le-wm/config/eval", OUT / "official_eval_config")
    rel = helper.unique_matching("models--quentinll--lewm-tworooms/weights.pt", helper.WEIGHT_HASHES["tworoom"]["released"])
    rt = helper.unique_matching("tworoom_retrained_v2/weights.pt", helper.WEIGHT_HASHES["tworoom"]["retrained"])
    bil = helper.unique_matching("koopman_bilinear.pt", helper.WEIGHT_HASHES["tworoom"]["bilinear"])
    assert digest(rel.parent / "config.json") == helper.CONFIG_HASHES["tworoom_released"]
    assert digest(rt.parent / "config.json") == helper.CONFIG_HASHES["tworoom_retrained"]
    a, b = [torch.load(p, map_location="cpu", weights_only=True) for p in (rel, rt)]
    keys = {k for k in a if k.startswith(("encoder.", "projector."))}
    assert keys == {k for k in b if k.startswith(("encoder.", "projector."))}
    assert keys and all(torch.equal(a[k], b[k]) for k in keys)
    del a, b
    dataset = helper.unique_matching("tworoom.h5", DATA_SHA)
    datasets = TMP / "stablewm/datasets"
    datasets.mkdir(parents=True, exist_ok=True)
    (datasets / "tworoom.h5").symlink_to(dataset)
    cache = helper.unique_matching("emb_tworoom.npy").parent
    cache_files = ["emb_tworoom.npy", "pos_agent.npy", "ep_len.npy", "ep_offset.npy", "ep_idx.npy"]
    cache_hashes = {name: digest(cache / name) for name in cache_files}
    helper.save_json(OUT / "input_manifest.json", dict(released=str(rel.parent), retrained=str(rt.parent),
                     bilinear=str(bil), dataset=str(dataset), cache=str(cache), cache_hashes=cache_hashes,
                     dataset_sha256=DATA_SHA, checkpoint_sha256=helper.WEIGHT_HASHES["tworoom"],
                     source_sha256=sources, code_sha256=digest(__file__), helper_sha256=BASE_SHA))
    shutil.copyfile(__file__, OUT / "submitted_objective.py")
    shutil.copyfile(Path(helper.__file__), OUT / "frozen_selection.py")
    for seed, sha in TASK_HASHES.items():
        download_source(f"results/lewm_kaggle/exp21_selection/version2/tasks/tworoom_seed{seed}.json",
                        OUT / "tasks" / f"tworoom_seed{seed}.json", sha)


def prepare_probe():
    configure_environment()
    import hdf5plugin  # noqa: F401
    import h5py
    import numpy as np
    import torch
    import stable_worldmodel as swm
    from torchvision.transforms import v2 as transforms
    helper = base()
    manifest = json.loads((OUT / "input_manifest.json").read_text())
    cache = Path(manifest["cache"])
    for name, sha in manifest["cache_hashes"].items():
        assert digest(cache / name) == sha
    emb = np.load(cache / "emb_tworoom.npy", mmap_mode="r", allow_pickle=False)
    pos = np.load(cache / "pos_agent.npy", mmap_mode="r", allow_pickle=False)
    lengths, offsets = [np.load(cache / (name + ".npy"), allow_pickle=False) for name in ("ep_len", "ep_offset")]
    assert emb.shape == (int(lengths.sum()), 192) and emb.dtype == np.float32
    assert pos.shape == (len(emb), 2)
    np.testing.assert_array_equal(offsets, np.r_[0, np.cumsum(lengths)[:-1]])
    with h5py.File(manifest["dataset"], "r") as f:
        for name, values in (("ep_len", lengths), ("ep_offset", offsets), ("pos_agent", pos)):
            np.testing.assert_array_equal(f[name][:], values)
        episodes = np.asarray(f["ep_idx"][:]).reshape(-1)
        steps = np.asarray(f["step_idx"][:]).reshape(-1)
        np.testing.assert_array_equal(episodes, np.repeat(np.arange(len(lengths)), lengths))
        np.testing.assert_array_equal(episodes, np.load(cache / "ep_idx.npy", allow_pickle=False).reshape(-1))
        excluded = set()
        for seed in (42, *SEEDS):
            tasks = json.loads((OUT / "tasks" / f"tworoom_seed{seed}.json").read_text())
            assert sample_tasks(episodes, steps, seed) == tasks
            excluded.update(row["episode_id"] for row in tasks)
        train, valid, train_eps, valid_eps = probe_indices(lengths, offsets, excluded)
        np.savez_compressed(OUT / "probe_split.npz", train_indices=train, validation_indices=valid,
                            train_episodes=train_eps, validation_episodes=valid_eps, excluded_episodes=sorted(excluded))
        # Independent model.encode preprocessing parity on fixed training frames.
        # This checks 64 frames, not every cache entry; report that coverage.
        chosen = np.sort(train[:64])
        model = swm.wm.utils.load_pretrained(manifest["released"]).cuda().eval()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        transform = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
                        transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]), transforms.Resize((224, 224))])
        pixels = torch.stack([transform(torch.from_numpy(x).permute(2, 0, 1)) for x in f["pixels"][chosen]]).cuda()
        with torch.inference_mode():
            encoded = model.encode({"pixels": pixels[:, None]})["emb"][:, 0].cpu().numpy()
        error = float(np.max(np.abs(encoded - emb[chosen])))
        assert error <= 1e-4, f"Cache parity failed: {error}"
        del model, pixels
        torch.cuda.empty_cache()
    weights = fit_ridge(emb, pos, train)
    np.save(OUT / "decoder.npy", weights)
    metrics = probe_metrics(emb, pos, valid, weights)
    helper.save_json(OUT / "decoder_manifest.json", dict(sha256=digest(OUT / "decoder.npy"), metrics=metrics,
                     split_sha256=digest(OUT / "probe_split.npz"), input_cache_hashes=manifest["cache_hashes"],
                     cache_parity_frames=64, cache_parity_max_absolute=error,
                     train_frames=len(train), validation_frames=len(valid), excluded_episodes=len(excluded),
                     fit="float64 batched ridge, lambda=1e-3*N; float32 decoder; bias penalized"))
    return metrics["pass_"]


def checks(package):
    """Synthetic tests on actual pinned LeWM source; no dataset/model weights."""
    import numpy as np
    import torch
    from types import SimpleNamespace
    package = Path(package)
    assert digest(package / "wm/lewm/lewm.py") == EXTRA_SOURCES["wm/lewm/lewm.py"]
    LeWM = runpy.run_path(str(package / "wm/lewm/lewm.py"))["LeWM"]

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.ones(1))
        def forward(self, pixels, **unused):
            return SimpleNamespace(last_hidden_state=pixels.unsqueeze(1) * self.anchor)

    class Predictor(torch.nn.Module):
        num_frames = 3
        def forward(self, emb, act):
            return emb + .1 * act.sum(-1, keepdim=True)

    p = torch.tensor([[1., 2.], [-1., .5], [.25, -.25], [9., -4.]])
    expected_contexts = []
    for history in (1, 2, 3):
        for goal_index in range(8):
            models = [LeWM(Encoder(), Predictor(), torch.nn.Identity()).eval() for _ in range(3)]
            attach_objective(models[1], "latent", p)
            attach_objective(models[2], "position", p)
            plans = torch.arange(4 * 5 * 10, dtype=torch.float32).reshape(1, 4, 5, 10) / 100
            info = dict(pixels=torch.arange(history * 3, dtype=torch.float32).reshape(1, 1, history, 3).expand(1, 4, history, 3),
                        goal=torch.full((1, 4, 1, 3), float(goal_index)), action=torch.zeros(1, 4, history, 10))
            copies = [{k: v.clone() for k, v in info.items()} for _ in models]
            with torch.inference_mode():
                costs = [m.get_cost(x, plans) for m, x in zip(models, copies)]
            torch.testing.assert_close(costs[0], costs[1], rtol=0, atol=0)
            torch.testing.assert_close(copies[0]["predicted_emb"], copies[2]["predicted_emb"], rtol=0, atol=0)
            pred = copies[0]["predicted_emb"][..., -1, :]
            goal = copies[0]["goal_emb"][:, -1, :][:, None]
            expected = ((pred @ p[:-1] + p[-1]) - (goal @ p[:-1] + p[-1])).square().sum(-1)
            torch.testing.assert_close(costs[2], expected, rtol=2e-5, atol=1e-5)
            if history == 1:
                # Independent recurrence: first action adds .1*sum(block), five times.
                terminal = info["pixels"][:, :, 0] + .1 * plans.sum((-1, -2))[..., None]
                torch.testing.assert_close(pred, terminal, rtol=1e-6, atol=1e-5)
            expected_contexts.append(history)
    return dict(pass_=True, torch=torch.__version__, synthetic_calls=len(expected_contexts),
                tests=["official latent cost untouched", "same official rollouts under either objective",
                       "independent affine decoding with bias cancellation", "one/two/three-frame contexts and eight goals",
                       "one-frame rollout equals independent five-action recurrence"])


def main():
    bootstrap()
    helper = base()
    if not prepare_probe():
        helper.save_json(OUT / "complete.json", dict(complete=True, status="stopped_decoder_gate", evaluations=0))
        return
    for model in MODELS:
        for objective in OBJECTIVES:
            helper.command([sys.executable, __file__, "--worker", model, objective, "42", "gate"])
    for model in MODELS:
        for seed in SEEDS:
            helper.command([sys.executable, __file__, "--worker", model, "latent", str(seed), "evaluation"])
    baseline = analysis(OUT, baseline_only=True)
    helper.save_json(OUT / "baseline_analysis.json", baseline)
    if not baseline["baseline_reproduced"]:
        helper.save_json(OUT / "complete.json", dict(complete=True, status="stopped_baseline_not_reproduced", evaluations=9))
        return
    for model in MODELS:
        for seed in SEEDS:
            helper.command([sys.executable, __file__, "--worker", model, "position", str(seed), "evaluation"])
    result = analysis(OUT)
    helper.save_json(OUT / "analysis.json", result)
    helper.save_json(OUT / "complete.json", dict(complete=True, status="evaluated", evaluations=18,
                     primary_met=result["primary_met"], inference=result["status"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=4, metavar=("MODEL", "OBJECTIVE", "SEED", "PHASE"))
    parser.add_argument("--analyze", type=Path)
    parser.add_argument("--check", type=Path, help="Path to stable_worldmodel 0.1.1 source")
    args = parser.parse_args()
    try:
        if args.check:
            print(json.dumps(checks(args.check), indent=2))
        elif args.analyze:
            print(json.dumps(analysis(args.analyze), indent=2))
        elif args.worker:
            model, objective, seed, phase = args.worker
            worker(model, objective, int(seed), phase)
        else:
            main()
    except BaseException as error:
        if not args.check and not args.analyze:
            base().save_json(OUT / "failure.json", dict(type=type(error).__name__, message=str(error),
                              worker=args.worker, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        raise
