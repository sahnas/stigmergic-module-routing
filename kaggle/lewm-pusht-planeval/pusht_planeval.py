"""Experiment 19: twelve official PushT evaluations, with paired result capture.

The pinned eval.py, solver and environment are not edited. Hooks select the
fixed checkpoint/operator, check normalization, and record tasks and metrics.
The video destination is changed only to keep each run's recordings separate.
"""
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

MODEL_REV = "22b330c28c27ead4bfd1888615af1340e3fe9052"
DATA_REV = "655cd446b992"
CODE_REV = "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"
FIT_SOURCE_SHA256 = "e9f6ae86e4b989c5b6f54b3e4946aa517f4df49bcce60570def0582f36ea20ba"
SEEDS = (42, 43, 44, 45)
MODELS = ("released", "retrained", "bilinear")
OUT = Path("/kaggle/working")
TMP = Path("/kaggle/tmp/exp19_eval")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def log(event, **value):
    text = json.dumps(dict(event=event, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **value), allow_nan=False)
    print(text, flush=True)
    with (OUT / "progress.jsonl").open("a") as f:
        f.write(text + "\n")


def command(args):
    print("$ " + " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True)


def jsonable(value):
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


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
                bil = (z0.unsqueeze(2) * a.unsqueeze(1)).reshape(batch, -1)
                x = torch.cat([z0, z1, z2, a, bil,
                               torch.ones(batch, 1, device=emb.device, dtype=emb.dtype)], 1)
                predictions.append(x @ self.W.to(emb.dtype))
            return torch.stack(predictions, 1)

    return BilinearPredictor()


def paired_stats(operator_rows, reference_rows):
    """Exact two-sided McNemar test on the same ordered starting tasks."""
    assert len(operator_rows) == len(reference_rows) > 0
    key = lambda r: (r["episode_id"], r["start_step"], r["goal_step"])
    assert [key(r) for r in operator_rows] == [key(r) for r in reference_rows], "Task pairing mismatch"
    op = [bool(r["success"]) for r in operator_rows]
    ref = [bool(r["success"]) for r in reference_rows]
    wins = sum(a and not b for a, b in zip(op, ref))
    losses = sum(b and not a for a, b in zip(op, ref))
    discordant = wins + losses
    p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2 ** discordant) if discordant else 1.0
    return dict(n=len(op), bilinear_successes=sum(op), reference_successes=sum(ref),
                difference_successes=sum(op) - sum(ref),
                difference_percentage_points=100 * (sum(op) - sum(ref)) / len(op),
                bilinear_only=wins, reference_only=losses,
                both_success=sum(a and b for a, b in zip(op, ref)),
                both_failure=sum(not a and not b for a, b in zip(op, ref)),
                mcnemar_exact_two_sided_p=p,
                strictly_more_successes=sum(op) > sum(ref))


def worker(model_name, seed):
    import numpy as np
    import torch
    import hdf5plugin  # register filters before HDF5Dataset is used
    import stable_worldmodel as swm

    assert model_name in MODELS and seed in SEEDS
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fit = Path(os.environ["EXP19_FIT"])
    upstream = TMP / "le-wm"
    run_dir = OUT / f"{model_name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((fit / "provenance.json").read_text())
    original_load = swm.wm.utils.load_pretrained

    def load_selected(name, *args, **kwargs):
        assert name == model_name, f"Unexpected policy: {name}"
        folder = "pusht_retrained_v2" if name == "retrained" else "pusht_released"
        model = original_load(str(fit / folder), *args, **kwargs)
        if name == "bilinear":
            saved = torch.load(fit / "pusht_bilinear.pt", map_location="cpu", weights_only=True)
            assert saved["model_revision"] == MODEL_REV and saved["history"] == 3 and saved["frameskip"] == 5
            assert tuple(saved["W"].shape) == (2507, 192)
            model.predictor = make_bilinear_predictor(saved["W"])
            model.action_encoder = torch.nn.Identity()
            model.pred_proj = torch.nn.Identity()
        return model

    swm.wm.utils.load_pretrained = load_selected
    original_set_policy = swm.World.set_policy

    def checked_set_policy(world, policy):
        scaler = policy.process["action"]
        np.testing.assert_allclose(scaler.mean_, provenance["action_mean"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(scaler.scale_, provenance["action_scale"], rtol=0, atol=1e-12)
        assert (policy.cfg.horizon, policy.cfg.receding_horizon, policy.cfg.action_block) == (5, 5, 5)
        solver = policy.solver
        assert solver.__class__.__name__ == "CEMSolver"
        assert (solver.batch_size, solver.num_samples, solver.n_steps, solver.topk, solver.var_scale) == (1, 300, 30, 30, 1.0)
        save_json(run_dir / "solver_and_normalization.json",
                  dict(horizon=5, receding_horizon=5, action_block=5, num_samples=300,
                       iterations=30, elites=30, batch_size=1, var_scale=1.0, seed=seed,
                       action_mean=scaler.mean_.tolist(), action_scale=scaler.scale_.tolist()))
        return original_set_policy(world, policy)

    swm.World.set_policy = checked_set_policy
    original_evaluate = swm.World.evaluate

    def recorded_evaluate(world, *args, **kwargs):
        assert not args, "Unexpected positional evaluator arguments"
        assert kwargs["goal_offset"] == 25 and kwargs["eval_budget"] == 50
        assert len(kwargs["episodes_idx"]) == len(kwargs["start_steps"]) == 100
        tasks = [dict(episode_id=int(e), start_step=int(s), goal_step=int(s) + 25)
                 for e, s in zip(kwargs["episodes_idx"], kwargs["start_steps"])]
        reference_path = OUT / f"seed{seed}_tasks.json"
        if reference_path.exists():
            assert json.loads(reference_path.read_text()) == tasks, "Paired evaluation starts/goals differ"
        else:
            save_json(reference_path, tasks)
        save_json(run_dir / "tasks.json", tasks)
        kwargs["video"] = run_dir / "videos"  # record exactly the same videos in a distinct directory
        log("official_evaluation_started", model=model_name, seed=seed, episodes=len(tasks))
        t0 = time.monotonic()
        result = original_evaluate(world, **kwargs)
        successes = np.asarray(result["episode_successes"])
        assert successes.shape == (100,) and np.isin(successes, [False, True]).all()
        successes = successes.astype(bool)
        assert abs(float(result["success_rate"]) - successes.mean() * 100) < 1e-9
        rows = [dict(**task, success=bool(success), model=model_name, evaluation_seed=seed)
                for task, success in zip(tasks, successes)]
        save_json(run_dir / "episodes.json", rows)
        save_json(run_dir / "metrics.json", dict(model=model_name, evaluation_seed=seed,
                                                  elapsed_seconds=time.monotonic() - t0, metrics=jsonable(result)))
        log("official_evaluation_complete", model=model_name, seed=seed,
            successes=int(successes.sum()), episodes=len(tasks), seconds=time.monotonic() - t0)
        return result

    swm.World.evaluate = recorded_evaluate
    sys.argv = [str(upstream / "eval.py"), "--config-name=pusht", f"policy={model_name}",
                f"seed={seed}", "eval.num_eval=100", f"output.filename={run_dir / 'official_results.txt'}"]
    save_json(run_dir / "command.json", sys.argv)
    runpy.run_path(str(upstream / "eval.py"), run_name="__main__")


def bootstrap():
    OUT.mkdir(exist_ok=True, parents=True)
    TMP.mkdir(exist_ok=True, parents=True)
    os.environ.update(STABLEWM_HOME=str(TMP / "stablewm"), HF_HOME=str(TMP / "hf"),
                      WANDB_MODE="disabled", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
                      SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy", PYTHONUNBUFFERED="1")
    command(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    command(["apt-get", "install", "-y", "-qq", "zstd", "libegl1", "libgl1", "ffmpeg"])
    command([sys.executable, "-m", "pip", "install", "-q", "stable-worldmodel[train,format]==0.1.1",
             "stable-pretraining==0.1.8", "transformers==4.57.6", "hydra-core",
             "pygame", "pymunk", "shapely", "opencv-python-headless"])
    (OUT / "environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    upstream = TMP / "le-wm"
    command(["git", "clone", "--quiet", "--no-checkout", "https://github.com/lucas-maes/le-wm.git", upstream])
    command(["git", "-C", upstream, "checkout", "--quiet", "--detach", CODE_REV])
    assert subprocess.check_output(["git", "-C", upstream, "rev-parse", "HEAD"], text=True).strip() == CODE_REV
    shutil.copyfile(upstream / "eval.py", OUT / "official_eval.py")
    shutil.copytree(upstream / "config/eval", OUT / "official_eval_config")
    return upstream


def main():
    t0 = time.monotonic()
    upstream = bootstrap()
    from huggingface_hub import hf_hub_download
    fits = list(Path("/kaggle/input").rglob("fit_complete.json"))
    assert len(fits) == 1, fits
    fit = fits[0].parent
    status = json.loads(fits[0].read_text())
    assert status["complete"] and status["completed_epochs"] == status["required_epochs"] == 8
    assert status["reload_max_abs"] <= 1e-6
    provenance = json.loads((fit / "provenance.json").read_text())
    assert provenance["model_revision"] == MODEL_REV and provenance["upstream_revision"] == CODE_REV
    assert provenance["dataset_revision"] == DATA_REV and provenance["script_sha256"] == FIT_SOURCE_SHA256
    assert provenance["cache_parity"]["pass_"]
    assert sha256(fit / "pusht_released/weights.pt") == provenance["weights_sha256"]
    assert sha256(fit / "pusht_released/config.json") == provenance["config_sha256"]
    assert (fit / "pusht_retrained_v2/config.json").read_bytes() == (fit / "pusht_released/config.json").read_bytes()
    for f in ("fit_complete.json", "provenance.json", "context_errors.json", "cache_validation.json", "neural_training.json", "bilinear_meta.json"):
        shutil.copyfile(fit / f, OUT / ("fit_" + f))
    model_hashes = {name: sha256(fit / name) for name in ("pusht_released/weights.pt", "pusht_retrained_v2/weights.pt", "pusht_bilinear.pt")}
    os.environ["EXP19_FIT"] = str(fit)
    save_json(OUT / "evaluation_manifest.json", dict(script_sha256=sha256(__file__), upstream_revision=CODE_REV,
                                                      official_eval_sha256=sha256(upstream / "eval.py"),
                                                      model_revision=MODEL_REV, dataset_revision=DATA_REV,
                                                      model_hashes=model_hashes, seeds=list(SEEDS), models=list(MODELS),
                                                      episodes_per_run=100, primary_seeds=[43, 44, 45],
                                                      hooks=["fixed model selection", "normalization assertions", "task and result capture", "video output directory"],
                                                      action_bounding="unchanged official solver and environment"))
    log("fit_inputs_verified", model_hashes=model_hashes)
    archive = hf_hub_download("quentinll/lewm-pusht", "pusht_expert_train.h5.zst", repo_type="dataset",
                              revision=DATA_REV, local_dir=TMP / "download")
    dataset = TMP / "stablewm/datasets/pusht_expert_train.h5"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    command(["zstd", "-d", "-q", archive, "-o", dataset])
    Path(archive).unlink()
    log("dataset_ready", bytes=dataset.stat().st_size)
    comparisons = {}
    for seed in SEEDS:
        for name in MODELS:
            log("job_started", model=name, seed=seed)
            run_dir = OUT / f"{name}_seed{seed}"
            run_dir.mkdir(exist_ok=True)
            with (run_dir / "execution.log").open("w") as f:
                p = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--job", name, str(seed)],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in p.stdout:
                    f.write(line)
                    f.flush()
                    print(line, end="", flush=True)
                returncode = p.wait()
            assert returncode == 0, f"Official evaluation failed for {name}, seed {seed}: exit {returncode}"
            assert (run_dir / "episodes.json").exists()
        op = json.loads((OUT / f"bilinear_seed{seed}/episodes.json").read_text())
        comparisons[str(seed)] = {}
        for name in ("released", "retrained"):
            reference = json.loads((OUT / f"{name}_seed{seed}/episodes.json").read_text())
            comparisons[str(seed)][name] = paired_stats(op, reference)
        save_json(OUT / "paired_comparisons.json", comparisons)
        log("seed_complete", seed=seed, comparisons=comparisons[str(seed)])
    verdict = dict(primary_bilinear_above_released_on_each_fresh_seed=all(comparisons[str(s)]["released"]["strictly_more_successes"] for s in (43, 44, 45)),
                   secondary_bilinear_above_retrained_on_each_fresh_seed=all(comparisons[str(s)]["retrained"]["strictly_more_successes"] for s in (43, 44, 45)),
                   note="Strict preregistered count criteria; inspect effect sizes and paired p-values. A one-episode win is not strong evidence by itself.")
    save_json(OUT / "evaluation_complete.json", dict(complete=True, runs=12, total_seconds=time.monotonic() - t0,
                                                    verdict=verdict, comparisons=comparisons))
    log("all_evaluations_complete", verdict=verdict, seconds=time.monotonic() - t0)


if __name__ == "__main__":
    try:
        if len(sys.argv) == 4 and sys.argv[1] == "--job":
            worker(sys.argv[2], int(sys.argv[3]))
        else:
            main()
    except BaseException as exc:
        # Hydra exits with SystemExit(0) after a successful worker; leave that untouched.
        if isinstance(exc, SystemExit) and exc.code in (None, 0):
            raise
        if OUT.exists():
            save_json(OUT / "failure.json", dict(type=type(exc).__name__, message=str(exc), argv=sys.argv))
        raise
