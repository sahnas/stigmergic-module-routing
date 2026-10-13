"""CPU checks on pinned source and synthetic data; never research outcomes."""
import argparse
import ast
import contextlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import numpy as np
import torch
import objective as exp


def must_fail(fn):
    try:
        fn()
    except (AssertionError, ValueError):
        return
    raise AssertionError("Invalid input was accepted")


def source_module(path, removals, injected):
    tree = ast.parse(Path(path).read_text())
    tree.body = [n for n in tree.body if not (
        isinstance(n, ast.Import) and any(a.name in removals for a in n.names)
        or isinstance(n, ast.ImportFrom) and n.module in removals)]
    module = ModuleType(Path(path).stem)
    module.__dict__.update(injected)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def check_solver(package):
    # Reuse experiment 21's importer of the unmodified, hash-checked CEM core.
    check_path = Path(__file__).parent.parent / "lewm-cem-selection/check_local.py"
    import sys
    sys.modules["selection"] = exp.base()
    spec = importlib.util.spec_from_file_location("selection_checks", check_path)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    CEM = previous.load_solver(package)
    LeWM = runpy.run_path(str(package / "wm/lewm/lewm.py"))["LeWM"]
    from gymnasium.spaces import Box

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.ones(1))
        def forward(self, pixels, **unused):
            return SimpleNamespace(last_hidden_state=pixels.unsqueeze(1) * self.p)

    class Predictor(torch.nn.Module):
        num_frames = 3
        def forward(self, emb, act):
            return emb + act[..., :3] * .05

    weights = torch.tensor([[1., 0.], [0., .2], [.3, .5], [3., 8.]])
    def solver(objective):
        model = LeWM(Encoder(), Predictor(), torch.nn.Identity()).eval()
        if objective:
            exp.attach_objective(model, objective, weights)
        s = CEM(model, batch_size=1, num_samples=30, n_steps=4, topk=5, seed=77)
        s.configure(action_space=Box(-1, 1, shape=(2, 2), dtype=np.float32), n_envs=2,
                    config=SimpleNamespace(horizon=5, action_block=5))
        return s

    solvers = [solver(x) for x in (None, "latent", "position")]
    info = dict(pixels=torch.zeros(2, 1, 3), goal=torch.tensor([[[1., 2., 3.]], [[3., 2., 1.]]]),
                action=torch.zeros(2, 1, 10))
    with torch.inference_mode():
        outputs = [s.solve({k: v.clone() for k, v in info.items()}) for s in solvers]
    torch.testing.assert_close(outputs[0]["actions"], outputs[1]["actions"], rtol=0, atol=0)
    assert not torch.equal(outputs[1]["actions"], outputs[2]["actions"])
    assert torch.equal(solvers[0].torch_gen.get_state(), solvers[1].torch_gen.get_state())


def check_sampling_and_probe():
    # Independent literal evaluation draw; last valid row is intentionally excluded.
    lengths = np.arange(45, 65)
    offsets = np.r_[0, np.cumsum(lengths)[:-1]]
    episodes = np.repeat(np.arange(20), lengths)
    steps = np.concatenate([np.arange(n) for n in lengths])
    max_per_row = np.array([lengths[e] - 26 for e in episodes])
    valid = np.nonzero(steps <= max_per_row)[0]
    for seed in (42, 43, 44, 45):
        rows = np.sort(valid[np.random.default_rng(seed).choice(len(valid) - 1, 100, replace=False)])
        expected = [dict(episode_id=int(episodes[r]), start_step=int(steps[r]), goal_step=int(steps[r]) + 25) for r in rows]
        assert exp.sample_tasks(episodes, steps, seed) == expected
    train, validation, te, ve = exp.probe_indices(lengths, offsets, {0, 2, 4})
    assert set(te).isdisjoint(ve) and set(te + ve).isdisjoint({0, 2, 4})
    assert set(episodes[train]).isdisjoint(episodes[validation])
    assert len(train) == len(set(train))
    rng = np.random.default_rng(7)
    z = rng.normal(size=(len(episodes), 3)).astype(np.float32)
    true_p = np.array([[4., 1.], [1., 3.], [-2., 2.], [40., 60.]], dtype=np.float32)
    y = np.column_stack([z, np.ones(len(z))]) @ true_p
    actual = exp.fit_ridge(z, y, train, batch_size=17)
    x = np.column_stack([z[train], np.ones(len(train))]).astype(np.float64)
    expected = np.linalg.solve(x.T @ x + 1e-3 * len(x) * np.eye(4), x.T @ y[train]).astype(np.float32)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert exp.probe_metrics(z, y, validation, actual)["pass_"]
    assert not exp.probe_metrics(z, y, validation, np.zeros_like(actual))["pass_"]
    bad = z.copy(); bad[train[0], 0] = np.nan
    must_fail(lambda: exp.fit_ridge(bad, y, train))


def check_cloning(package):
    # Only strip package-root imports; execute the actual spaces and native env.
    swm = SimpleNamespace(utils=source_module(package / "utils.py", set(), {}))
    spaces = source_module(package / "spaces.py", {"stable_worldmodel"}, {"swm": swm})
    swm.spaces = spaces
    environment = source_module(package / "envs/two_room/env.py", {"stable_worldmodel"}, {"swm_spaces": spaces})
    env = environment.TwoRoomEnv()
    env.reset(seed=42)
    env._set_state(np.array([30., 30.], dtype=np.float32))
    env._set_goal_state(np.array([30., 30.], dtype=np.float32))
    start = env.agent_position.clone()
    # A zero first control already terminates. The diagnostic must still execute
    # later controls without autoreset, and clamp physical actions natively.
    controls = np.zeros((25, 2), dtype=np.float32)
    controls[1:4, 0] = 4
    px, state = exp.replay_clone(env, controls)
    speed = float(env.variation_space['agent']['speed'].value.item())
    np.testing.assert_array_equal(state, np.array([30. + 3 * speed, 30.], dtype=np.float32))
    torch.testing.assert_close(env.agent_position, start, rtol=0, atol=0)
    assert px.shape == (224, 224, 3) and px.dtype == np.uint8


def check_analysis_and_flow():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for seed in exp.SEEDS:
            for model in exp.MODELS:
                for objective in exp.OBJECTIVES:
                    count = 95 if model == "bilinear" else 80 if objective == "latent" else 90
                    rows = [dict(episode_id=i, start_step=0, goal_step=25, success=i < count) for i in range(100)]
                    path = root / "evaluation" / f"{model}_{objective}_seed{seed}" / "episodes.json"
                    path.parent.mkdir(parents=True)
                    path.write_text(json.dumps(rows))
        result = exp.analysis(root)
        assert result["primary_met"]
        assert all(r["interaction_pp"] == 10 for r in result["seeds"].values())
        path = root / "evaluation/retrained_position_seed43/episodes.json"
        rows = json.loads(path.read_text()); rows[0]["start_step"] += 1
        path.write_text(json.dumps(rows))
        must_fail(lambda: exp.analysis(root))
        rows[0]["start_step"] -= 1
        for row in rows: row["success"] = row["episode_id"] < 80
        path.write_text(json.dumps(rows))
        assert not exp.analysis(root)["primary_met"]

    # Orchestrator controls: no position research run after a failed decoder or
    # baseline gate; all nine baselines precede all nine interventions.
    old = exp.OUT, exp.bootstrap, exp.prepare_probe, exp.analysis, exp.base
    for decoder_ok, baseline_ok, expected in ((False, True, 0), (True, False, 15), (True, True, 24)):
        calls, saved = [], {}
        exp.bootstrap = lambda: None
        exp.prepare_probe = lambda: decoder_ok
        exp.analysis = lambda root, baseline_only=False: dict(baseline_reproduced=baseline_ok, primary_met=True, status="criterion_met")
        exp.base = lambda: SimpleNamespace(command=lambda cmd: calls.append(cmd), save_json=lambda p, v: saved.update({p.name: v}))
        exp.main()
        assert len(calls) == expected
        if baseline_ok and decoder_ok:
            assert all(c[-3] == "latent" for c in calls[6:15])
            assert all(c[-3] == "position" for c in calls[15:24])
        if not decoder_ok: assert saved["complete.json"]["status"] == "stopped_decoder_gate"
        if decoder_ok and not baseline_ok: assert saved["complete.json"]["status"] == "stopped_baseline_not_reproduced"
    exp.OUT, exp.bootstrap, exp.prepare_probe, exp.analysis, exp.base = old


def main(package):
    torch.set_num_threads(2)
    for name, sha in exp.EXTRA_SOURCES.items():
        assert exp.digest(package / name) == sha
    result = exp.checks(package)
    with contextlib.redirect_stdout(sys.stderr):
        check_solver(package)
    check_sampling_and_probe()
    check_cloning(package)
    check_analysis_and_flow()
    result["tests"] += ["official CEM baseline parity and objective intervention", "official task draw with multiple lengths",
                        "probe episode exclusion and independent ridge solve", "probe rejects invalid values and poor decoding",
                        "native TwoRoom cloned replay, clipping, no autoreset or early stop", "paired analysis rejects task mismatch",
                        "orchestrator stops on failed gates and orders all baselines first"]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    main(parser.parse_args().package)
