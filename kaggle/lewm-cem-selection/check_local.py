"""Exercise the unchanged 0.1.1 solver core, intervention and paired analysis.

CPU check does not load visual models or execute any research episode.
The loader strips package imports only, providing the original helper and
protocol in their place; every solver method is the pinned upstream source.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import torch
import numpy as np
from selection import (SOURCE_HASHES, FinalElites, attach_selection, effective_plan,
                       expanded_info, make_bilinear_predictor, analyze, MODELS, SEEDS, TASKS)


def load_solver(package):
    source = package / "solver/cem.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_HASHES["solver/cem.py"]
    protocols = runpy.run_path(str(package / "protocols.py"))
    ns = dict(Actionable=protocols["Actionable"], Callback=object, Costable=object)
    helper_tree = ast.parse((package / "solver/utils.py").read_text())
    helper_tree.body = [node for node in helper_tree.body
                        if not (isinstance(node, ast.ImportFrom) and node.module == "stable_worldmodel.protocols")]
    exec(compile(helper_tree, str(package / "solver/utils.py"), "exec"), ns)
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and (node.level or node.module in ("loguru", "stable_worldmodel.solver.utils")))]
    import logging
    ns["logging"] = logging
    exec(compile(tree, str(source), "exec"), ns)
    return ns["CEMSolver"]


def main(package):
    from gymnasium.spaces import Box
    CEMSolver = load_solver(package)

    class ToyCost(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = []

        def get_cost(self, info, candidates):
            self.seen.append(candidates.detach().clone())
            return ((candidates - 0.37) ** 2).sum((-1, -2))

    def make():
        cost = ToyCost()
        solver = CEMSolver(cost, batch_size=2, num_samples=300, n_steps=7, topk=30, seed=17)
        solver.configure(action_space=Box(-1, 1, shape=(3, 2), dtype=np.float32), n_envs=3,
                         config=SimpleNamespace(horizon=5, action_block=5))
        return solver, cost

    original, oc = make()
    mean_solver, mc = make()
    best_solver, bc = make()
    mean_rec = attach_selection(mean_solver, "mean")
    best_rec = attach_selection(best_solver, "best")
    info = {"pixels": torch.zeros(3, 1, 1)}
    a, b, c = [solver.solve(info) for solver in (original, mean_solver, best_solver)]
    assert torch.equal(a["actions"], b["actions"])
    assert torch.equal(c["actions"], torch.cat(best_rec.best))
    assert torch.equal(a["actions"], torch.cat(mean_rec.means))
    assert not torch.equal(b["actions"], c["actions"])
    assert a["costs"] == b["costs"] == c["costs"]
    assert torch.equal(a["var"][0], b["var"][0]) and torch.equal(a["var"][0], c["var"][0])
    assert len(oc.seen) == len(mc.seen) == len(bc.seen) == 14
    assert all(torch.equal(x, y) and torch.equal(x, z) for x, y, z in zip(oc.seen, mc.seen, bc.seen))
    assert torch.equal(original.torch_gen.get_state(), mean_solver.torch_gen.get_state())
    assert torch.equal(original.torch_gen.get_state(), best_solver.torch_gen.get_state())
    # A second decision exercises callback.reset and history clearing.
    d, e = original.solve(info), mean_solver.solve(info)
    assert torch.equal(d["actions"], e["actions"])
    assert len(mean_rec.means) == 2
    assert len(mean_rec.costs) == 2

    # The post-search diagnostic uses a separate expanded info dictionary,
    # scores four plans without advancing the solver RNG, and leaves the
    # original plan intact. It is extra logging compute, not search budget.
    inspected, ic = make()
    captured = []

    def inspect(original_info, mean, best, elites, costs):
        before = inspected.torch_gen.get_state().clone()
        plans = torch.stack([mean, best, mean, best], dim=1)
        expanded = expanded_info(original_info, 4, "cpu")
        score = inspected.model.get_cost(expanded, plans)
        assert score.shape == (3, 4)
        assert expanded["pixels"].shape[:2] == (3, 4)
        assert torch.equal(before, inspected.torch_gen.get_state())
        captured.append(mean.clone())

    attach_selection(inspected, "mean", inspect)
    inspected_result = inspected.solve(info)
    assert len(ic.seen) == 15 and len(captured) == 1
    assert torch.equal(inspected_result["actions"], captured[0])
    assert set(info) == {"pixels"} and info["pixels"].shape == (3, 1, 1)

    class Scaler:
        def inverse_transform(self, x):
            return x * np.array([2., 3.]) + np.array([.1, .2])

        def transform(self, x):
            return (x - np.array([.1, .2])) / np.array([2., 3.])

    raw = np.ones((2, 5, 10), dtype=np.float32) * 5
    sent, applied, renorm = effective_plan("tworoom", raw, Scaler())
    assert np.max(applied) == 1 and np.max(sent) > 1 and np.max(renorm) < 1
    sent_pt, applied_pt, renorm_pt = effective_plan("pusht", raw, Scaler())
    np.testing.assert_array_equal(sent_pt, applied_pt)
    np.testing.assert_allclose(renorm_pt, raw)

    torch.manual_seed(47)
    W = torch.randn(2507, 192)
    emb, act = torch.randn(8, 3, 192), torch.randn(8, 3, 10)
    op = make_bilinear_predictor(W)
    actual = op(emb, act)
    for t in range(3):
        z0, z1, z2 = emb[:, t], emb[:, max(0, t - 1)], emb[:, max(0, t - 2)]
        expected = torch.cat([z0, z1, z2, act[:, t],
                              (z0[:, :, None] * act[:, t, None, :]).reshape(8, -1), torch.ones(8, 1)], 1) @ W
        torch.testing.assert_close(actual[:, t], expected, rtol=0, atol=0)
        torch.testing.assert_close(actual[:, :t + 1], op(emb[:, :t + 1], act[:, :t + 1]), rtol=0, atol=0)

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for task in TASKS:
            for seed in SEEDS:
                for model in MODELS:
                    for mode in ("mean", "best"):
                        count = 95 if model == "bilinear" else 80 if mode == "mean" else 90
                        rows = [dict(episode_id=i, start_step=0, goal_step=25, success=i < count) for i in range(100)]
                        path = root / "evaluation" / task / f"{model}_{mode}_seed{seed}" / "episodes.json"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(rows))
        result = analyze(root)
        assert result["primary_material_selection_interaction"]
        for values in result["results"]["tworoom"].values():
            assert values["reduction_of_bilinear_advantage_pp"] == 10
        # Swapping one best task's identity must invalidate episode pairing.
        path = root / "evaluation/tworoom/retrained_best_seed43/episodes.json"
        rows = json.loads(path.read_text())
        rows[0]["start_step"] = 1
        path.write_text(json.dumps(rows))
        try:
            analyze(root)
        except AssertionError:
            pass
        else:
            raise AssertionError("Mismatched episode identities accepted")
    print(json.dumps(dict(pass_=True, torch=torch.__version__, tests=[
        "mean hook equals original returned actions bit for bit",
        "same candidate tensors, reported costs, variance and solver RNG in both selection modes",
        "best action comes from the final scored elite set, reset works across decisions and batches",
        "four post-search diagnostic plans preserve solver RNG, original info and returned plan",
        "TwoRoom clamp and PushT identity mapped separately",
        "bilinear feature and short-context parity",
        "paired interaction analysis and task-mismatch rejection"]), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    main(parser.parse_args().package)
