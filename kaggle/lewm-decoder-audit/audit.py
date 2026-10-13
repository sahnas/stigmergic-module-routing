"""Exploratory audit of the unchanged experiment-23-v1 decoder; no fitting/planning."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

OUT = Path(os.environ.get("AUDIT_OUT", "/kaggle/working"))
INPUT = Path(os.environ.get("AUDIT_INPUT", "/kaggle/input"))
FROZEN = {
    "decoder.npy": "aeb3cd4c91df3de9a612d929d6d2494830aaea45afcc5085026244847d4b0442",
    "probe_split.npz": "939f3cfefa210f5e552f4ba37d389fdc16865d2052fa22bb4c0846b7fb21acfa",
    "decoder_manifest.json": "be317ef62e719fdf2ac812a018083e5e896a6f1ecb2377df3c079132c597e86c",
    "input_manifest.json": "567855e306e4b707895a58fa5b63a5b2d038a6efdf25bcfdab16dfd9d7020329",
    "complete.json": "c6c91a326984331515f8c79b47513d6c6d7d9576354cbe6cddf014bf38cfa4a1",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def locate(name, sha=None):
    paths = sorted(p for p in INPUT.rglob(name) if p.is_file())
    if sha:
        paths = [p for p in paths if digest(p) == sha]
    assert len(paths) == 1, f"Expected one input {name}; found {len(paths)}"
    return paths[0]


def decode(z, truth, indices, weights):
    predictions = []
    for start in range(0, len(indices), 8192):
        x = np.asarray(z[indices[start:start + 8192]], dtype=np.float32)
        assert np.isfinite(x).all()
        predictions.append(x @ weights[:-1] + weights[-1])
    predicted = np.concatenate(predictions)
    actual = np.asarray(truth[indices], dtype=np.float32)
    residual = predicted - actual
    assert np.isfinite(residual).all()
    return dict(indices=indices, actual=actual, predicted=predicted,
                residual=residual, distance=np.linalg.norm(residual, axis=1))


def stats(distance, residual=None, truth=None):
    d = np.asarray(distance)
    if not len(d):
        return dict(n=0)
    result = dict(n=len(d), mean=float(d.mean()), rmse=float(np.sqrt(np.square(d).mean())),
                  median=float(np.quantile(d, .5)), p90=float(np.quantile(d, .9)),
                  p95=float(np.quantile(d, .95)), p99=float(np.quantile(d, .99)),
                  maximum=float(d.max()), over4=int((d > 4).sum()), over8=int((d > 8).sum()),
                  over16=int((d > 16).sum()), rate_over8=float((d > 8).mean()))
    if residual is not None:
        result.update(bias_xy=residual.mean(0).tolist(), mae_xy=np.abs(residual).mean(0).tolist(),
                      rmse_xy=np.sqrt(np.square(residual).mean(0)).tolist())
    if truth is not None:
        result["r2_xy"] = (1 - np.square(residual).mean(0) / truth.var(0)).tolist()
    return result


def table(data, labels):
    rows = []
    for label in np.unique(labels):
        mask = labels == label
        rows.append(dict(group=int(label), **stats(data["distance"][mask], data["residual"][mask])))
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def spatial_labels(xy, bins=16):
    cell = np.clip(np.floor(xy / (224 / bins)).astype(int), 0, bins - 1)
    return cell[:, 1] * bins + cell[:, 0]


def independent_worst(distance, episode, count):
    seen, selected = set(), []
    for i in np.argsort(-distance, kind="stable"):
        e = int(episode[i])
        if e not in seen:
            seen.add(e)
            selected.append(int(i))
        if len(selected) == count:
            break
    return selected


def red_centroid(pixels):
    rgb = np.asarray(pixels, dtype=np.float64)
    mass = np.maximum(rgb[..., 0] - np.maximum(rgb[..., 1], rgb[..., 2]), 0)
    if mass.sum() <= 0:
        return None
    y, x = np.indices(mass.shape)
    return np.array([(x * mass).sum(), (y * mass).sum()]) / mass.sum()


def figures(validation, training, grid, phase, episodes, selected, pixels, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 150, "axes.spines.top": False,
                         "axes.spines.right": False})
    d = validation["distance"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    for label, values, color in (("Training (frozen selection)", training["distance"], "#597d9d"),
                                  ("Validation", d, "#c65131")):
        ordered = np.sort(values)
        axes[0].plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered), label=label, color=color)
    axes[0].axvline(8, ls="--", color="black", label="Prespecified limit: 8")
    axes[0].axhline(.95, ls=":", color="gray")
    axes[0].set(xlim=(0, max(16, float(np.quantile(d, .999)))), ylim=(0, 1.01),
                xlabel="Euclidean error (native units)", ylabel="Cumulative fraction", title="Error distribution")
    axes[0].legend(fontsize=8)
    episode_rows = sorted(episodes, key=lambda r: r["rate_over8"], reverse=True)
    axes[1].bar(np.arange(len(episode_rows)), [100 * r["rate_over8"] for r in episode_rows], color="#376c85")
    axes[1].axhline(100 * np.mean(d > 8), color="#c65131", ls="--", label="All validation frames")
    axes[1].set(xlabel="Held-out episode (ordered by rate)", ylabel="Frames with error > 8 (%)",
                title=f"{len(episode_rows)} held-out episodes; serial dependence")
    axes[1].legend(fontsize=8)
    fig.savefig(out / "distribution_episodes.png"); plt.close(fig)

    maps = {key: np.full((16, 16), np.nan) for key in ("n", "mean", "rate_over8", "dx", "dy")}
    for row in grid:
        y, x = divmod(row["group"], 16)
        for key in ("n", "mean", "rate_over8"):
            maps[key][y, x] = row[key]
        maps["dx"][y, x], maps["dy"][y, x] = row["bias_xy"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    specs = [("n", "Validation occupancy", "viridis"), ("mean", "Mean error and mean residual", "magma"),
             ("rate_over8", "Fraction with error > 8", "magma")]
    for ax, (key, title, cmap) in zip(axes, specs):
        values = maps[key].copy()
        if key != "n": values[maps["n"] < 30] = np.nan
        kwargs = {"norm": LogNorm(vmin=1, vmax=np.nanmax(maps["n"]))} if key == "n" else {}
        im = ax.imshow(values, origin="upper", extent=(0, 224, 224, 0), cmap=cmap, **kwargs)
        fig.colorbar(im, ax=ax, shrink=.75)
        ax.set(xlabel="True x", ylabel="True y", title=title, xlim=(0, 224), ylim=(224, 0))
    centers = np.arange(7, 224, 14); xx, yy = np.meshgrid(centers, centers)
    good = maps["n"] >= 30
    axes[1].quiver(xx[good], yy[good], maps["dx"][good], maps["dy"][good], color="#4fe3d1",
                   angles="xy", scale_units="xy", scale=1, width=.006)
    fig.suptitle("Fixed 14-unit cells; error maps omit cells with fewer than 30 frames")
    fig.savefig(out / "spatial_errors.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for ax, coordinate, rows in zip(axes, ("x", "y"), (phase["x"], phase["y"])):
        idx = 0 if coordinate == "x" else 1
        ax.bar([r["group"] + .5 for r in rows], [r["bias_xy"][idx] for r in rows], color="#597d9d")
        ax.axhline(0, color="black", lw=.8)
        ax.set(xlabel=f"True {coordinate} modulo 14 (unit-width bins)", ylabel=f"Mean signed {coordinate} error",
               title=f"Patch-phase association: {coordinate}")
    fig.suptitle("Descriptive association; spatial occupancy can confound patch phase")
    fig.savefig(out / "patch_phase.png"); plt.close(fig)

    for name, indices in selected.items():
        fig, axes = plt.subplots(4, 4, figsize=(12, 13), constrained_layout=True)
        for ax in axes.flat: ax.axis("off")
        for ax, i in zip(axes.flat, indices):
            row = int(validation["indices"][i])
            ax.imshow(pixels[row])
            true, predicted = validation["actual"][i], validation["predicted"][i]
            ax.plot(*true, marker="+", ms=11, color="#12e7db", mew=1.7)
            ax.plot(*predicted, marker="x", ms=9, color="#ff22ce", mew=1.7)
            ax.plot([true[0], predicted[0]], [true[1], predicted[1]], color="#ff22ce", lw=.8)
            ax.set_title(f"ep {validation['episode'][i]} / step {validation['step'][i]}\nerror {d[i]:.2f}", fontsize=9)
        fig.suptitle(f"{name.replace('_', ' ')} | cyan +: label; magenta x: frozen decoder", fontsize=13)
        fig.savefig(out / f"images_{name}.png"); plt.close(fig)

    worst_eps = [int(validation["episode"][i]) for i in selected["worst_distinct_episodes"][:8]]
    fig, axes = plt.subplots(4, 2, figsize=(12, 10), constrained_layout=True)
    for ax, ep in zip(axes.flat, worst_eps):
        mask = validation["episode"] == ep
        ax.plot(validation["step"][mask], d[mask], color="#376c85", lw=1)
        ax.axhline(8, color="#c65131", ls="--")
        ax.set(title=f"Episode {ep}", xlabel="Step", ylabel="Error")
    fig.suptitle("Error traces of episodes containing the eight largest distinct-episode errors")
    fig.savefig(out / "worst_episode_traces.png"); plt.close(fig)


def run():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "audit_summary.json").exists(), "Use a fresh output directory"
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hdf5plugin==7.1.0"], check=True)
    import h5py
    import hdf5plugin  # noqa: F401
    from PIL import Image
    frozen = {name: locate(name, sha) for name, sha in FROZEN.items()}
    dm = json.loads(frozen["decoder_manifest.json"].read_text())
    original = json.loads(frozen["input_manifest.json"].read_text())
    assert json.loads(frozen["complete.json"].read_text())["status"] == "stopped_decoder_gate"
    weights = np.load(frozen["decoder.npy"], allow_pickle=False)
    split = np.load(frozen["probe_split.npz"], allow_pickle=False)
    assert weights.shape == (193, 2) and np.isfinite(weights).all()
    cache = {name: locate(name, sha) for name, sha in original["cache_hashes"].items()}
    dataset = locate("tworoom.h5", original["dataset_sha256"])
    print("Input hashes verified", flush=True)
    z = np.load(cache["emb_tworoom.npy"], mmap_mode="r", allow_pickle=False)
    position = np.load(cache["pos_agent.npy"], mmap_mode="r", allow_pickle=False)
    episode = np.load(cache["ep_idx.npy"], mmap_mode="r", allow_pickle=False).reshape(-1)
    train, valid = split["train_indices"], split["validation_indices"]
    assert set(episode[train]).isdisjoint(episode[valid])
    assert set(episode[valid]).isdisjoint(split["excluded_episodes"])
    assert set(episode[train]).isdisjoint(split["excluded_episodes"])
    training, validation = [decode(z, position, indices, weights) for indices in (train, valid)]
    train_stats, valid_stats = [stats(x["distance"], x["residual"], x["actual"]) for x in (training, validation)]
    np.testing.assert_allclose(valid_stats["r2_xy"], dm["metrics"]["r2"], rtol=0, atol=2e-6)
    for metric in ("median", "p95"):
        assert abs(valid_stats[metric] - dm["metrics"][metric]) <= 2e-4, f"Metric parity failed: {metric}"
    print("Frozen validation metrics reproduced", json.dumps(valid_stats), flush=True)
    xy, d, residual = validation["actual"], validation["distance"], validation["residual"]
    validation["episode"] = np.asarray(episode[valid])
    with h5py.File(dataset, "r") as f:
        assert f["pixels"].shape[1:] == (224, 224, 3)
        np.testing.assert_array_equal(f["pos_agent"][:], position)
        validation["step"] = f["step_idx"][:].reshape(-1)[valid]
        observations = f["observation"][:][valid]
        np.testing.assert_allclose(observations[:, :2], xy, rtol=0, atol=1e-4)
        validation["target"] = f["pos_target"][:][valid]
        distance_edge = np.minimum(xy, 224 - xy).min(1)
        validation["distance_canvas_edge"] = distance_edge
        regions = {
            "outer_band_28": distance_edge < 28,
            "inner_region_28": distance_edge >= 28,
            "left_half": xy[:, 0] < 112, "right_half": xy[:, 0] >= 112,
            "top_half": xy[:, 1] < 112, "bottom_half": xy[:, 1] >= 112,
            "central_x_band_16": np.abs(xy[:, 0] - 112) < 16,
            "outside_central_x_band_16": np.abs(xy[:, 0] - 112) >= 16,
        }
        geometry = dict(observation_shape=list(observations.shape), h5_keys=list(f.keys()))
        if observations.shape[1] == 10:
            doors = observations[:, 4:].reshape(-1, 3, 2)
            active = np.any(doors != 0, axis=2)
            distances = np.linalg.norm(doors - xy[:, None], axis=2)
            distances[~active] = np.inf
            closest = distances.min(1)
            if np.isfinite(closest).all():
                validation["distance_door_center"] = closest
                regions.update(near_door_center_32=closest < 32, away_from_door_center_32=closest >= 32)
            geometry["unique_door_layouts"] = np.unique(observations[:, 4:], axis=0).tolist()
        region_stats = {name: stats(d[mask], residual[mask]) for name, mask in regions.items()}
        grid = table(validation, spatial_labels(xy))
        episode_rows = table(validation, validation["episode"])
        phase = {name: table(validation, np.floor(xy[:, i] % 14).astype(int)) for i, name in enumerate(("x", "y"))}
        ranked = sorted(episode_rows, key=lambda row: row["over8"], reverse=True)
        total_tail = max(1, int((d > 8).sum()))
        concentration = {str(n): dict(episodes=min(n, len(ranked)), tail_share=sum(r["over8"] for r in ranked[:n]) / total_tail,
                                     frame_share=sum(r["n"] for r in ranked[:n]) / len(d)) for n in (1, 5, 10)}
        cells4 = spatial_labels(xy, bins=4)
        tail = np.flatnonzero(d > 8)
        selected = {
            "worst_distinct_episodes": independent_worst(d, validation["episode"], 16),
            "worst_per_spatial_quadrant": [int(np.flatnonzero(cells4 == cell)[np.argmax(d[cells4 == cell])]) for cell in np.unique(cells4)],
            "random_tail": sorted(np.random.default_rng(23023).choice(tail, min(16, len(tail)), replace=False).tolist()),
            "random_validation": sorted(np.random.default_rng(23024).choice(len(d), min(16, len(d)), replace=False).tolist()),
        }
        selected_indices = sorted(set(i for indices in selected.values() for i in indices))
        pixels = {int(valid[i]): f["pixels"][int(valid[i])] for i in selected_indices}
        image_dir = OUT / "sample_images"; image_dir.mkdir()
        image_rows = []
        for i in selected_indices:
            row = int(valid[i]); px = pixels[row]
            Image.fromarray(px).save(image_dir / f"row_{row}.png")
            centroid = red_centroid(px)
            image_rows.append(dict(row=row, episode=int(validation["episode"][i]), step=int(validation["step"][i]),
                                   true_xy=xy[i].tolist(), decoded_xy=validation["predicted"][i].tolist(),
                                   error=float(d[i]), red_centroid=None if centroid is None else centroid.tolist(),
                                   red_centroid_label_error=None if centroid is None else float(np.linalg.norm(centroid - xy[i])),
                                   selections=[name for name, indices in selected.items() if i in indices]))
        save(OUT / "image_selection.json", image_rows)
        figures(validation, training, grid, phase, episode_rows, selected, pixels, OUT)
    np.savez_compressed(OUT / "validation_residuals.npz", **validation)
    write_csv(OUT / "spatial_cells.csv", grid)
    write_csv(OUT / "episodes.csv", episode_rows)
    for name, rows in phase.items(): write_csv(OUT / f"phase_{name}.csv", rows)
    summary = dict(status="audit_complete", purpose="Post-gate exploratory audit, not confirmatory evaluation",
                   original_decoder_unchanged=True, decoder_sha256=digest(frozen["decoder.npy"]),
                   split_sha256=digest(frozen["probe_split.npz"]), training=train_stats, validation=valid_stats,
                   validation_episode_count=len(episode_rows), episode_equal_weight_rate_over8=float(np.mean([r["rate_over8"] for r in episode_rows])),
                   tail_concentration=concentration, regions=region_stats, geometry=geometry,
                   top_10_cells_by_tail_count=sorted(grid, key=lambda r: r["over8"], reverse=True)[:10],
                   sample_red_centroid_label_error=stats(np.array([r["red_centroid_label_error"] for r in image_rows if r["red_centroid_label_error"] is not None])),
                   image_selection_count=len(image_rows), seconds=time.monotonic() - started,
                   caveats=["Frame errors are serially dependent; no frame-independent significance tests.",
                            "Spatial and phase associations are descriptive, not causal mechanisms.",
                            "Image examples include deliberately selected extremes; separate random panels are included.",
                            "Red-excess centroid is a sampled rendering/label sanity check, assuming a red agent.",
                            "No fit, alternative decoder, threshold change, planner or predictor evaluation."])
    save(OUT / "audit_summary.json", summary)
    shutil.copyfile(__file__, OUT / "submitted_audit.py")
    save(OUT / "audit_inputs.json", dict(source_sha256=digest(__file__), frozen_inputs=FROZEN,
         cache_sha256=original["cache_hashes"], dataset_sha256=original["dataset_sha256"], numpy=np.__version__, h5py=h5py.__version__))
    print(json.dumps(summary, indent=2), flush=True)


def checks():
    xy = np.array([[1., 2.], [223., 223.], [14., 14.]], dtype=np.float32)
    assert spatial_labels(xy).tolist() == [0, 255, 17]
    assert independent_worst(np.array([3., 9., 7., 6.]), np.array([0, 0, 1, 2]), 3) == [1, 2, 3]
    s = stats(np.array([0., 5., 10., 20.]))
    assert (s["median"], s["over8"], s["over16"]) == (7.5, 2, 1)
    z = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    p = np.array([[2., 0.], [0., 3.], [7., 11.]], dtype=np.float32)
    truth = np.array([[9., 17.], [13., 23.]], dtype=np.float32)
    result = decode(z, truth, np.array([1, 0]), p)
    np.testing.assert_array_equal(result["distance"], [0., 0.])
    px = np.zeros((15, 15, 3), dtype=np.uint8); px[8, 3, 0] = 255
    np.testing.assert_array_equal(red_centroid(px), [3., 8.])
    assert red_centroid(np.ones_like(px) * 255) is None
    print(json.dumps(dict(pass_=True, checks=["spatial bin boundaries", "distinct-episode selection",
          "error counts and quantiles", "affine decoding and row order", "pixel x/y centroid convention"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        checks()
    else:
        try:
            run()
        except BaseException as error:
            OUT.mkdir(parents=True, exist_ok=True)
            save(OUT / "audit_failure.json", dict(type=type(error).__name__, message=str(error)))
            raise
