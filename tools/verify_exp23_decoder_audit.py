"""Verify retrieved frozen-decoder audit outputs without the private cache."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/lewm_kaggle/exp23_decoder_audit"
RAW = RESULTS / "version1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    summary = json.loads((RAW / "audit_summary.json").read_text())
    inputs = json.loads((RAW / "audit_inputs.json").read_text())
    submission = json.loads((RESULTS / "submission_v1.json").read_text())
    freeze = json.loads((ROOT / "kaggle/lewm-decoder-audit/freeze.json").read_text())
    original = ROOT / "results/lewm_kaggle/exp23_objective/version1"
    assert summary["status"] == "audit_complete"
    assert summary["original_decoder_unchanged"] is True
    for name, expected in freeze["files"].items():
        assert digest(ROOT / "kaggle/lewm-decoder-audit" / name) == expected
    assert digest(RAW / "submitted_audit.py") == submission["source_sha256"] == inputs["source_sha256"]
    for name, expected in inputs["frozen_inputs"].items():
        assert digest(original / name) == expected
    original_inputs = json.loads((original / "input_manifest.json").read_text())
    assert inputs["cache_sha256"] == original_inputs["cache_hashes"]
    assert inputs["dataset_sha256"] == original_inputs["dataset_sha256"]

    data = np.load(RAW / "validation_residuals.npz", allow_pickle=False)
    split = np.load(original / "probe_split.npz", allow_pickle=False)
    rows = data["indices"]
    np.testing.assert_array_equal(rows, split["validation_indices"])
    assert len(np.unique(rows)) == len(rows)
    assert set(data["episode"]).isdisjoint(split["excluded_episodes"])
    actual, predicted = data["actual"], data["predicted"]
    assert np.isfinite(actual).all() and np.isfinite(predicted).all()
    residual = predicted.astype(np.float64) - actual.astype(np.float64)
    np.testing.assert_allclose(residual, data["residual"], rtol=0, atol=2e-5)
    distance = np.hypot(residual[:, 0], residual[:, 1])
    np.testing.assert_allclose(distance, data["distance"], rtol=0, atol=2e-5)
    reported = summary["validation"]
    assert len(rows) == reported["n"] == 89618
    for key, quantile in (("median", .5), ("p90", .9), ("p95", .95), ("p99", .99)):
        np.testing.assert_allclose(np.quantile(distance, quantile), reported[key], rtol=0, atol=2e-5)
    for threshold in (4, 8, 16):
        assert int((distance > threshold).sum()) == reported[f"over{threshold}"]
    # Float64 accumulation intentionally differs from the original fp32 reporting.
    r2 = 1 - np.mean(residual ** 2, axis=0) / actual.astype(np.float64).var(axis=0)
    np.testing.assert_allclose(r2, reported["r2_xy"], rtol=0, atol=2e-6)
    unique_episodes = np.unique(data["episode"])
    assert len(unique_episodes) == summary["validation_episode_count"] == 969
    counts = [(int((data["episode"] == ep).sum()), int(((data["episode"] == ep) & (distance > 8)).sum()))
              for ep in unique_episodes]
    counts.sort(key=lambda pair: pair[1], reverse=True)
    for n in (1, 5, 10):
        group = summary["tail_concentration"][str(n)]
        np.testing.assert_allclose(sum(t for _, t in counts[:n]) / reported["over8"], group["tail_share"])
        np.testing.assert_allclose(sum(c for c, _ in counts[:n]) / len(rows), group["frame_share"])

    edge = np.minimum(actual, 224 - actual).min(axis=1)
    regions = {"outer_band_28": edge < 28, "inner_region_28": edge >= 28,
               "left_half": actual[:, 0] < 112, "right_half": actual[:, 0] >= 112,
               "top_half": actual[:, 1] < 112, "bottom_half": actual[:, 1] >= 112,
               "central_x_band_16": np.abs(actual[:, 0] - 112) < 16,
               "outside_central_x_band_16": np.abs(actual[:, 0] - 112) >= 16}
    if "distance_door_center" in data:
        regions.update(near_door_center_32=data["distance_door_center"] < 32,
                       away_from_door_center_32=data["distance_door_center"] >= 32)
    for name, mask in regions.items():
        reference = summary["regions"][name]
        assert int(mask.sum()) == reference["n"]
        assert int((distance[mask] > 8).sum()) == reference["over8"]
        assert int((distance[mask] > 16).sum()) == reference["over16"]
        np.testing.assert_allclose(np.quantile(distance[mask], .95), reference["p95"], rtol=0, atol=2e-5)

    images = json.loads((RAW / "image_selection.json").read_text())
    assert len(images) == summary["image_selection_count"]
    lookup = {int(row): i for i, row in enumerate(rows)}
    for item in images:
        i = lookup[item["row"]]
        np.testing.assert_array_equal(item["true_xy"], actual[i])
        np.testing.assert_array_equal(item["decoded_xy"], predicted[i])
        assert item["episode"] == int(data["episode"][i])
        assert item["step"] == int(data["step"][i])
        rgb = np.asarray(Image.open(RAW / "sample_images" / f"row_{item['row']}.png"), dtype=float)
        assert rgb.shape == (224, 224, 3)
        red = np.maximum(rgb[:, :, 0] - rgb[:, :, 1:].max(axis=2), 0)
        if red.sum():
            centroid = [np.dot(red.sum(axis=0), np.arange(224)) / red.sum(),
                        np.dot(red.sum(axis=1), np.arange(224)) / red.sum()]
            np.testing.assert_allclose(centroid, item["red_centroid"], rtol=0, atol=1e-9)
    # These groups were added after viewing the selected extreme images.
    # They are explicitly exploratory and never change the feasibility gate.
    temporal = {}
    step = data["step"]
    for name, mask in (("step_0", step == 0), ("after_step_0", step > 0),
                       ("steps_0_to_9", step < 10), ("steps_10_plus", step >= 10)):
        temporal[name] = {"n": int(mask.sum()), "p95": float(np.quantile(distance[mask], .95)),
                          "over8": int((distance[mask] > 8).sum()),
                          "rate_over8": float((distance[mask] > 8).mean()),
                          "over16": int((distance[mask] > 16).sum())}
    print(json.dumps({"verified": True, "validation_frames": len(rows),
                      "episodes": len(unique_episodes), "sample_images": len(images),
                      "p95_recomputed_float64": float(np.quantile(distance, .95)),
                      "episodes_with_error_over8": int(len(np.unique(data["episode"][distance > 8]))),
                      "posthoc_after_image_inspection": {"temporal_groups": temporal},
                      "scope": "Saved predictions, source and input provenance, split, summary, selected pixels; private embeddings not re-decoded locally"}, indent=2))


if __name__ == "__main__":
    main()
