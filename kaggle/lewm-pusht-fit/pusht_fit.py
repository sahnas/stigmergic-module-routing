"""Experiment 19: validate the existing cache, fit ridge, retrain the predictor.

No planning evaluation, parameter search, or change to preregistration 19.
All scientific outputs are written to /kaggle/working; raw images stay on tmp.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

MODEL_REV = "22b330c28c27ead4bfd1888615af1340e3fe9052"
DATA_REV = "655cd446b992"
CODE_REV = "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"
MODEL_ID = "quentinll/lewm-pusht"
OUT = Path("/kaggle/working")
TMP = Path("/kaggle/tmp/exp19")
PARITY_ATOL = 1e-5  # fixed before this validation run; fp32, absolute max error


def save_json(name, value):
    path = OUT / name
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def log(event, **value):
    row = dict(event=event, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **value)
    text = json.dumps(row, allow_nan=False)
    print(text, flush=True)
    with (OUT / "progress.jsonl").open("a") as f:
        f.write(text + "\n")


def command(args):
    print("$ " + " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def episode_windows(lengths, offsets, history, frameskip, rng):
    """Same seed-0 episode split and valid start convention as experiment 18."""
    import numpy as np
    lengths, offsets = np.asarray(lengths), np.asarray(offsets)
    assert lengths.ndim == offsets.ndim == 1 and lengths.shape == offsets.shape
    assert np.all(lengths > 0)
    assert np.array_equal(offsets, np.r_[0, np.cumsum(lengths[:-1], dtype=np.int64)])
    test_ids = rng.choice(len(lengths), size=len(lengths) // 10, replace=False)
    test_set = set(test_ids.tolist())
    train, test = [], []
    for e, (length, offset) in enumerate(zip(lengths, offsets)):
        starts = np.arange(int(offset), int(offset) + max(0, int(length) - history * frameskip), dtype=np.int64)
        (test if e in test_set else train).append(starts)
    return np.concatenate(train), np.concatenate(test), test_ids


def features(emb, act):
    """Causal experiment-18 features, with left repetition for short contexts."""
    import torch
    b, t, _ = emb.shape
    z0 = emb[:, -1]
    z1 = emb[:, -2] if t >= 2 else z0
    z2 = emb[:, -3] if t >= 3 else z1
    a = act[:, -1]
    interaction = (z0.unsqueeze(2) * a.unsqueeze(1)).reshape(b, -1)
    return torch.cat([z0, z1, z2, a, interaction, torch.ones(b, 1, device=emb.device, dtype=emb.dtype)], dim=1)


def bootstrap():
    OUT.mkdir(exist_ok=True, parents=True)
    TMP.mkdir(exist_ok=True, parents=True)
    os.environ.update(STABLEWM_HOME=str(TMP / "stablewm"), HF_HOME=str(TMP / "hf"),
                      WANDB_MODE="disabled", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
                      PYTHONUNBUFFERED="1")
    command(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    command(["apt-get", "install", "-y", "-qq", "zstd"])
    command([sys.executable, "-m", "pip", "install", "-q", "stable-worldmodel[train,format]==0.1.1",
             "stable-pretraining==0.1.8", "transformers==4.57.6", "hydra-core"])
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (OUT / "environment.txt").write_text(freeze)
    upstream = TMP / "le-wm"
    command(["git", "clone", "--quiet", "--no-checkout", "https://github.com/lucas-maes/le-wm.git", upstream])
    command(["git", "-C", upstream, "checkout", "--quiet", "--detach", CODE_REV])
    assert subprocess.check_output(["git", "-C", upstream, "rev-parse", "HEAD"], text=True).strip() == CODE_REV
    shutil.copyfile(upstream / "eval.py", OUT / "official_eval.py")
    shutil.copytree(upstream / "config" / "eval", OUT / "official_eval_config")
    return upstream


def main():
    t0 = time.monotonic()
    upstream = bootstrap()
    import copy
    import h5py
    import hdf5plugin  # registers HDF5 compression filters
    import hydra
    from huggingface_hub import hf_hub_download
    import numpy as np
    from omegaconf import OmegaConf
    from sklearn.preprocessing import StandardScaler
    import stable_worldmodel as swm
    import torch

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    dev = torch.device("cuda")
    cache_paths = list(Path("/kaggle/input").rglob("emb_pusht.npy"))
    assert len(cache_paths) == 1, cache_paths
    cache = cache_paths[0].parent
    cache_meta = json.loads((cache / "pusht_cache_meta.json").read_text())
    emb = np.load(cache / "emb_pusht.npy", mmap_mode="r")
    act = np.load(cache / "action.npy")
    lengths, offsets = [np.load(cache / f"{k}.npy") for k in ("ep_len", "ep_offset")]
    assert emb.dtype == np.float32 and emb.ndim == 2
    assert len(emb) == int(lengths.sum()) == cache_meta["frames"]
    assert len(lengths) == cache_meta["episodes"]
    assert act.shape[0] == len(emb) and act.ndim == 2
    assert cache_meta["model_revision"] == MODEL_REV
    for start in range(0, len(emb), 65536):
        assert np.isfinite(emb[start:start + 65536]).all(), f"Nonfinite cache near {start}"
    log("cache_found", frames=len(emb), episodes=len(lengths), embedding_dim=emb.shape[1])

    # Download the precise checkpoint, rather than relying on the library's /main loader.
    released_dir = OUT / "pusht_released"
    for filename in ("config.json", "weights.pt"):
        hf_hub_download(MODEL_ID, filename, revision=MODEL_REV, local_dir=released_dir)
    cfg = json.loads((released_dir / "config.json").read_text())
    eval_cfg = OmegaConf.load(upstream / "config/eval/pusht.yaml")
    history = int(cfg["predictor"]["num_frames"])
    fs = int(eval_cfg.plan_config.action_block)
    dim = int(cfg["predictor"]["input_dim"])
    action_dim = int(cfg["action_encoder"]["input_dim"])
    assert (history, fs, dim, action_dim) == (3, 5, 192, 10)
    assert emb.shape[1] == dim and fs * act.shape[1] == action_dim
    released = swm.wm.utils.load_pretrained(str(released_dir)).to(dev).eval()
    released.requires_grad_(False)

    log("dataset_download_started", revision=DATA_REV)
    archive = hf_hub_download(MODEL_ID, "pusht_expert_train.h5.zst", repo_type="dataset",
                              revision=DATA_REV, local_dir=TMP / "download")
    dataset = TMP / "stablewm/datasets/pusht_expert_train.h5"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    command(["zstd", "-d", "-q", archive, "-o", dataset])
    Path(archive).unlink()
    log("dataset_ready", bytes=dataset.stat().st_size)

    # Independent reference: import the actual pinned eval.py img_transform function.
    spec = importlib.util.spec_from_file_location("exp19_official_eval", upstream / "eval.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    transform = official.img_transform(eval_cfg)
    idx = np.arange(1000, 1064)  # same 64 frames as the cache's original sanity check
    with h5py.File(dataset, "r") as f:
        assert tuple(f["pixels"].shape) == tuple(cache_meta["pixels_shape"])
        assert np.array_equal(f["ep_len"][:], lengths)
        assert np.array_equal(f["ep_offset"][:], offsets)
        assert np.array_equal(f["action"][:], act, equal_nan=True)
        raw = f["pixels"][idx]
        px = torch.stack([transform(image) for image in raw]).to(dev)
        with torch.no_grad():
            reference = released.encode({"pixels": px.unsqueeze(1)})["emb"][:, 0].cpu().numpy()
    delta = np.abs(reference - emb[idx])
    parity = dict(max_abs=float(delta.max()), mean_abs=float(delta.mean()), atol=PARITY_ATOL,
                  indices=idx.tolist(), pass_=bool(delta.max() <= PARITY_ATOL),
                  reference="pinned eval.py img_transform then pinned model.encode")
    save_json("cache_validation.json", parity)
    log("cache_parity", **parity)
    assert parity["pass_"], "Independent preprocessing parity failed; stop before fitting"

    rng = np.random.default_rng(0)
    tr, te, test_ids = episode_windows(lengths, offsets, history, fs, rng)
    assert len(tr) > 0 and len(te) > 0
    # Match eval.py exactly: fit on all dataset actions, excluding rows with NaN.
    finite_rows = ~np.isnan(act).any(axis=1)
    assert np.isfinite(act[finite_rows]).all()
    scaler = StandardScaler().fit(act[finite_rows])
    actn = scaler.transform(act).astype(np.float32)
    np.savez(OUT / "action_normalization.npz", mean=scaler.mean_, scale=scaler.scale_)
    np.save(OUT / "heldout_episode_ids.npy", test_ids)
    # Every action consumed by a window must be finite; never silently repair inner NaNs.
    invalid = (~np.isfinite(actn).all(axis=1)).astype(np.int64)
    cumulative = np.r_[0, np.cumsum(invalid)]
    for starts in (tr, te):
        assert np.all(cumulative[starts + history * fs] == cumulative[starts]), "Invalid action in a training/test window"
    # Boundary actions not used by windows may contain NaN, as in the official data.
    actn = np.nan_to_num(actn)
    E = torch.from_numpy(np.array(emb)).to(dev)
    A = torch.from_numpy(actn).to(dev)
    del emb, actn, act

    provenance = dict(model_id=MODEL_ID, model_revision=MODEL_REV, dataset_revision=DATA_REV,
                      upstream_revision=CODE_REV, cache_metadata=cache_meta, cache_parity=parity,
                      script_sha256=sha256(__file__), cache_sha256=sha256(cache / "emb_pusht.npy"),
                      weights_sha256=sha256(released_dir / "weights.pt"), config_sha256=sha256(released_dir / "config.json"),
                      history=history, frameskip=fs, embedding_dim=dim, action_block_dim=action_dim,
                      train_windows=len(tr), heldout_windows=len(te), heldout_episodes=len(test_ids),
                      split_seed=0, init_seed=1, normalization="eval.py StandardScaler, full dataset valid rows",
                      action_mean=scaler.mean_.tolist(), action_scale=scaler.scale_.tolist(),
                      note="Cache source fetched main; this run checks cached embeddings against the pinned checkpoint.")
    save_json("provenance.json", provenance)
    log("windows_ready", train=len(tr), heldout=len(te), heldout_episodes=len(test_ids))

    def window(indices):
        ix = torch.as_tensor(indices, device=dev)
        h = torch.stack([E[ix + k * fs] for k in range(history)], dim=1)
        a = torch.stack([torch.cat([A[ix + k * fs + j] for j in range(fs)], dim=1)
                         for k in range(history)], dim=1)
        y = torch.stack([E[ix + (k + 1) * fs] for k in range(history)], dim=1)
        return h, a, y

    @torch.no_grad()
    def errors(predict_last, batch_size=1024):
        se = torch.zeros(history, device=dev, dtype=torch.float64)
        sums = torch.zeros(history, dim, device=dev, dtype=torch.float64)
        squares = torch.zeros_like(sums)
        n = 0
        for i in range(0, len(te), batch_size):
            h, a, y = window(te[i:i + batch_size])
            for L in range(1, history + 1):
                target = y[:, L - 1].double()
                pred = predict_last(h[:, :L], a[:, :L]).double()
                se[L - 1] += ((pred - target) ** 2).sum()
                sums[L - 1] += target.sum(0)
                squares[L - 1] += (target ** 2).sum(0)
            n += len(h)
        variance = ((squares - sums ** 2 / n) / (n - 1)).mean(1)
        assert (variance > 0).all()
        return {f"context_{L}": float((se[L - 1] / (n * dim) / variance[L - 1]).item())
                for L in range(1, history + 1)}

    def neural(m, h, a):
        return m.predict(h, m.action_encoder(a))

    # Identical feature order, fp64 sufficient statistics, ridge convention and bias penalty to experiment 18.
    dx = history * dim + action_dim + dim * action_dim + 1
    xtx = torch.zeros(dx, dx, device=dev, dtype=torch.float64)
    xty = torch.zeros(dx, dim, device=dev, dtype=torch.float64)
    fit_started = time.monotonic()
    for i in range(0, len(tr), 8192):
        h, a, y = window(tr[i:i + 8192])
        x, target = features(h, a).double(), y[:, -1].double()
        xtx += x.T @ x
        xty += x.T @ target
        if i % (8192 * 32) == 0:
            log("ridge_progress", windows=min(i + 8192, len(tr)), total=len(tr))
    lam = 1e-2 * len(tr)
    system = xtx + lam * torch.eye(dx, device=dev, dtype=torch.float64)
    w64 = torch.linalg.solve(system, xty)
    residual = float((torch.linalg.norm(system @ w64 - xty) / torch.linalg.norm(xty)).item())
    assert torch.isfinite(w64).all() and residual < 1e-8
    W = w64.float()
    torch.save(dict(W=W.cpu(), mu=torch.tensor(scaler.mean_), sd=torch.tensor(scaler.scale_),
                    model_revision=MODEL_REV, history=history, frameskip=fs), OUT / "pusht_bilinear.pt")
    del xtx, xty, system, w64, x, target, h, a, y
    torch.cuda.empty_cache()
    context_errors = {"bilinear": errors(lambda h, a: features(h, a) @ W),
                      "released": errors(lambda h, a: neural(released, h, a)[:, -1])}
    save_json("bilinear_meta.json", dict(features=dx, ridge=lam, solve_relative_residual=residual,
                                         context_errors=context_errors, seconds=time.monotonic() - fit_started))
    log("bilinear_complete", context_errors=context_errors, seconds=time.monotonic() - fit_started)

    # Only these three modules train; encoder and projector remain the released ones.
    new = copy.deepcopy(released)
    torch.manual_seed(1)
    for name in ("action_encoder", "predictor", "pred_proj"):
        setattr(new, name, hydra.utils.instantiate(cfg[name]).to(dev))
    params = [p for name in ("action_encoder", "predictor", "pred_proj") for p in getattr(new, name).parameters()]
    assert all(p.requires_grad for p in params)
    assert not any(p.requires_grad for p in new.encoder.parameters())
    assert not any(p.requires_grad for p in new.projector.parameters())
    batch_size, epochs = 256, 8
    steps_per_epoch = (len(tr) + batch_size - 1) // batch_size
    optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4,
                                                   total_steps=epochs * steps_per_epoch, pct_start=0.05)
    checkpoint_dir = OUT / "pusht_retrained_v2"
    checkpoint_dir.mkdir(exist_ok=True)
    shutil.copyfile(released_dir / "config.json", checkpoint_dir / "config.json")
    train_started = time.monotonic()
    epoch_records = []
    for epoch in range(epochs):
        epoch_started = time.monotonic()
        for name in ("action_encoder", "predictor", "pred_proj"):
            getattr(new, name).train()
        order = rng.permutation(len(tr))
        loss_sum = 0.0
        count = 0
        for step, i in enumerate(range(0, len(tr), batch_size)):
            h, a, target = window(tr[order[i:i + batch_size]])
            prediction = neural(new, h, a)
            loss = ((prediction - target) ** 2).mean()  # all positions, including contexts 1 and 2
            assert torch.isfinite(loss), "Nonfinite neural training loss"
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.detach()) * len(h)
            count += len(h)
            if step % 500 == 0:
                log("neural_progress", epoch=epoch + 1, step=step + 1, steps_per_epoch=steps_per_epoch,
                    loss=float(loss.detach()), elapsed_epoch_seconds=time.monotonic() - epoch_started)
        new.eval()
        # Checkpoint every completed epoch without selecting an epoch using held-out error.
        temp = checkpoint_dir / "weights.tmp"
        torch.save({k: v.detach().cpu() for k, v in new.state_dict().items()}, temp)
        temp.replace(checkpoint_dir / "weights.pt")
        torch.save(dict(epoch_completed=epoch + 1, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                        numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()),
                   OUT / "training_state.pt")
        record = dict(epoch=epoch + 1, train_mse=loss_sum / count, seconds=time.monotonic() - epoch_started)
        epoch_records.append(record)
        save_json("neural_training.json", dict(required_epochs=epochs, completed_epochs=epoch + 1,
                                               batch_size=batch_size, init_seed=1, records=epoch_records))
        log("neural_epoch_complete", **record)

    for name in ("encoder", "projector"):
        assert all(torch.equal(v, getattr(released, name).state_dict()[k])
                   for k, v in getattr(new, name).state_dict().items()), f"Frozen {name} changed"
    reloaded = swm.wm.utils.load_pretrained(str(checkpoint_dir)).to(dev).eval()
    h, a, _ = window(te[:64])
    with torch.no_grad():
        reload_delta = float((neural(new, h, a) - neural(reloaded, h, a)).abs().max())
    assert reload_delta <= 1e-6
    context_errors["retrained_all_positions"] = errors(lambda h, a: neural(reloaded, h, a)[:, -1])
    save_json("context_errors.json", context_errors)
    save_json("fit_complete.json", dict(complete=True, required_epochs=epochs, completed_epochs=epochs,
                                        neural_seconds=time.monotonic() - train_started,
                                        total_seconds=time.monotonic() - t0, reload_max_abs=reload_delta,
                                        context_errors=context_errors))
    log("fit_complete", context_errors=context_errors, total_seconds=time.monotonic() - t0)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if OUT.exists():
            save_json("failure.json", dict(type=type(exc).__name__, message=str(exc)))
        raise
