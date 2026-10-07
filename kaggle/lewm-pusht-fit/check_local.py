"""CPU checks for episode leakage, experiment-18 parity and short contexts."""
import numpy as np
import torch
from pusht_fit import episode_windows, features


def main():
    lengths = np.array([15, 16, 18, 20, 17, 21, 30, 16, 19, 25])
    offsets = np.r_[0, np.cumsum(lengths[:-1])]
    tr, te, heldout = episode_windows(lengths, offsets, 3, 5, np.random.default_rng(0))
    owners = np.repeat(np.arange(len(lengths)), lengths)
    assert len(tr) + len(te) == sum(max(0, int(n) - 15) for n in lengths)
    assert set(owners[tr]).isdisjoint(set(owners[te]))
    assert set(owners[te]) == set(heldout)
    for starts in (tr, te):
        assert np.array_equal(owners[starts], owners[starts + 15])
        assert np.array_equal(owners[starts], owners[starts + 14])
    assert 0 not in set(owners[np.r_[tr, te]])  # exactly 15 frames cannot contain a 15-step target
    assert sum(owners[np.r_[tr, te]] == 1) == 1  # 16 frames yield exactly one start
    try:
        episode_windows(lengths, offsets + 1, 3, 5, np.random.default_rng(0))
        raise RuntimeError("Invalid offsets accepted")
    except AssertionError:
        pass

    torch.manual_seed(4)
    emb = torch.randn(4, 3, 192)
    act = torch.randn(4, 3, 10)
    # Reference expression from experiment 18, independent of features().
    z0, z1, z2, a = emb[:, 2], emb[:, 1], emb[:, 0], act[:, 2]
    expected = torch.cat([z0, z1, z2, a, (z0[:, :, None] * a[:, None, :]).reshape(4, -1), torch.ones(4, 1)], 1)
    torch.testing.assert_close(features(emb, act), expected, rtol=0, atol=0)
    assert expected.shape == (4, 2507)
    for context in (1, 2):
        x = features(emb[:, :context], act[:, :context])
        torch.testing.assert_close(x[:, :192], emb[:, context - 1])
        torch.testing.assert_close(x[:, 192:384], emb[:, 0])
        torch.testing.assert_close(x[:, 384:576], emb[:, 0])
    # Prefix predictions must not depend on later observations/actions.
    prefix = features(emb[:, :2], act[:, :2]).clone()
    emb[:, 2] += 1000
    act[:, 2] -= 1000
    torch.testing.assert_close(prefix, features(emb[:, :2], act[:, :2]), rtol=0, atol=0)
    print("PASS: disjoint episode split, boundary windows, experiment-18 feature parity, causal short-context padding")


if __name__ == "__main__":
    main()
