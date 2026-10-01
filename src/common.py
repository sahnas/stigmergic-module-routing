"""Shared helpers for the analysis scripts."""
import json
import numpy as np


def load(path):
    return [json.loads(l) for l in open(path)]


def col(rows, variant, key):
    """Values of `key` for `variant`, sorted by seed (None -> nan)."""
    rs = sorted([r for r in rows if r['variant'] == variant], key=lambda r: r['seed'])
    return np.array([np.nan if r.get(key) is None else r[key] for r in rs], dtype=float)


def boot(d, n=10000):
    b = np.random.default_rng(0).choice(d, (n, len(d))).mean(1)
    return np.percentile(b, [2.5, 97.5])
