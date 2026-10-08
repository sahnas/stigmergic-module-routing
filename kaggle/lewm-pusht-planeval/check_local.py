"""Check deployment parity, causal outputs, and paired-test bookkeeping."""
import importlib.util
from pathlib import Path
import sys
import torch
from pusht_planeval import make_bilinear_predictor, paired_stats


def main():
    fit_path = Path(__file__).parents[1] / "lewm-pusht-fit/pusht_fit.py"
    spec = importlib.util.spec_from_file_location("exp19_fit_reference", fit_path)
    fit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fit)
    torch.manual_seed(47)
    emb, act = torch.randn(8, 3, 192), torch.randn(8, 3, 10)
    W = torch.randn(2507, 192)
    op = make_bilinear_predictor(W)
    out = op(emb, act)
    assert out.shape == (8, 3, 192) and op.num_frames == 3
    for t in range(3):
        expected = fit.features(emb[:, :t + 1], act[:, :t + 1]) @ W
        torch.testing.assert_close(out[:, t], expected, rtol=0, atol=0)
    # Full-sequence prefix outputs agree with a genuinely short-context call.
    torch.testing.assert_close(out[:, :1], op(emb[:, :1], act[:, :1]), rtol=0, atol=0)
    changed = emb.clone()
    changed[:, 2] += 999
    torch.testing.assert_close(out[:, :2], op(changed, act)[:, :2], rtol=0, atol=0)

    def rows(successes):
        return [dict(episode_id=i, start_step=0, goal_step=25, success=s) for i, s in enumerate(successes)]
    same = rows([True, False, True, False])
    equal = paired_stats(same, same)
    assert equal['mcnemar_exact_two_sided_p'] == 1.0 and not equal['strictly_more_successes']
    extreme = paired_stats(rows([True] * 10), rows([False] * 10))
    assert extreme['bilinear_only'] == 10 and extreme['reference_only'] == 0
    assert extreme['mcnemar_exact_two_sided_p'] == 2 / 1024
    mixed = paired_stats(rows([True, True, False, False]), rows([True, False, True, False]))
    assert [mixed[k] for k in ['both_success', 'bilinear_only', 'reference_only', 'both_failure']] == [1, 1, 1, 1]
    assert mixed['mcnemar_exact_two_sided_p'] == 1.0
    try:
        mismatched = rows([True, False, True, False])
        mismatched[0]['start_step'] = 1
        paired_stats(same, mismatched)
        raise RuntimeError('Unpaired tasks accepted')
    except AssertionError:
        pass
    print('PASS: deployed operator equals fitted features at all contexts; causal prefix parity; paired counts, exact p-values and mismatch rejection')


if __name__ == '__main__':
    main()
