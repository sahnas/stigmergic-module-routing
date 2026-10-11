# Version 1: incomplete implementation run

Kaggle status: `ERROR`. Failure at 2026-10-11 18:40:21 UTC, after the first
original CEM solve for the released TwoRoom predictor, before any control
was returned to the environment. The supplemental cost diagnostic called
the pinned LeWM cost path with eight starting rows and four plans per row;
its cached goal embedding did not broadcast at that batch size. The
official CEM configuration uses batch size one. The corrected diagnostic
uses that same row-wise batching and preserves four extra scored plans per
row, the original search and all preregistered decisions.

`submitted_selection.py` is the exact version-1 source pulled from Kaggle.
Its SHA-256 agrees with `run_manifest.json` and the original submission
record. `retrieval_manifest.json` hashes the retrieved files. `failure.json`
contains the parent subprocess error; `lewm-cem-selection.log` contains the
underlying traceback. `runtime_smoke.json` only covers the original solver
smoke check, which passed; it did not exercise the failing LeWM diagnostic.

No episode evaluation completed. There are no successes, gate decision,
conditional comparison or scientific verdict to report from this version.
