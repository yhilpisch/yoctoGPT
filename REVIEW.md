# yoctoGPT — Open Issues

**Updated**: 2025-06-08

Issues resolved in the quality sweep have been removed. Only open items remain.

---

## LOW — Minor Improvements

### L1. Metrics CSV rows written one-by-one with `open("a")` inside the training loop

Each iteration opens the file, appends a row, and closes it. This is safe but slow. A buffered writer or periodic flush would be more efficient for long runs.

---

## TEST COVERAGE GAPS

| Area | Status |
|---|---|
| **DDP multi-process** | **Not covered** (only single-process paths tested) |
