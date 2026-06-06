# Implementation Plan: Per-Shot Memory Leak Fix (Phase 2)

**Date:** 2026-06-03  
**Branch:** `fix/per-shot-memory-leak`  
**Spec:** `docs/specs/2026-06-03-per-shot-memory-leak-design.md`  
**Trigger:** Apply after job 54405873 OOMs (projected ~shot 1481)

---

## Prerequisites

- Confirm job 54405873 has OOM'd (check log for `oom_kill` / `Out Of Memory`)
- Checkout `fix/per-shot-memory-leak` branch

```bash
git checkout fix/per-shot-memory-leak
```

---

## Step 1 — Cache miller arrays in `make_crystal.py`

**File:** `resonet/resonet/sims/make_crystal.py`

Add module-level cache dict after the imports block (after `from resonet.sims import process_pdb`):

```python
# Per-rank cache: avoids reloading the same MTZ files on every shot.
# 117 PDBs x ~14.5 MB C++ flex per miller_array = ~1.7 GB total, bounded.
_miller_array_cache = {}
```

In `load_crystal()`, replace the MTZ loading block:

```python
# BEFORE:
ma = any_reflection_file(fmodel_file).as_miller_arrays()[0]
if ma.is_complex_array():
    ma = ma.as_amplitude_array()
C.miller_array = ma
C.symbol = ma.space_group_info().type().lookup_symbol()

# AFTER:
if fmodel_file not in _miller_array_cache:
    ma = any_reflection_file(fmodel_file).as_miller_arrays()[0]
    if ma.is_complex_array():
        ma = ma.as_amplitude_array()
    _miller_array_cache[fmodel_file] = ma
C.miller_array = _miller_array_cache[fmodel_file]
C.symbol = _miller_array_cache[fmodel_file].space_group_info().type().lookup_symbol()
```

---

## Step 2 — Update `main.py`: gc frequency + RSS logging

**File:** `resonet/resonet/sims/main.py`

2a. Add `import resource` alongside the existing `import gc` line.

2b. Change `gc.collect()` from every 10 shots to every shot:

```python
# BEFORE:
if i_shot % 10 == 0:
    gc.collect()

# AFTER:
gc.collect()
```

2c. Add RSS logging every 50 shots (insert after the timing print):

```python
if i_shot % 50 == 0:
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"RANK {jid+1}/{njobs}: Shot {i_shot+1} RSS={rss_mb:.0f} MB", flush=True)
```

---

## Step 3 — Commit

```bash
git add resonet/resonet/sims/make_crystal.py resonet/resonet/sims/main.py
git commit -m "Fix steady-state memory growth: cache miller arrays + per-shot gc + RSS logging

Cache the loaded miller.array per MTZ file path (117 PDBs x ~14.5 MB each).
Without this, any_reflection_file() reloads 14.5 MB of C++ flex data every shot,
driving ~18.5 MB/shot steady-state growth. With cache, rate drops to ~4 MB/shot.

Also: run gc.collect() every shot (not every 10) and add per-shot RSS logging
every 50 shots to track memory in future runs.

Expected: job completes all 7143 shots/rank within 43 GB/rank budget (~35 GB used)."
```

---

## Step 4 — Push and resubmit

```bash
git push origin fix/per-shot-memory-leak
sbatch run_hitfinder.sh
```

---

## Step 5 — Verify

Monitor the new job log for:
- RSS lines every 50 shots — confirm growth ≤ 5 MB/shot after shot ~200
- No `oom_kill` events
- All 14 ranks reach shot 7143/7143

If RSS growth stays flat after shot ~200 → fix confirmed.  
If still growing → investigate residual with finer RSS logging (every 10 shots).

---

## Step 6 — Merge to main (after successful run)

Once a complete 100k-shot run finishes without OOM:

```bash
git checkout main
git merge fix/per-shot-memory-leak
git push origin main
```

Then run:
```bash
resonet-mergefiles "hitfinder_100k/compressed*.h5" hitfinder_100k_merged.h5
```
