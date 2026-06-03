# Per-Shot Memory Leak in resonet-simulate — Design Spec

**Date:** 2026-06-03  
**Branch:** `fix/per-shot-memory-leak`  
**Status:** Partially fixed; steady-state leak remains

---

## Problem

The 20-rank MPI simulation job (`resonet-simulate`, 100k shots) OOMs well before
completion. Job 54265402 (200 GB / 10 GB per rank) died at shot ~160/5000.
After initial fixes (see below), job 54405873 (600 GB / 43 GB per rank, 14 ranks)
is projected to OOM at shot ~1481/7143 — still far short.

---

## Evidence Collected

### Job 54265402 (before any fix)
- 20 ranks, 10 GB/rank
- OOM at shot ~160/5000
- Growth rate: ~62 MB/shot

### Job 54405873 (after Phase 1 fixes)
- 14 ranks, 43 GB/rank, 6 GPUs (H100)
- Shot timing: consistent ~8–9 sec/shot throughout (no slowdown → not swapping)

| Shot | MaxRSS | Growth Rate |
|------|--------|-------------|
| ~134 | 13.7 GB | — |
| ~256 | 18.4 GB | 38.5 MB/shot |
| ~333 | 20.6 GB | 28.6 MB/shot |
| ~363 | 22.3 GB | 56.7 MB/shot |
| ~762 | 29.7 GB | 18.5 MB/shot |

Rate decreases from ~40–56 MB/shot (early) to ~18.5 MB/shot (after shot ~762).

### Key diagnostic results

- `tracemalloc` on loading 5 PDB miller arrays: only **~9 KB Python-visible allocation** per array
- Actual miller_array data: **~14.5 MB in C++ flex storage** (906k reflections × 16 bytes)
- `gc.collect()` after deleting miller arrays: **0 cycles collected** → no Python cyclic garbage
- Number of unique PDB structures: **117**
- By shot ~762: all 117 PDBs statistically seen (`117 × (1 − e^{−762/117}) ≈ 116.8`)

---

## Root Cause Analysis

### Phase 1 leak (fixed): simtbx C++ objects not freed between shots

`simulate()` and `sim_background()` created `SimData`, `nanoBragg`, and CUDA wrapper
objects every shot but never explicitly freed them. Python's GC does not aggressively
collect C++ extension objects. Fix: added `del S, C, nb_beam` and `del gpu_simulation, SIM`.

**Reduction: ~62 → ~40 MB/shot**

### Phase 2 leak (root cause identified): per-shot MTZ file reload

`make_crystal.load_crystal()` calls `any_reflection_file(fmodel_file).as_miller_arrays()`
**every shot**, loading the full structure factor array (~14.5 MB C++ flex) from disk each
time. With 117 PDB structures randomly selected per shot, each new PDB incurs a
~14.5 MB first-load cost. After all 117 PDBs are seen (~shot 750), the rate drops from
~40 MB/shot to a steady-state of ~18.5 MB/shot.

The residual ~18.5 MB/shot after all PDBs seen is the steady-state contribution from
C++ flex allocations in the shot pipeline that are freed but leave RSS high (glibc
heap fragmentation / not returned to OS) plus any unidentified residual leak.

### Budget projection with current code

- At shot 762: 29.7 GB used, 13.3 GB remaining
- At 18.5 MB/shot: OOM at shot ~1481 (out of 7143 needed)

---

## Proposed Fix (Phase 2)

### 1. Cache miller arrays in `make_crystal.py`

Add a module-level dict that caches the loaded `miller.array` per MTZ file path.
Each MPI rank maintains its own cache (safe: no shared state between processes).

```python
_miller_array_cache = {}  # {fmodel_file: miller.array}

# In load_crystal():
if fmodel_file not in _miller_array_cache:
    ma = any_reflection_file(fmodel_file).as_miller_arrays()[0]
    if ma.is_complex_array():
        ma = ma.as_amplitude_array()
    _miller_array_cache[fmodel_file] = ma
C.miller_array = _miller_array_cache[fmodel_file]
```

Cache size: 117 PDBs × ~14.5 MB = ~1.7 GB per rank. Bounded and acceptable.

**Expected reduction: ~14.5 MB/shot → steady-state drops from ~18.5 to ~4 MB/shot**

### 2. `gc.collect()` every shot (not every 10)

Change the gc call frequency in `main.py` from `i_shot % 10 == 0` to every shot.
Adds ~1 ms overhead per shot (negligible vs 8–9 sec/shot).

### 3. Per-shot RSS logging in `main.py`

Add `resource.getrusage()` print every 50 shots to track actual growth in future runs:

```python
import resource
if i_shot % 50 == 0:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"RANK {jid+1}: Shot {i_shot+1} RSS={rss:.0f} MB", flush=True)
```

---

## Budget Projection After Fix

| Component | Memory |
|-----------|--------|
| Baseline (simtbx env) | ~5 GB |
| PDB miller_array cache (117 × 14.5 MB) | ~1.7 GB |
| Residual per-shot leak (4 MB × 7143 shots) | ~28.6 GB |
| **Total projected** | **~35.3 GB** |
| Budget (600 GB / 14 ranks) | **43 GB/rank** |
| Margin | **~7.7 GB** |

---

## Files to Modify

| File | Change |
|------|--------|
| `resonet/resonet/sims/make_crystal.py` | Add `_miller_array_cache` dict + cache lookup in `load_crystal()` |
| `resonet/resonet/sims/main.py` | `gc.collect()` every shot; add RSS logging every 50 shots |

## Already Applied (Phase 1)

| File | Change |
|------|--------|
| `resonet/resonet/sims/simulator.py` | `del S, C, nb_beam` after `S.D.free_all()` |
| `resonet/resonet/sims/simulator.py` | `del gpu_simulation, SIM` in `sim_background()` |
| `resonet/resonet/sims/main.py` | `gc.collect()` every 10 shots (to be changed to every shot) |
| `resonet/resonet/sims/main.py` / `runme.py` | `--rankOffset`/`--totalRanks` for resuming partial runs |

---

## Verification

1. Submit job with Phase 2 fixes applied on `fix/per-shot-memory-leak` branch
2. Monitor per-shot RSS via log output (`resource.getrusage` lines)
3. Confirm growth rate ≤ 5 MB/shot after shot ~200 (all PDBs cached)
4. Confirm job completes all 7143 shots/rank without OOM
5. Merge branch to `main` after successful completion
