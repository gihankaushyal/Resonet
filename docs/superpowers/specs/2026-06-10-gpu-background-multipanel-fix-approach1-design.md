# GPU Background Multipanel Fix — Approach 1: Single-panel GPU detector per panel

**Date:** 2026-06-10  
**Branch:** `eiger4m-unassembled-cxi`  
**Status:** Implemented

---

## Problem

`sim_background_multipanel` in `simulator.py` computes per-panel background for the 64-panel Eiger4M detector by looping over all panels. The original GPU path crashed at runtime:

```
RuntimeError: scitbx Internal Error: detector.cu(195): SCITBX_ASSERT(cu_n_panels == 1) failure.
```

**Root cause:** `gpu_det = self.gpud(deviceId=dev, detector=det, beam=beam)` with a 64-panel `det` creates a GPU detector with `cu_n_panels=64`. Then `gpu_det.write_raw_pixels(SIM)` — where `SIM` is a single-panel `nanoBragg` object — asserts `cu_n_panels==1` at `detector.cu:195` before copying GPU memory to CPU.

A CPU fallback (`SIM.add_background()`) was applied as a temporary fix but is ~64× slower (~282s/shot), making a 100-shot job impossible within a 30-minute wall time.

---

## Solution

`gpu_detector` exposes two constructors:
- `gpud(deviceId, detector, beam)` — allocates for all panels in the dxtbx detector
- `gpud(deviceId, nanoBragg)` — allocates for the single panel described by a `nanoBragg` object

Using `gpud(deviceId=dev, nanoBragg=SIM)` — where `SIM = nanoBragg(det, beam, panel_id=pid)` is already the per-panel single-panel object — produces `cu_n_panels=1`, satisfying the CUDA constraint.

`get_raw_pixels()` (used to retrieve results) has no panel-count restriction (`detector.cu:208-220`) and returns a flex array of shape `(n_panels, slow, fast)`. For a single-panel `gpu_det`, this is `(1, slow, fast)` and `.ravel()` gives the flat panel pixels.

---

## Change

**File:** `resonet/resonet/sims/simulator.py`, function `sim_background_multipanel` (~line 505)

```python
# BEFORE (broken — cu_n_panels=64):
gpu_det = self.gpud(deviceId=dev, detector=det, beam=beam)
...
gpu_det.write_raw_pixels(SIM)
panel_bgs.append(SIM.raw_pixels.as_numpy_array().ravel())

# AFTER (fixed — cu_n_panels=1):
gpu_det = self.gpud(deviceId=dev, nanoBragg=SIM)
...
panel_bgs.append(gpu_det.get_raw_pixels().as_numpy_array().ravel())
```

All other logic (`gpu_sim`, `each_image_allocate`, `scale_in_place`, `add_background`, `redo_air_water` loop, `each_image_free`, cleanup) is unchanged.

---

## Data Flow

```
for pid in 0..63:
    SIM = nanoBragg(det, beam, panel_id=pid)   # single-panel, correct geometry
    gpu_sim = exascale_api(nanoBragg=SIM)       # GPU sim scoped to panel pid
    gpu_det = gpud(nanoBragg=SIM)              # ← single-panel GPU detector (FIXED)
    gpu_sim.add_background(gpu_det)            # computes background on GPU for panel pid
    pixels = gpu_det.get_raw_pixels().ravel()  # ← no cu_n_panels assertion (FIXED)
    panel_bgs.append(pixels)

return np.concatenate(panel_bgs)               # shape: (64 * panel_pixels,)
```

---

## Performance

| Mode | Time/shot |
|------|-----------|
| CPU fallback (`add_background()`) | ~282s |
| GPU per-panel (this fix) | ~5–15s (estimated) |
| GPU single-call (Approach 2, future) | ~1–3s (target) |

64 GPU allocations per shot — not optimal but correct and fast enough for production use.

---

## Files Modified

- `resonet/resonet/sims/simulator.py` — source
- `simforge/envs/simtbx_mpi/lib/python3.9/site-packages/resonet/sims/simulator.py` — manual copy (non-editable install)

---

## Verification

1. Submit job with `sbatch run_cxi_100.sh`
2. Shot 1 completes in ~5–15s (not ~282s)
3. No `cu_n_panels==1` error in log
4. CXI output frames have shape `(5632, 384)` per panel

---

## Limitations

This approach still loops 64× per shot with separate GPU allocations. Approach 2 (see `2026-06-10-gpu-background-multipanel-fix-approach2-design.md`) targets a single GPU call for all panels.
