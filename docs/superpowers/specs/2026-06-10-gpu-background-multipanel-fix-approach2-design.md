# GPU Background Multipanel Fix — Approach 2: Single multi-panel GPU call

**Date:** 2026-06-10  
**Branch:** `gpu-background-singlechot` (new, from `eiger4m-unassembled-cxi` after Approach 1 merges)  
**Status:** Planned — not yet implemented

---

## Context

Approach 1 (`2026-06-10-gpu-background-multipanel-fix-approach1-design.md`) fixes the `cu_n_panels==1` crash by using a single-panel GPU detector per panel. It loops 64× with 64 separate GPU allocations per shot. This is fast enough for production but leaves performance on the table.

Approach 2 targets a single GPU allocation and single `add_background` call for all 64 panels per shot — the theoretically optimal path.

---

## Goal

Replace the 64-iteration GPU loop in `sim_background_multipanel` with a single GPU call that computes background for all panels simultaneously, using the multi-panel `gpu_det` and `get_raw_pixels()` to retrieve all panel pixels at once.

---

## Proposed Flow

```python
def sim_background_multipanel(self, det, beam, dev, stol_name, redo_air_water=False):
    spectrum = [(beam.get_wavelength(), 1)]
    xray_beams = get_xray_beams(spectrum, beam)

    # Use panel 0 geometry to initialize beam/flux settings
    SIM0 = nanoBragg(det, beam, panel_id=0)
    SIM0.beamsize_mm = paths_and_const.BEAM_SIZE_MM
    SIM0.xray_beams = xray_beams
    SIM0.flux = paths_and_const.FLUX
    SIM0.Fbg_vs_stol = make_sims.load_stol(stol_name)
    SIM0.amorphous_sample_thick_mm = paths_and_const.XTALSIZE_MM
    SIM0.amorphous_density_gcm3 = 1
    SIM0.amorphous_molecular_weight_Da = 12

    gpu_sim = self.exascale_api(nanoBragg=SIM0)
    gpu_sim.allocate()
    gpu_det = self.gpud(deviceId=dev, detector=det, beam=beam)  # all 64 panels
    gpu_det.each_image_allocate()
    gpu_det.scale_in_place(0)
    gpu_sim.add_background(gpu_det)   # single call for all panels?

    if redo_air_water:
        for mw, thick, density, stol_f in [
            (14, 5, 1.2e-3, paths_and_const.AIR_STOL),
            (18, paths_and_const.XTALSIZE_MM, 1, paths_and_const.WATER_STOL),
        ]:
            SIM0.Fbg_vs_stol = make_sims.load_stol(stol_f)
            SIM0.amorphous_sample_thick_mm = thick
            SIM0.amorphous_density_gcm3 = density
            SIM0.amorphous_molecular_weight_Da = mw
            gpu_sim.add_background(gpu_det)

    all_pixels = gpu_det.get_raw_pixels().as_numpy_array()  # shape: (64, slow, fast)
    gpu_det.each_image_free()
    del gpu_det, gpu_sim, SIM0

    return np.concatenate([all_pixels[pid].ravel() for pid in range(len(det))])
```

---

## Critical Research Questions

Before implementation, these must be answered by reading the exascale API source in `easyBragg/simtbx_project/simtbx/gpu/`:

1. **Does `gpu_sim.add_background(gpu_det)` compute correct per-panel geometry when `gpu_det` has 64 panels but `gpu_sim` was initialized from a single-panel `SIM0`?**
   - If the exascale API uses `SIM0`'s geometry for all panels → background will be wrong (same geometry replicated 64×)
   - If the exascale API uses `gpu_det`'s per-panel geometry → correct

2. **Does `redo_air_water` require per-panel SIM re-init?**
   - The air/water background uses `amorphous_*` params on the SIM object. Are these propagated per-panel or global?

3. **What is the actual speedup vs Approach 1?**
   - Measure: `time/shot` with 64-panel Eiger4M, 1 GPU

---

## Risk

If the exascale API computes background only for the panel whose geometry was used to initialize `gpu_sim` (i.e., panel 0), all 64 panels will get identical background — incorrect physics. In that case, Approach 1 (per-panel loop) remains the correct implementation.

**Fallback:** If Approach 2 produces incorrect output (verified by comparing panel-to-panel background variation), fall back to Approach 1 and mark this branch as abandoned.

---

## Performance Target

| Approach | GPU allocations/shot | Expected time/shot |
|----------|---------------------|-------------------|
| CPU fallback | 0 | ~282s |
| Approach 1 (per-panel GPU) | 64 | ~5–15s |
| Approach 2 (single GPU call) | 1 | ~1–3s |

---

## Branch Workflow

1. Branch `gpu-background-singlechot` from `eiger4m-unassembled-cxi` (after Approach 1 merges to main)
2. Investigate exascale API source to answer research questions above
3. Implement and benchmark
4. If correct: PR → main; if wrong physics: abandon and document why
