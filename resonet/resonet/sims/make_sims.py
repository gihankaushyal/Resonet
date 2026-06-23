# coding: utf-8
import glob
import numpy as np
from scipy.interpolate import interp1d
import time

import dxtbx
from scitbx.array_family import flex
from simtbx.diffBragg import utils
from simtbx.nanoBragg import utils as nb_utils
from simtbx.nanoBragg import sim_data


from resonet.sims import paths_and_const
from resonet.sims import make_crystal


def choose_res(hres=None):
    """ choose a random resolution"""
    if paths_and_const.FIX_RES is not None:
        return paths_and_const.FIX_RES
    if hres is None:
        res = 0.5/(paths_and_const.STOL_MIN + np.random.random()*paths_and_const.STOL_RNG)
    else:
        res = np.random.uniform(hres, 6)  # maximum of 6 Angstrom  
    return res


def choose_pdb():
    # choose a random PDB file
    return np.random.choice(paths_and_const.RANDOM_PDBS)


def choose_deltaB(hres=None):
    # choose a delta -Bfactor to scale the scattering
    res = choose_res(hres)
    B = 4*res**2 + 12
    return B-10


def choose_mos(mos_min=None, mos_max=None):
    # choose a random mosaicity
    if mos_min is None:
        mos_min = paths_and_const.MOS_MIN
    if mos_max is None:
        mos_max = paths_and_const.MOS_MAX
    assert mos_min < mos_max

    mos_rng = (np.sqrt(mos_max) - np.sqrt(mos_min))**2    
    r = np.random.random()
    mosaic = mos_min + r*mos_rng
    mosdoms = int(1000 + 50*mosaic**2)
    return mosaic, mosdoms


def choose_stol():
    # choose a random plastic scattering profile
    stol, Fbg = np.loadtxt(np.random.choice(paths_and_const.RANDOM_STOLS)).T
    flex.vec2_double(list(zip(Fbg, stol)))
    return flex.vec2_double( Fbg, stol)


def random_bg(D,B, stol_name, roi=None):
    """

    :param D: dxtbx detector
    :param B: dxtbx beam
    :param stol_name: flex.vec2d scattering profile (sin theta vs lambda)
    :param  roi: nanoBragg region_of_interest
    :return:
    """
    # simulate scattering from a plastic scattering profile
    funky_bg = nb_utils.sim_background(D, B, [B.get_wavelength()], [1], paths_and_const.FLUX,
                molecular_weight=12, sample_thick_mm=paths_and_const.XTALSIZE_MM,
                Fbg_vs_stol=load_stol(stol_name), roi=roi)
    return funky_bg.as_numpy_array()


def load_stol(name):
    """

    :param name:  name of a stol text file (first col is sqrt(intensity), second is the sin-theta-over-lambda value)
    :return:
    """
    Fbg, stol = np.loadtxt(name).T
    return flex.vec2_double(list(zip(Fbg, stol)))


def get_background(D,B, no_air=False, no_water=False, water_path_mm=None, air_path_mm=None, roi=None):

    air = 0
    if not no_air:
        if air_path_mm is None:
            air_path_mm = 5
        air = nb_utils.sim_background(D, B, [B.get_wavelength()], [1], paths_and_const.FLUX, molecular_weight=14,
                                      sample_thick_mm=air_path_mm,
                                   Fbg_vs_stol=load_stol(paths_and_const.AIR_STOL), density_gcm3=1.2e-3, roi=roi)  #

    water = 0
    if not no_water:
        if water_path_mm is None:
            water_path_mm = paths_and_const.XTALSIZE_MM
        water = nb_utils.sim_background(D, B, [B.get_wavelength()], [1], paths_and_const.FLUX, molecular_weight=18,
                                   sample_thick_mm=water_path_mm,
                                   Fbg_vs_stol=load_stol(paths_and_const.WATER_STOL), density_gcm3=1, roi=roi)  #

    background = air + water
    return background.as_numpy_array()


def get_Bfac_img(STOL, hres=None):
    """

    :param STOL: sin-theta-over-lambda of every pixel on detector
    :return: delta-Bfactor at every pixel (for aadjusting the spot resolution)
    """
    B, stol, factor = get_deltaB_factor(hres)
    I = interp1d(stol, factor, bounds_error=False, fill_value=0)
    Bfac_img = I(STOL.ravel()).reshape(STOL.shape)
    reso = np.sqrt(.25*(B + 10 - 12))
    return reso, Bfac_img


def get_deltaB_factor(hres=None):
    """

    :return: 3-tuple,
        -first element is the delta-B factor (can be converted to resolution)
        -second element is the sin-theta-over-lambda values
        -third element is the B-factor scale at each sin-theta-over-lambda value
    """
    stol = np.arange(0, 0.5, 0.01)
    B = choose_deltaB(hres)
    exponent = 2 * B * stol ** 2
    is_bad = exponent > 100
    fac = np.exp(-exponent)
    fac[is_bad] = 0
    return B, stol, fac


def get_theta_map(detector, beam, panel_id=None):
    """
    :param detector: dxtbx detector
    :param beam: dxtbx beam
    :param panel_id: None or 0 → return STOL for panel 0 (backward compat).
                     'all' → return list of STOL arrays, one per panel.
    :return: sin-theta-over-lambda for each pixel
    """
    Qmags = {}
    unit_s0 = beam.get_unit_s0()
    for pid in range(len(detector)):
        xdim, ydim = detector[pid].get_image_size()
        panel_sh = ydim, xdim

        FAST = np.array(detector[pid].get_fast_axis())
        SLOW = np.array(detector[pid].get_slow_axis())
        ORIG = np.array(detector[pid].get_origin())

        Ypos, Xpos = np.indices(panel_sh)
        px = detector[pid].get_pixel_size()[0]
        Ypos = Ypos * px
        Xpos = Xpos * px

        SX = ORIG[0] + FAST[0]*Xpos + SLOW[0]*Ypos
        SY = ORIG[1] + FAST[1]*Xpos + SLOW[1]*Ypos
        SZ = ORIG[2] + FAST[2]*Xpos + SLOW[2]*Ypos

        Snorm = np.sqrt(SX**2 + SY**2 + SZ**2)
        SX /= Snorm
        SY /= Snorm
        SZ /= Snorm

        QX = (SX - unit_s0[0]) / beam.get_wavelength()
        QY = (SY - unit_s0[1]) / beam.get_wavelength()
        QZ = (SZ - unit_s0[2]) / beam.get_wavelength()
        Qmags[pid] = np.sqrt(QX**2 + QY**2 + QZ**2)

        # Early exit for single-panel case to avoid unnecessary computation
        if panel_id is None or panel_id == 0:
            break

    if panel_id == 'all':
        return [Qmags[pid] / 2 for pid in range(len(detector))]
    return Qmags[0] / 2


def get_Bfac_img_flat(stol_list, hres=None):
    """Like get_Bfac_img but for a list of per-panel STOL arrays.

    Draws a single B-factor and applies it consistently across all panels
    so that the resolution truncation is uniform for the whole shot.

    :param stol_list: list of per-panel STOL arrays
    :param hres: optional high-resolution limit
    :return: (reso, Bfac_flat) where Bfac_flat is a 1D array for all panels
    """
    B, stol, factor = get_deltaB_factor(hres)
    I = interp1d(stol, factor, bounds_error=False, fill_value=0)
    Bfac_flat = np.concatenate([I(s.ravel()) for s in stol_list])
    reso = np.sqrt(.25 * (B + 10 - 12))
    return reso, Bfac_flat


def set_noise(noise_sim, calib_noise_percent=paths_and_const.CALIB_NOISE_PCT):
    """

    :param noise_sim: nanoBragg simulator instance
    :param calib_noise_percent: calibration noise (how much each pixels gain varies)
    :return: nanoBragg simulator instance
    """
    noise_sim.detector_calibration_noise_pct = calib_noise_percent
    noise_sim.exposure_s = 1
    noise_sim.calib_seed=0
    noise_sim.seed=0
    # flux is NOT set here: quantum_gain=1 means raw_pixels are already in photon units,
    # so Poisson noise is drawn from pixel values directly. Setting flux here would
    # double-count it and break the per-shot flux scaling in simulator.py.
    noise_sim.adc_offset_adu =0
    noise_sim.detector_psf_kernel_radius_pixels = 5
    noise_sim.detector_psf_fwhm_mm =0
    noise_sim.quantum_gain = 1
    noise_sim.readout_noise_adu = 0
    return noise_sim


def apply_epix_noise(img, t1=80, t2=270,
                     sigma_hg=0.02, sigma_mg=0.023, sigma_lg=0.27,
                     sat_lg=11000,
                     rng=None):
    """Per-pixel auto-ranging noise model for ePix10k detector.

    Physical order: Poisson shot noise → gain-zone classification →
    Gaussian readout noise → LG saturation clip.

    Gain zones: count in [0, t1] → HG; (t1, t2] → MG; > t2 → LG.
    sat_lg is the LG well capacity; pixels above it are clipped.

    :param img: noiseless photon-count image (numpy float32 array, any shape)
    :param t1: HG→MG switch threshold in photon counts (default 80)
    :param t2: MG→LG switch threshold in photon counts (default 270)
    :param sigma_hg: readout noise RMS in photon-equivalent for HG zone (default 0.02)
    :param sigma_mg: readout noise RMS in photon-equivalent for MG zone (default 0.023)
    :param sigma_lg: readout noise RMS in photon-equivalent for LG zone (default 0.27)
    :param sat_lg: LG saturation limit in photon counts (default 11000)
    :param rng: numpy.random.Generator instance (created internally if None)
    :return: noised image as float32 array, same shape as img, clipped to [0, sat_lg]
    """
    if t1 >= t2:
        raise ValueError(f"apply_epix_noise: t1 ({t1}) must be < t2 ({t2}).")
    if sat_lg <= t2:
        raise ValueError(f"apply_epix_noise: sat_lg ({sat_lg}) must be > t2 ({t2}).")
    if any(s < 0 for s in (sigma_hg, sigma_mg, sigma_lg)):
        raise ValueError(
            f"apply_epix_noise: all sigma values must be non-negative; "
            f"got hg={sigma_hg}, mg={sigma_mg}, lg={sigma_lg}."
        )
    if rng is None:
        rng = np.random.default_rng()
    out = rng.poisson(np.maximum(img, 0)).astype(np.float32)
    hg = out <= t1
    mg = (out > t1) & (out <= t2)
    lg = out > t2
    for mask, sigma in [(hg, sigma_hg), (mg, sigma_mg), (lg, sigma_lg)]:
        n = int(np.sum(mask))
        if n:
            out[mask] += rng.normal(0, sigma, size=n).astype(np.float32)
    out = np.maximum(out, 0)
    # Clip the full array: lg mask is stale after readout noise, so MG pixels pushed
    # above sat_lg by noise must also be caught here.
    np.minimum(out, sat_lg, out=out)
    return out


def apply_jungfrau_noise(img, t1=34, t2=342,
                         sigma_g0=0.2, sigma_g1=1.5, sigma_g2=15.0,
                         sat_g2=3400,
                         rng=None):
    """Per-pixel auto-ranging noise model for Jungfrau detector.

    Physical order: Poisson shot noise → gain-zone classification →
    Gaussian readout noise → floor clip to 0 → G2 saturation clip.

    Gain zones: count in [0, t1] → G0; (t1, t2] → G1; > t2 → G2.
    sat_g2 is the G2 ADC saturation in photon counts; pixels above it are clipped.

    Defaults derived from Jungfrau 4M at ~12 keV: G0 gain 478.6 ADU/photon,
    14-bit ADC (max 16384), 10× gain steps between modes (values rounded).

    :param img: noiseless photon-count image (numpy float32 array, any shape)
    :param t1: G0→G1 switch threshold in photon counts (default 34)
    :param t2: G1→G2 switch threshold in photon counts (default 342)
    :param sigma_g0: readout noise RMS in photon-equivalent for G0 zone (default 0.2)
    :param sigma_g1: readout noise RMS in photon-equivalent for G1 zone (default 1.5)
    :param sigma_g2: readout noise RMS in photon-equivalent for G2 zone (default 15.0)
    :param sat_g2: G2 ADC saturation in photon counts (default 3400)
    :param rng: numpy.random.Generator instance (created internally if None)
    :return: noised image as float32 array, same shape as img, clipped to [0, sat_g2]
    """
    if t1 >= t2:
        raise ValueError(f"apply_jungfrau_noise: t1 ({t1}) must be < t2 ({t2}).")
    if sat_g2 <= t2:
        raise ValueError(f"apply_jungfrau_noise: sat_g2 ({sat_g2}) must be > t2 ({t2}).")
    if any(s < 0 for s in (sigma_g0, sigma_g1, sigma_g2)):
        raise ValueError(
            f"apply_jungfrau_noise: all sigma values must be non-negative; "
            f"got g0={sigma_g0}, g1={sigma_g1}, g2={sigma_g2}."
        )
    if rng is None:
        rng = np.random.default_rng()
    out = rng.poisson(np.maximum(img, 0)).astype(np.float32)
    g0 = out <= t1
    g1 = (out > t1) & (out <= t2)
    g2 = out > t2
    for mask, sigma in [(g0, sigma_g0), (g1, sigma_g1), (g2, sigma_g2)]:
        n = int(np.sum(mask))
        if n:
            out[mask] += rng.normal(0, sigma, size=n).astype(np.float32)
    out = np.maximum(out, 0)
    np.minimum(out, sat_g2, out=out)
    return out


def apply_agipd_noise(img, t1=65, t2=2000,
                      adu_hg=64, adu_mg=8, adu_lg=1,
                      sigma_hg=7.0, sigma_mg=3.0, sigma_lg=1.5,
                      rng=None):
    """Per-pixel auto-ranging noise model for AGIPD 1M detector.

    Physical order: Poisson shot noise → gain-zone classification →
    ADU conversion (multiply by zone gain) → Gaussian readout noise →
    floor clip to 0.

    Unlike apply_epix_noise/apply_jungfrau_noise, output is in ADU (not
    photon-equivalent) because AGIPD has large gain differences between zones.

    Gain zones (threshold-based, per pixel):
      HG (high gain):   photons ≤ t1   → adu_hg ADU/photon, sigma_hg ADU readout
      MG (medium gain): t1 < photons ≤ t2 → adu_mg ADU/photon, sigma_mg ADU readout
      LG (low gain):    photons > t2   → adu_lg ADU/photon,  sigma_lg ADU readout

    Defaults from AGIPD 1M at ~9.4 keV:
      HG→MG threshold: 65 photons (midpoint of 50–80 ph range at 12.4 keV)
      MG→LG threshold: 2000 photons
      HG gain: 64 ADU/photon, σ_read ≈ 7 ADU (from 350 e⁻ r.m.s. noise floor)
      MG gain: 8 ADU/photon,  σ_read ≈ 3 ADU (literature approximation)
      LG gain: 1 ADU/photon,  σ_read ≈ 1.5 ADU (literature approximation)

    :param img: noiseless photon-count array (numpy float32, any shape)
    :param t1: HG→MG switch threshold in photon counts (default 65)
    :param t2: MG→LG switch threshold in photon counts (default 2000)
    :param adu_hg: ADU per photon for HG zone (default 64)
    :param adu_mg: ADU per photon for MG zone (default 8)
    :param adu_lg: ADU per photon for LG zone (default 1)
    :param sigma_hg: readout noise RMS in ADU for HG zone (default 7.0)
    :param sigma_mg: readout noise RMS in ADU for MG zone (default 3.0)
    :param sigma_lg: readout noise RMS in ADU for LG zone (default 1.5)
    :param rng: numpy.random.Generator instance (created internally if None)
    :return: noised image as float32 array in ADU, same shape as img, clipped to [0, ∞)
    """
    if t1 >= t2:
        raise ValueError(f"apply_agipd_noise: t1 ({t1}) must be < t2 ({t2}).")
    if any(s < 0 for s in (sigma_hg, sigma_mg, sigma_lg)):
        raise ValueError(
            f"apply_agipd_noise: all sigma values must be non-negative; "
            f"got hg={sigma_hg}, mg={sigma_mg}, lg={sigma_lg}."
        )
    if rng is None:
        rng = np.random.default_rng()
    out = rng.poisson(np.maximum(img, 0)).astype(np.float32)
    hg = out <= t1
    mg = (out > t1) & (out <= t2)
    lg = out > t2
    # Convert to ADU per zone
    out[hg] *= adu_hg
    out[mg] *= adu_mg
    out[lg] *= adu_lg
    # Add Gaussian readout noise in ADU
    for mask, sigma in [(hg, sigma_hg), (mg, sigma_mg), (lg, sigma_lg)]:
        n = int(np.sum(mask))
        if n:
            out[mask] += rng.normal(0, sigma, size=n).astype(np.float32)
    out = np.maximum(out, 0)
    return out


def main():
    fnames = glob.glob("/mnt/data/s2/blstaff/SOLTIS/AI_PREDICTION/3.15A/*cbf")
    loader = dxtbx.load(fnames[0])
    D = loader.get_detector()
    D0 = utils.set_detector_thickness(D)

    C = make_crystal.load_crystal("pdbs/3t4x")
    C.mos_spread_deg = 2.60736
    C.n_mos_domains = 1340//2

    S = sim_data.SimData()
    S.crystal = C
    S.beam= make_crystal.load_beam(loader.get_beam())
    S.detector = D0
    S.instantiate_nanoBragg(oversample=1)

    S.D.divergence_hv_mrad = 2e-5,2e-5
    S.D.divsteps_hv = 1,1
    S.D.show_params()

    S.D.add_nanoBragg_spots_cuda()
    img = S.D.raw_pixels.as_numpy_array()
    vol = 125000000
    img *= vol
    print("done")

    #import pylab as plt
    #plt.imshow(img * 125000000, vmax=100000)
    #plt.show()

if __name__=="__main__":
    main()
