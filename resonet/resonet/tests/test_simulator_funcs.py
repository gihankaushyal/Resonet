"""Tests for simulator utility functions that don't require simtbx.

shift_center, shift_distance: pure dxtbx operations.
reso2radius: pure math + dxtbx accessor calls.
"""
import math
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# shift_center
# ---------------------------------------------------------------------------

def test_shift_center_moves_origin_along_fast_axis(two_panel_detector):
    """Shifting center by delta_x pixels moves origin by delta_x * pixel_size along fast axis."""
    from resonet.sims.simulator import shift_center
    delta_x = 10.0
    panel = two_panel_detector[0]
    pixel_size = panel.get_pixel_size()[0]
    fast = np.array(panel.get_fast_axis())
    orig_origin = np.array(panel.get_origin())

    shifted = shift_center(two_panel_detector, delta_x, 0.0)
    new_origin = np.array(shifted[0].get_origin())
    expected_shift = fast * pixel_size * delta_x
    np.testing.assert_allclose(new_origin, orig_origin + expected_shift, atol=1e-9)


def test_shift_center_moves_origin_along_slow_axis(two_panel_detector):
    """delta_y shifts origin along slow axis."""
    from resonet.sims.simulator import shift_center
    delta_y = -5.0
    panel = two_panel_detector[0]
    pixel_size = panel.get_pixel_size()[0]
    slow = np.array(panel.get_slow_axis())
    orig_origin = np.array(panel.get_origin())

    shifted = shift_center(two_panel_detector, 0.0, delta_y)
    new_origin = np.array(shifted[0].get_origin())
    expected_shift = slow * pixel_size * delta_y
    np.testing.assert_allclose(new_origin, orig_origin + expected_shift, atol=1e-9)


def test_shift_center_applies_to_all_panels(two_panel_detector):
    """All panels are shifted by the same lab-frame vector (rigid body)."""
    from resonet.sims.simulator import shift_center
    shifted = shift_center(two_panel_detector, 10.0, 0.0)
    assert len(shifted) == len(two_panel_detector)
    displacements = []
    for i in range(len(two_panel_detector)):
        orig = np.array(two_panel_detector[i].get_origin())
        new = np.array(shifted[i].get_origin())
        assert not np.allclose(orig, new), f"Panel {i} origin was not changed"
        displacements.append(new - orig)
    np.testing.assert_allclose(displacements[0], displacements[1], atol=1e-9)


def test_shift_center_zero_is_noop(two_panel_detector):
    """shift_center(0, 0) leaves all origins unchanged."""
    from resonet.sims.simulator import shift_center
    shifted = shift_center(two_panel_detector, 0.0, 0.0)
    for i in range(len(two_panel_detector)):
        orig = np.array(two_panel_detector[i].get_origin())
        new = np.array(shifted[i].get_origin())
        np.testing.assert_allclose(orig, new, atol=1e-10)


def test_shift_center_returns_new_detector(two_panel_detector):
    """shift_center returns a new Detector, not the original."""
    from resonet.sims.simulator import shift_center
    shifted = shift_center(two_panel_detector, 5.0, 0.0)
    assert shifted is not two_panel_detector


# ---------------------------------------------------------------------------
# shift_distance
# ---------------------------------------------------------------------------

def test_shift_distance_changes_z_component(two_panel_detector):
    """Shifting distance moves origin along the panel normal (z for axis-aligned panel)."""
    from resonet.sims.simulator import shift_distance
    delta_z = 5.0  # mm
    panel = two_panel_detector[0]
    fast = np.array(panel.get_fast_axis())
    slow = np.array(panel.get_slow_axis())
    normal = np.cross(fast, slow)
    orig_origin = np.array(panel.get_origin())

    shifted = shift_distance(two_panel_detector, delta_z)
    new_origin = np.array(shifted[0].get_origin())
    expected = orig_origin - normal * delta_z
    np.testing.assert_allclose(new_origin, expected, atol=1e-9)


def test_shift_distance_applies_to_all_panels(two_panel_detector):
    """All panels shifted, not just first."""
    from resonet.sims.simulator import shift_distance
    shifted = shift_distance(two_panel_detector, 10.0)
    assert len(shifted) == len(two_panel_detector)
    for i in range(len(two_panel_detector)):
        orig = np.array(two_panel_detector[i].get_origin())
        new = np.array(shifted[i].get_origin())
        assert not np.allclose(orig, new), f"Panel {i} origin was not changed"


def test_shift_distance_negative(two_panel_detector):
    """Negative delta_z shifts in the opposite direction."""
    from resonet.sims.simulator import shift_distance
    shifted_pos = shift_distance(two_panel_detector, 10.0)
    shifted_neg = shift_distance(two_panel_detector, -10.0)
    orig = np.array(two_panel_detector[0].get_origin())
    pos_origin = np.array(shifted_pos[0].get_origin())
    neg_origin = np.array(shifted_neg[0].get_origin())
    # Positive and negative shifts should be symmetric around origin
    np.testing.assert_allclose(pos_origin + neg_origin, 2 * orig, atol=1e-9)


def test_shift_distance_zero_is_noop(two_panel_detector):
    """shift_distance(0) leaves all origins unchanged."""
    from resonet.sims.simulator import shift_distance
    shifted = shift_distance(two_panel_detector, 0.0)
    for i in range(len(two_panel_detector)):
        orig = np.array(two_panel_detector[i].get_origin())
        new = np.array(shifted[i].get_origin())
        np.testing.assert_allclose(orig, new, atol=1e-10)


# ---------------------------------------------------------------------------
# apply_epix_noise
# ---------------------------------------------------------------------------

def test_apply_epix_noise_output_shape_and_dtype():
    """Output has the same shape as input and is float32."""
    from resonet.sims.make_sims import apply_epix_noise
    img = np.zeros((64, 128), dtype=np.float32)
    out = apply_epix_noise(img, rng=np.random.default_rng(0))
    assert out.shape == img.shape
    assert out.dtype == np.float32


def test_apply_epix_noise_nonnegative():
    """Output is non-negative even for zero-count input."""
    from resonet.sims.make_sims import apply_epix_noise
    img = np.zeros((1000,), dtype=np.float32)
    out = apply_epix_noise(img, rng=np.random.default_rng(42))
    assert np.all(out >= 0), f"Found {np.sum(out < 0)} negative pixels"


def test_apply_epix_noise_saturation_clip():
    """Pixels well above sat_lg are clipped to sat_lg (contract: no pixel exceeds sat_lg)."""
    from resonet.sims.make_sims import apply_epix_noise
    img = np.full((200,), 50000.0, dtype=np.float32)
    out = apply_epix_noise(img, sat_lg=11000, rng=np.random.default_rng(7))
    assert np.max(out) <= 11000, f"max {np.max(out)} exceeds sat_lg=11000"


def test_apply_epix_noise_default_rng():
    """rng=None constructs an internal RNG; function runs without error."""
    from resonet.sims.make_sims import apply_epix_noise
    out = apply_epix_noise(np.zeros((20,), dtype=np.float32))
    assert out.shape == (20,)
    assert np.all(out >= 0)


def test_apply_epix_noise_hg_zone_receives_sigma_hg():
    """HG pixels (≤ t1) receive sigma_hg noise; nonzero sigma_hg produces more spread than zero."""
    from resonet.sims.make_sims import apply_epix_noise
    img = np.full((2000,), 50.0, dtype=np.float32)  # all in HG zone (t1=80)
    out_large = apply_epix_noise(img, t1=80, t2=270,
                                 sigma_hg=30.0, sigma_mg=0.0, sigma_lg=0.0,
                                 sat_lg=int(1e9), rng=np.random.default_rng(0))
    out_zero = apply_epix_noise(img, t1=80, t2=270,
                                sigma_hg=0.0, sigma_mg=0.0, sigma_lg=0.0,
                                sat_lg=int(1e9), rng=np.random.default_rng(0))
    assert np.std(out_large) > np.std(out_zero) + 5.0, (
        f"sigma_hg=30 std={np.std(out_large):.2f} not significantly > sigma_hg=0 std={np.std(out_zero):.2f}"
    )


def test_apply_epix_noise_mg_zone_receives_sigma_mg():
    """MG pixels ((t1, t2]) receive sigma_mg noise, not sigma_hg or sigma_lg."""
    from resonet.sims.make_sims import apply_epix_noise
    img = np.full((2000,), 150.0, dtype=np.float32)  # all in MG zone
    out_large = apply_epix_noise(img, t1=80, t2=270,
                                 sigma_hg=0.0, sigma_mg=30.0, sigma_lg=0.0,
                                 sat_lg=int(1e9), rng=np.random.default_rng(0))
    out_zero = apply_epix_noise(img, t1=80, t2=270,
                                sigma_hg=0.0, sigma_mg=0.0, sigma_lg=0.0,
                                sat_lg=int(1e9), rng=np.random.default_rng(0))
    assert np.std(out_large) > np.std(out_zero) + 5.0, (
        f"sigma_mg=30 std={np.std(out_large):.2f} not significantly > sigma_mg=0 std={np.std(out_zero):.2f}"
    )


def test_apply_epix_noise_lg_zone_receives_sigma_lg():
    """LG pixels (> t2) receive sigma_lg noise, not sigma_hg or sigma_mg."""
    from resonet.sims.make_sims import apply_epix_noise
    img = np.full((2000,), 500.0, dtype=np.float32)  # all in LG zone (t2=270)
    out_large = apply_epix_noise(img, t1=80, t2=270,
                                 sigma_hg=0.0, sigma_mg=0.0, sigma_lg=30.0,
                                 sat_lg=int(1e9), rng=np.random.default_rng(0))
    out_zero = apply_epix_noise(img, t1=80, t2=270,
                                sigma_hg=0.0, sigma_mg=0.0, sigma_lg=0.0,
                                sat_lg=int(1e9), rng=np.random.default_rng(0))
    assert np.std(out_large) > np.std(out_zero) + 5.0, (
        f"sigma_lg=30 std={np.std(out_large):.2f} not significantly > sigma_lg=0 std={np.std(out_zero):.2f}"
    )


def test_apply_epix_noise_explicit_thresholds():
    """t1, t2, sigma_* are wired correctly: custom thresholds change zone boundaries."""
    from resonet.sims.make_sims import apply_epix_noise
    rng = np.random.default_rng(5)
    # With t1=200, t2=400: img=150 is in HG zone; with sigma_hg=50, sigma_mg=0 → large spread
    img = np.full((1000,), 150.0, dtype=np.float32)
    out = apply_epix_noise(img, t1=200, t2=400,
                           sigma_hg=50.0, sigma_mg=0.0, sigma_lg=0.0,
                           sat_lg=int(1e9), rng=rng)
    assert np.std(out) > 10.0, "Custom t1=200 should place img=150 in HG zone with large sigma_hg"


# ---------------------------------------------------------------------------
# apply_epix_noise input validation
# ---------------------------------------------------------------------------

def test_apply_epix_noise_t1_ge_t2_raises():
    """t1 >= t2 raises ValueError immediately (would otherwise silently corrupt zone masks)."""
    from resonet.sims.make_sims import apply_epix_noise
    with pytest.raises(ValueError, match="t1"):
        apply_epix_noise(np.zeros((10,), dtype=np.float32), t1=270, t2=80)


def test_apply_epix_noise_t1_equal_t2_raises():
    from resonet.sims.make_sims import apply_epix_noise
    with pytest.raises(ValueError, match="t1"):
        apply_epix_noise(np.zeros((10,), dtype=np.float32), t1=100, t2=100)


def test_apply_epix_noise_sat_lg_le_t2_raises():
    """sat_lg <= t2 raises ValueError."""
    from resonet.sims.make_sims import apply_epix_noise
    with pytest.raises(ValueError, match="sat_lg"):
        apply_epix_noise(np.zeros((10,), dtype=np.float32), t1=80, t2=270, sat_lg=270)


def test_apply_epix_noise_negative_sigma_raises():
    """Negative sigma raises ValueError."""
    from resonet.sims.make_sims import apply_epix_noise
    with pytest.raises(ValueError, match="sigma"):
        apply_epix_noise(np.zeros((10,), dtype=np.float32), sigma_hg=-0.1)


def test_apply_epix_noise_deterministic_with_rng():
    """Same RNG seed produces identical output."""
    from resonet.sims.make_sims import apply_epix_noise
    img = (np.random.default_rng(1).random((128,)) * 500).astype(np.float32)
    out1 = apply_epix_noise(img, rng=np.random.default_rng(99))
    out2 = apply_epix_noise(img, rng=np.random.default_rng(99))
    np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# reso2radius
# ---------------------------------------------------------------------------

class _MockBeam:
    def get_wavelength(self): return 1.0  # Angstroms


class _MockPanel:
    def get_distance(self): return 100.0  # mm
    def get_pixel_size(self): return (0.1, 0.1)  # mm


class _MockDet:
    def __getitem__(self, i): return _MockPanel()


def test_reso2radius_formula():
    """reso2radius matches the geometric formula: tan(2*arcsin(λ/2r)) * dist/pixsize."""
    from resonet.sims.simulator import reso2radius
    wavelen = 1.0   # Angstroms
    dist_mm = 100.0
    pixsize_mm = 0.1
    reso = 2.0  # Angstroms

    det = _MockDet()
    beam = _MockBeam()
    result = reso2radius(reso, det, beam)

    theta = math.asin(wavelen / (2 * reso))
    expected = math.tan(2 * theta) * dist_mm / pixsize_mm
    assert abs(result - expected) < 1e-6


# ---------------------------------------------------------------------------
# apply_jungfrau_noise — behavioral tests
# ---------------------------------------------------------------------------

def test_apply_jungfrau_noise_output_shape_and_dtype():
    """Output shape matches input; dtype is float32."""
    from resonet.sims.make_sims import apply_jungfrau_noise
    img = np.ones((64, 64), dtype=np.float32) * 10.0
    out = apply_jungfrau_noise(img, rng=np.random.default_rng(0))
    assert out.shape == img.shape
    assert out.dtype == np.float32


def test_apply_jungfrau_noise_nonnegative():
    """All output pixels are >= 0."""
    from resonet.sims.make_sims import apply_jungfrau_noise
    img = np.zeros((200, 200), dtype=np.float32)
    out = apply_jungfrau_noise(img, rng=np.random.default_rng(1))
    assert np.all(out >= 0)


def test_apply_jungfrau_noise_saturation_clip():
    """No output pixel exceeds sat_g2."""
    from resonet.sims.make_sims import apply_jungfrau_noise
    img = np.full((500,), 1e6, dtype=np.float32)
    out = apply_jungfrau_noise(img, sat_g2=3400, rng=np.random.default_rng(2))
    assert np.all(out <= 3400)


def test_apply_jungfrau_noise_default_rng():
    """rng=None constructs an internal RNG and runs without error."""
    from resonet.sims.make_sims import apply_jungfrau_noise
    img = np.ones((10, 10), dtype=np.float32) * 5.0
    out = apply_jungfrau_noise(img)
    assert out.shape == (10, 10)


def test_apply_jungfrau_noise_t1_ge_t2_raises():
    """t1 >= t2 raises ValueError."""
    from resonet.sims.make_sims import apply_jungfrau_noise
    with pytest.raises(ValueError, match="t1"):
        apply_jungfrau_noise(np.ones((10,), dtype=np.float32), t1=500, t2=100)


def test_apply_jungfrau_noise_t1_equal_t2_raises():
    """t1 == t2 raises ValueError."""
    from resonet.sims.make_sims import apply_jungfrau_noise
    with pytest.raises(ValueError, match="t1"):
        apply_jungfrau_noise(np.ones((10,), dtype=np.float32), t1=200, t2=200)


def test_apply_jungfrau_noise_sat_g2_le_t2_raises():
    """sat_g2 <= t2 raises ValueError."""
    from resonet.sims.make_sims import apply_jungfrau_noise
    with pytest.raises(ValueError, match="sat_g2"):
        apply_jungfrau_noise(np.ones((10,), dtype=np.float32), t1=34, t2=342, sat_g2=300)


def test_apply_jungfrau_noise_negative_sigma_raises():
    """Negative sigma raises ValueError."""
    from resonet.sims.make_sims import apply_jungfrau_noise
    with pytest.raises(ValueError, match="sigma"):
        apply_jungfrau_noise(np.ones((10,), dtype=np.float32), sigma_g0=-0.1)


def test_apply_jungfrau_noise_g0_zone_receives_sigma_g0():
    """G0 pixels (<= t1) receive sigma_g0 noise; larger sigma_g0 → more spread."""
    from resonet.sims.make_sims import apply_jungfrau_noise
    img = np.full((2000,), 10.0, dtype=np.float32)  # all in G0 zone (t1=34)
    out_large = apply_jungfrau_noise(img, t1=34, t2=342,
                                     sigma_g0=30.0, sigma_g1=0.0, sigma_g2=0.0,
                                     sat_g2=int(1e9), rng=np.random.default_rng(0))
    out_zero = apply_jungfrau_noise(img, t1=34, t2=342,
                                    sigma_g0=0.0, sigma_g1=0.0, sigma_g2=0.0,
                                    sat_g2=int(1e9), rng=np.random.default_rng(0))
    assert np.std(out_large) > np.std(out_zero) + 5.0


def test_apply_jungfrau_noise_g1_zone_receives_sigma_g1():
    """G1 pixels ((t1, t2]) receive sigma_g1 noise, not sigma_g0 or sigma_g2."""
    from resonet.sims.make_sims import apply_jungfrau_noise
    img = np.full((2000,), 100.0, dtype=np.float32)  # all in G1 zone (34 < 100 <= 342)
    out_large = apply_jungfrau_noise(img, t1=34, t2=342,
                                     sigma_g0=0.0, sigma_g1=30.0, sigma_g2=0.0,
                                     sat_g2=int(1e9), rng=np.random.default_rng(0))
    out_zero = apply_jungfrau_noise(img, t1=34, t2=342,
                                    sigma_g0=0.0, sigma_g1=0.0, sigma_g2=0.0,
                                    sat_g2=int(1e9), rng=np.random.default_rng(0))
    assert np.std(out_large) > np.std(out_zero) + 5.0


def test_apply_jungfrau_noise_g2_zone_receives_sigma_g2():
    """G2 pixels (> t2) receive sigma_g2 noise, not sigma_g0 or sigma_g1."""
    from resonet.sims.make_sims import apply_jungfrau_noise
    img = np.full((2000,), 500.0, dtype=np.float32)  # all in G2 zone (t2=342)
    out_large = apply_jungfrau_noise(img, t1=34, t2=342,
                                     sigma_g0=0.0, sigma_g1=0.0, sigma_g2=30.0,
                                     sat_g2=int(1e9), rng=np.random.default_rng(0))
    out_zero = apply_jungfrau_noise(img, t1=34, t2=342,
                                    sigma_g0=0.0, sigma_g1=0.0, sigma_g2=0.0,
                                    sat_g2=int(1e9), rng=np.random.default_rng(0))
    assert np.std(out_large) > np.std(out_zero) + 5.0, (
        f"sigma_g2=30 std={np.std(out_large):.2f} not significantly > sigma_g2=0 std={np.std(out_zero):.2f}"
    )


def test_apply_jungfrau_noise_explicit_thresholds():
    """Custom t1, t2 correctly shift zone boundaries."""
    from resonet.sims.make_sims import apply_jungfrau_noise
    rng = np.random.default_rng(7)
    # With t1=200, t2=400: img=150 is in G0; with sigma_g0=50 → large spread
    img = np.full((1000,), 150.0, dtype=np.float32)
    out = apply_jungfrau_noise(img, t1=200, t2=400,
                               sigma_g0=50.0, sigma_g1=0.0, sigma_g2=0.0,
                               sat_g2=int(1e9), rng=rng)
    assert np.std(out) > 10.0, "Custom t1=200 should place img=150 in G0 zone with large sigma_g0"


def test_apply_jungfrau_noise_deterministic_with_rng():
    """Same RNG seed produces identical output."""
    from resonet.sims.make_sims import apply_jungfrau_noise
    img = np.random.default_rng(42).random((100, 100)).astype(np.float32) * 50
    out1 = apply_jungfrau_noise(img, rng=np.random.default_rng(99))
    out2 = apply_jungfrau_noise(img, rng=np.random.default_rng(99))
    np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------

def test_reso2radius_high_reso_gives_large_radius():
    """Higher resolution (small d-spacing) → larger pixel radius."""
    from resonet.sims.simulator import reso2radius
    det = _MockDet()
    beam = _MockBeam()
    r_hi = reso2radius(1.5, det, beam)  # high reso (small d) → large angle
    r_lo = reso2radius(10.0, det, beam)  # low reso (large d) → small angle
    assert r_hi > r_lo
