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
    """All panels in a multi-panel detector are shifted, not just the first."""
    from resonet.sims.simulator import shift_center
    shifted = shift_center(two_panel_detector, 10.0, 0.0)
    assert len(shifted) == len(two_panel_detector)
    for i in range(len(two_panel_detector)):
        orig = np.array(two_panel_detector[i].get_origin())
        new = np.array(shifted[i].get_origin())
        assert not np.allclose(orig, new), f"Panel {i} origin was not changed"


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


def test_reso2radius_high_reso_gives_large_radius():
    """Higher resolution (small d-spacing) → larger pixel radius."""
    from resonet.sims.simulator import reso2radius
    det = _MockDet()
    beam = _MockBeam()
    r_hi = reso2radius(1.5, det, beam)  # high reso (small d) → large angle
    r_lo = reso2radius(10.0, det, beam)  # low reso (large d) → small angle
    assert r_hi > r_lo
