"""Deterministic transformations: identity, 3 × 120°, 90°, per-axis scale, non-commutativity."""

import itertools

import numpy as np
import pytest

from ecg3kg import Transform, ecg_to_vcg


def test_identity_zero_transform(ecg_standard, ecg_with_residual):
    t = Transform()
    assert t.is_identity()
    np.testing.assert_allclose(t.matrix(), np.eye(3), atol=0)
    np.testing.assert_allclose(t.apply_ecg(ecg_standard, leads="standard"), ecg_standard, rtol=1e-6)
    out = t.apply_ecg(ecg_with_residual, leads="standard")
    np.testing.assert_array_equal(out, ecg_with_residual)
    assert not Transform(angles_deg=(1, 0, 0)).is_identity()
    assert not Transform(scales=(1, 1, 1.01)).is_identity()


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_three_rotations_of_120_deg_compose_to_identity(ecg_standard, ecg_with_residual, axis):
    angles = [0.0, 0.0, 0.0]
    angles[axis] = 120.0
    t = Transform(angles_deg=tuple(angles))
    for ecg in (ecg_standard, ecg_with_residual):
        once = t.apply_ecg(ecg, leads="standard")
        assert np.abs(once - ecg).max() > 1e-2  # a single 120° rotation is not the identity
        thrice = t.apply_ecg(t.apply_ecg(once, leads="standard"), leads="standard")
        np.testing.assert_allclose(thrice, ecg, rtol=1e-6, atol=1e-9)
    # Also on the rotation matrix itself.
    R = t.rotation_matrix()
    np.testing.assert_allclose(R @ R @ R, np.eye(3), atol=1e-12)


def test_rotation_90_maps_y_to_z():
    """Right-hand rule: rotating +90° about x sends +y onto +z."""
    t = np.linspace(0, 2 * np.pi, 100)
    vcg = np.stack([np.zeros_like(t), np.sin(t), np.zeros_like(t)])  # along y only
    out = Transform(angles_deg=(90.0, 0.0, 0.0)).apply_vcg(vcg)
    np.testing.assert_allclose(out[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(out[1], 0.0, atol=1e-12)
    np.testing.assert_allclose(out[2], np.sin(t), atol=1e-12)


def test_rotation_matrix_orthogonal_det_one(rng):
    for _ in range(20):
        angles = tuple(rng.uniform(-180, 180, 3))
        for perm in itertools.permutations(("x", "y", "z")):
            R = Transform(angles_deg=angles, axes_order=perm).rotation_matrix()
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
            assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)


def test_scale_per_axis(vcg_synthetic):
    out = Transform(scales=(2.0, 1.0, 1.0)).apply_vcg(vcg_synthetic)
    np.testing.assert_allclose(out[0], 2 * vcg_synthetic[0], rtol=1e-12)
    np.testing.assert_array_equal(out[1:], vcg_synthetic[1:])
    S = Transform(scales=(2.0, 3.0, 0.5)).scale_matrix()
    np.testing.assert_array_equal(S, np.diag([2.0, 3.0, 0.5]))


def test_scale_on_ecg_doubles_vcg_axis(ecg_standard):
    out = Transform(scales=(2.0, 1.0, 1.0)).apply_ecg(ecg_standard, leads="standard")
    v_in = ecg_to_vcg(ecg_standard, leads="standard")
    v_out = ecg_to_vcg(out, leads="standard")
    np.testing.assert_allclose(v_out[0], 2 * v_in[0], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(v_out[1:], v_in[1:], rtol=1e-6, atol=1e-12)


def test_rotate_scale_not_commutative(ecg_standard):
    # Rotation about z with a scale along x: S·R ≠ R·S (a scale along the rotation axis commutes).
    a = Transform(angles_deg=(0, 0, 30), scales=(2, 1, 1), order="rotate_then_scale")
    b = Transform(angles_deg=(0, 0, 30), scales=(2, 1, 1), order="scale_then_rotate")
    assert np.abs(a.matrix() - b.matrix()).max() > 1e-3
    assert np.abs(
        a.apply_ecg(ecg_standard, leads="standard") - b.apply_ecg(ecg_standard, leads="standard")
    ).max() > 1e-3
    # Isotropic scale commutes with any rotation.
    a2 = Transform(angles_deg=(30, 40, 50), scales=(1.5, 1.5, 1.5), order="rotate_then_scale")
    b2 = Transform(angles_deg=(30, 40, 50), scales=(1.5, 1.5, 1.5), order="scale_then_rotate")
    np.testing.assert_allclose(a2.matrix(), b2.matrix(), atol=1e-12)
    # And the matrices are what they say: S·R vs R·S.
    R, S = a.rotation_matrix(), a.scale_matrix()
    np.testing.assert_array_equal(a.matrix(), S @ R)
    np.testing.assert_array_equal(b.matrix(), R @ S)


def test_axes_order_matters():
    a = Transform(angles_deg=(30, 40, 0), axes_order=("x", "y", "z"))
    b = Transform(angles_deg=(30, 40, 0), axes_order=("y", "x", "z"))
    assert np.abs(a.rotation_matrix() - b.rotation_matrix()).max() > 1e-3
    # (x, y, z) with angles (θ1, θ2, θ3): R = Rz(θ3) · Ry(θ2) · Rx(θ1) — first axis applied first.
    c = Transform(angles_deg=(30, 0, 0), axes_order=("y", "x", "z"))  # 30° about y first
    d = Transform(angles_deg=(0, 30, 0), axes_order=("x", "y", "z"))  # 30° about y, same thing
    np.testing.assert_allclose(c.rotation_matrix(), d.rotation_matrix(), atol=1e-12)


def test_compose_matrix():
    a = Transform(angles_deg=(10, 20, 30), scales=(1.1, 0.9, 1.0))
    b = Transform(angles_deg=(-5, 15, 0), scales=(1.0, 1.2, 0.8))
    np.testing.assert_array_equal(a.compose(b), a.matrix() @ b.matrix())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"angles_deg": (181, 0, 0)},
        {"angles_deg": (0, -180.5, 0)},
        {"scales": (0, 1, 1)},
        {"scales": (1, -1, 1)},
        {"axes_order": ("x", "x", "z")},
        {"axes_order": ("x", "y")},
        {"axes_order": ("x", "y", "w")},
        {"order": "scale_rotate"},
        {"angles_deg": (0, 0)},
        {"scales": (1, 1, 1, 1)},
        {"angles_deg": (float("nan"), 0, 0)},
    ],
)
def test_validation(kwargs):
    with pytest.raises(ValueError):
        Transform(**kwargs)


def test_transform_is_frozen_and_hashable():
    t = Transform(angles_deg=(1, 2, 3))
    with pytest.raises(AttributeError):  # frozen dataclass
        t.angles_deg = (0, 0, 0)  # type: ignore[misc]
    assert hash(t) == hash(Transform(angles_deg=(1, 2, 3)))
