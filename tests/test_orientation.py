"""Tests for the canonical-orientation step applied before array extraction."""

import numpy as np
import pytest
import SimpleITK as sitk

from vision_engine.processing.orientation import (
    OrientationInfo,
    orient_to_canonical,
    orientation_code,
    orientation_target_from_config,
    validate_orientation_code,
)


def _tagged_image(direction, shape=(3, 4, 5), spacing=(1.0, 2.0, 3.0)):
    """Volume whose voxel value encodes its (z, y, x) index: z*100 + y*10 + x."""
    z, y, x = shape
    arr = (
        np.arange(z)[:, None, None] * 100
        + np.arange(y)[None, :, None] * 10
        + np.arange(x)[None, None, :]
    ).astype(np.int16)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetDirection(direction)
    return img, arr


LPS = (1, 0, 0, 0, 1, 0, 0, 0, 1)
LPI = (1, 0, 0, 0, 1, 0, 0, 0, -1)
RAS = (-1, 0, 0, 0, -1, 0, 0, 0, 1)


def test_lpi_input_is_a_noop():
    img, arr = _tagged_image(LPI)
    out, info = orient_to_canonical(img, "LPI")
    assert info == OrientationInfo(input="LPI", output="LPI", reoriented=False)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(out), arr)
    assert out.GetSpacing() == img.GetSpacing()


def test_dicom_style_lps_only_reverses_depth():
    """A DICOM series read feet-to-head keeps its in-plane pose, slices reverse."""
    img, arr = _tagged_image(LPS)
    out, info = orient_to_canonical(img, "LPI")
    assert info.input == "LPS" and info.output == "LPI" and info.reoriented
    np.testing.assert_array_equal(sitk.GetArrayFromImage(out), arr[::-1])
    assert orientation_code(out) == "LPI"


def test_ras_nifti_flips_all_three_axes():
    """An RAS NIfTI is spine-up, mirrored and feet-first; all three are fixed."""
    img, arr = _tagged_image(RAS)
    out, info = orient_to_canonical(img, "LPI")
    assert info.input == "RAS" and info.reoriented
    np.testing.assert_array_equal(sitk.GetArrayFromImage(out), arr[::-1, ::-1, ::-1])
    # first emitted voxel is the input's last voxel on every axis
    assert int(sitk.GetArrayFromImage(out)[0, 0, 0]) == 2 * 100 + 3 * 10 + 4


def test_permuted_axes_carry_spacing():
    """Axis order changes (e.g. a sagittal stack) permute the spacing too."""
    # x -> P, y -> S, z -> L : a sagittal acquisition
    sag = (0, 0, 1, 1, 0, 0, 0, 1, 0)
    img, arr = _tagged_image(sag, shape=(3, 4, 5), spacing=(1.0, 2.0, 3.0))
    out, info = orient_to_canonical(img, "LPI")
    assert orientation_code(out) == "LPI"
    # output axes (x=L, y=P, z=I) came from input (z, x, y) -> spacings (3, 1, 2)
    assert out.GetSpacing() == (3.0, 1.0, 2.0)
    assert sitk.GetArrayFromImage(out).shape == (4, 5, 3)  # (z=I, y=P, x=L)


def test_pixel_values_are_never_interpolated():
    img, arr = _tagged_image(RAS)
    out, _ = orient_to_canonical(img, "LPI")
    assert sorted(sitk.GetArrayFromImage(out).ravel()) == sorted(arr.ravel())


def test_target_none_records_but_does_not_reorient():
    img, arr = _tagged_image(RAS)
    out, info = orient_to_canonical(img, None)
    assert info == OrientationInfo(input="RAS", output="RAS", reoriented=False)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(out), arr)


def test_other_targets_are_honoured():
    img, _ = _tagged_image(LPS)
    out, info = orient_to_canonical(img, "ras")
    assert info.output == "RAS" and orientation_code(out) == "RAS"


def test_rejects_non_3d():
    img = sitk.GetImageFromArray(np.zeros((4, 5), dtype=np.int16))
    with pytest.raises(ValueError):
        orient_to_canonical(img, "LPI")


@pytest.mark.parametrize("bad", ["LP", "LPX", "LLS", "LPSI", 3])
def test_rejects_bad_codes(bad):
    with pytest.raises(ValueError):
        validate_orientation_code(bad)


def test_config_default_and_disable():
    assert orientation_target_from_config({}) == "LPI"
    assert orientation_target_from_config(None) == "LPI"
    assert orientation_target_from_config({"orientation": None}) is None
    assert orientation_target_from_config({"orientation": "null"}) is None
    assert orientation_target_from_config({"orientation": "ras"}) == "RAS"
    with pytest.raises(ValueError):
        orientation_target_from_config({"orientation": "XYZ"})
