"""Canonical patient orientation for 3D volumes.

RAVE reads DICOM series and NIfTI files through SimpleITK, which keeps each
input's own voxel order. A DICOM axial series usually comes out with the spine
at the bottom of the frame, the patient's left on the right of the image, and
slices running head to feet. A NIfTI stored in the common RAS convention comes
out reversed on all three axes, so the same scan is rotated 180 degrees
in-plane and runs feet to head. Models trained on one convention silently
mis-read the other.

This module reorients every volume to one declared pose before the array is
extracted. The default, ``"LPI"``, is the Pillar convention:

* ``L``: column index increases toward the patient's Left (radiological view),
* ``P``: row index increases toward Posterior (spine at the bottom),
* ``I``: slice index increases toward Inferior (first slice is the most cranial).

The transform is a pure axis permutation and flip. No resampling happens and
pixel values are untouched, so a volume already in the target pose is a no-op.
It relies on the direction matrix being honest: for NIfTI that is the header
affine, for DICOM it is derived from the file positions in the order they were
read. Inputs whose headers lie (a NIfTI affine with the slice axis reversed,
or a DICOM series with no usable ImagePositionPatient) cannot be detected
here; the recorded input orientation lets a downstream audit find them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import SimpleITK as sitk

logger = logging.getLogger(__name__)

DEFAULT_ORIENTATION = "LPI"
_VALID_LETTERS = {"L", "R", "A", "P", "S", "I"}


@dataclass(frozen=True)
class OrientationInfo:
    """What the reorientation did, for ``metadata.json``."""

    input: str
    """Closest axis-aligned orientation code of the input (e.g. ``"RAS"``)."""
    output: str
    """Orientation code of the emitted volume (e.g. ``"LPI"``)."""
    reoriented: bool
    """False when the input was already in the output pose."""
    convention: str = (
        "ITK/DICOM LPS codes: each letter is the direction the corresponding "
        "voxel axis (x, y, z) advances toward; numpy array order is (z, y, x)."
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_orientation_code(code: str) -> str:
    """Return ``code`` upper-cased, or raise if it is not a 3-letter axis code."""
    if not isinstance(code, str) or len(code) != 3:
        raise ValueError(f"orientation code must be 3 letters, got {code!r}")
    code = code.upper()
    if any(c not in _VALID_LETTERS for c in code):
        raise ValueError(f"orientation code letters must be in {sorted(_VALID_LETTERS)}, got {code!r}")
    axes = {c: ax for ax, pair in enumerate(("LR", "AP", "SI")) for c in pair}
    if sorted(axes[c] for c in code) != [0, 1, 2]:
        raise ValueError(f"orientation code must name each axis once, got {code!r}")
    return code


def orientation_code(image: sitk.Image) -> str:
    """Closest axis-aligned orientation code of ``image``'s direction matrix."""
    return sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(image.GetDirection())


def orient_to_canonical(
    image: sitk.Image, target: Optional[str] = DEFAULT_ORIENTATION
) -> tuple[sitk.Image, OrientationInfo]:
    """Permute/flip ``image`` so its axes advance along ``target``.

    ``target=None`` disables reorientation and only records the input pose,
    for callers that must reproduce caches built before this step existed.
    """
    if image.GetDimension() != 3:
        raise ValueError(f"orient_to_canonical expects a 3D image, got {image.GetDimension()}D")
    before = orientation_code(image)
    if target is None:
        return image, OrientationInfo(input=before, output=before, reoriented=False)
    target = validate_orientation_code(target)
    if before == target:
        return image, OrientationInfo(input=before, output=target, reoriented=False)
    oriented = sitk.DICOMOrient(image, target)
    after = orientation_code(oriented)
    if after != target:
        raise RuntimeError(f"DICOMOrient produced {after}, expected {target}")
    logger.info(f"Reoriented volume {before} -> {target}")
    return oriented, OrientationInfo(input=before, output=target, reoriented=True)


def orientation_target_from_config(processing_config: Any) -> Optional[str]:
    """Read ``processing.orientation`` (default ``"LPI"``; ``null`` disables)."""
    if processing_config is None:
        return DEFAULT_ORIENTATION
    getter = processing_config.get if hasattr(processing_config, "get") else None
    if getter is None:
        return DEFAULT_ORIENTATION
    if "orientation" not in processing_config:
        return DEFAULT_ORIENTATION
    value = getter("orientation")
    if value is None or (isinstance(value, str) and value.lower() in {"", "none", "null"}):
        return None
    return validate_orientation_code(value)
