"""
MR processor for breast MR and related sequences.
Pipeline: DICOM/NIfTI → resample (linear), slice-select, crop/pad, min-max map to uint16.
Auto-detects segmentation masks to preserve labels and use NN resampling.
"""

import os
import tempfile
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
from PIL import Image
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from ..core.data_structures import SeriesInfo, ProcessedSeries
from ..core.exceptions import ProcessingError
from .base_processor import BaseProcessor
from .orientation import OrientationInfo, orient_to_canonical, orientation_target_from_config

logger = logging.getLogger(__name__)


class MRProcessor(BaseProcessor):
    """MR processor that preserves raw signal intensity values"""

    def __init__(self, config):
        super().__init__(config)
        self.anatomy = config.anatomy
        self._last_resample_shape: Optional[Tuple[int, int, int]] = None
        # Tracks mask detection result for the most recently processed series
        self._last_is_mask: Optional[bool] = None
        self.orientation_target = orientation_target_from_config(config.processing)
        self._last_orientation: Optional[OrientationInfo] = None

    def process_series(self, series_info: SeriesInfo) -> ProcessedSeries:
        """Complete processing pipeline for one MR series"""
        logger.info(f"Processing MR series: {series_info}")

        try:
            # Step 1: DICOM → SimpleITK image with spacing info
            sitk_image, spacing = self._convert_dicom_to_nifti(series_info)

            # Step 2: Process volume (resampling, slice selection, resizing)
            processed_volume = self._process_volume(sitk_image, spacing, series_info)

            # Get processing configuration
            resampling_config = self.config.processing.get("resampling", {})
            target_spacing = resampling_config.get("target_spacing", None)
            crop_pad_config = self.config.processing.get("crop_pad", {})
            target_size = crop_pad_config.get("size", None)

            # Create processing metadata
            processing_metadata = {
                "modality": "MR",
                "anatomy": self.anatomy,
                "original_spacing": spacing.tolist(),
                "target_spacing": target_spacing,
                "original_shape": [
                    int(d) for d in sitk_image.GetSize()[::-1]
                ],  # Convert to Z,Y,X order
                "resample_shape": list(self._last_resample_shape)
                if self._last_resample_shape is not None
                else None,
                "final_shape": processed_volume.shape,
                "crop_pad_target": target_size,
                "slice_count": processed_volume.shape[0],
                "value_range": "raw_signal_intensity",  # MR uses signal intensity, not HU
                "orientation": self._last_orientation.to_dict()
                if self._last_orientation is not None
                else None,
                "series_instance_uid": series_info.series_uid,
                "slice_info": getattr(
                    series_info, "slice_info", []
                ),  # Will be populated if available
                "is_mask": bool(self._last_is_mask),
            }

            return ProcessedSeries(
                series_info=series_info,
                numpy_volume=processed_volume,
                processing_metadata=processing_metadata,
            )

        except Exception as e:
            raise ProcessingError(
                f"Failed to process MR series {series_info.accession}.{series_info.series_number}: {e}"
            )

    def _convert_dicom_to_nifti(
        self, series_info: SeriesInfo
    ) -> Tuple[sitk.Image, np.ndarray]:
        """Convert DICOM series to SimpleITK image with spacing info"""
        path = series_info.get_dicom_path()

        if not os.path.exists(path):
            raise ProcessingError(f"Path not found: {path}")

        # Check if this is a NIfTI file
        path_obj = Path(path)
        if path_obj.is_file() and path_obj.suffix in [".nii", ".gz"]:
            # This is already a NIfTI file, load it directly
            logger.debug(f"Loading NIfTI file: {path}")

            # Use SimpleITK to load NIfTI (consistent with DICOM loading)
            sitk_image = sitk.ReadImage(str(path))
            spacing = np.array(sitk_image.GetSpacing())

            logger.debug(
                f"Loaded NIfTI volume shape: {sitk_image.GetSize()}, spacing: {spacing}"
            )
            return sitk_image, spacing

        try:
            # It's a DICOM directory, convert as usual
            return self._sitk_convert(path)
        except Exception as e:
            raise ProcessingError(f"DICOM conversion failed: {e}")

    def _sitk_convert(self, dicom_path: str) -> Tuple[sitk.Image, np.ndarray]:
        """Convert using SimpleITK and return image with spacing"""
        logger.debug(f"Converting DICOM series using SimpleITK: {dicom_path}")

        # Get DICOM file names
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)

        if not dicom_names:
            raise ProcessingError(f"No DICOM files found in: {dicom_path}")

        # Sort by instance number and position
        dicom_names = self._sort_dicom_files(dicom_names)

        # Read series
        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        # Get spacing (x, y, z)
        spacing = np.array(image.GetSpacing())
        logger.debug(f"Original spacing: {spacing} mm")

        # Log image info
        volume_shape = image.GetSize()[::-1]  # Convert to numpy order (z,y,x)
        logger.debug(f"Image shape: {volume_shape}")

        # Check pixel type
        pixel_type = image.GetPixelIDTypeAsString()
        logger.debug(f"Pixel type: {pixel_type}")

        return image, spacing

    def _sort_dicom_files(self, dicom_names: List[str]) -> List[str]:
        """Sort DICOM files by instance number and position"""
        import pydicom

        positions = []
        for dcm_file in dicom_names:
            try:
                dcm = pydicom.dcmread(dcm_file, stop_before_pixels=True)
                if hasattr(dcm, "ImagePositionPatient") and hasattr(
                    dcm, "InstanceNumber"
                ):
                    positions.append(
                        {
                            "FileName": dcm_file,
                            "SliceLocation": float(dcm.ImagePositionPatient[-1]),
                            "InstanceNumber": dcm.InstanceNumber,
                        }
                    )
            except Exception:
                continue

        if not positions:
            return dicom_names  # Fallback to original order

        # Sort by instance number
        import pandas as pd

        df = pd.DataFrame(positions).sort_values("InstanceNumber")
        return list(df["FileName"])

    def _process_volume(
        self, sitk_image: sitk.Image, spacing: np.ndarray, series_info: SeriesInfo
    ) -> np.ndarray:
        """Process volume with 3D resampling and resizing"""
        # Step 1: Apply 3D resampling to target spacing if configured
        resampling_config = self.config.processing.get("resampling", {})
        target_spacing = resampling_config.get("target_spacing")

        # Detect masks early to control resampling and downstream processing
        is_mask = self._detect_mask(sitk_image, series_info)
        self._last_is_mask = is_mask
        if target_spacing:
            sitk_image = self._resample_volume(
                sitk_image, target_spacing, is_mask=is_mask
            )

        # Optional: save intermediate resampled NIfTI for debugging
        debug_cfg = (
            self.config.processing.get("debug", {})
            if isinstance(self.config.processing, dict)
            else {}
        )
        if debug_cfg.get("save_resampled_nifti", False):
            try:
                debug_dir = debug_cfg.get("output_dir") or os.path.join(
                    os.getcwd(), "test_output_mr", "intermediate_nifti"
                )
                os.makedirs(debug_dir, exist_ok=True)
                # Build filename using series UID or description
                series_name = getattr(series_info, "series_uid", None) or getattr(
                    series_info, "series_description", "series"
                )
                # Sanitize filename
                safe_name = str(series_name).replace("/", "_")
                out_path = os.path.join(debug_dir, f"{safe_name}_resampled.nii.gz")
                sitk.WriteImage(sitk_image, out_path)
                logger.info(f"Saved intermediate resampled NIfTI: {out_path}")
            except Exception as e:
                logger.warning(f"Failed to save intermediate NIfTI: {e}")

        # Put every input (DICOM or NIfTI) in the same patient pose before the
        # array is extracted; a label volume permutes/flips the same way.
        sitk_image, self._last_orientation = orient_to_canonical(
            sitk_image, self.orientation_target
        )

        # Convert to numpy array after resampling
        volume = sitk.GetArrayFromImage(sitk_image)  # Shape: (Z, Y, X)
        # Record resampled shape prior to any slice selection or crop/pad
        self._last_resample_shape = tuple(volume.shape)
        logger.info(
            f"Resampled volume shape (pre slice/crop): {self._last_resample_shape}"
        )

        # For MR, preserve the original data type (typically uint16 or float)
        # Don't force int16 like CT does since MR doesn't use Hounsfield units
        logger.debug(
            f"MR volume dtype: {volume.dtype}, shape: {volume.shape}, "
            f"range: [{volume.min()}, {volume.max()}]"
        )

        # Apply MR intensity mapping unless it's a mask
        vol_float = volume.astype(np.float32, copy=False)
        if not is_mask:
            low_val = float(vol_float.min())
            high_val = float(vol_float.max())
            if high_val > low_val:
                mapped = (vol_float - low_val) / (high_val - low_val)
                volume = (mapped * 65535.0).astype(np.uint16)
                logger.info(
                    f"MR → uint16 via minmax: [{low_val:.3f}, {high_val:.3f}] → [0, 65535]"
                )
            else:
                volume = np.zeros_like(vol_float, dtype=np.uint16)
                logger.info("MR volume has no range, converting to zeros")
        else:
            # Preserve mask-like volumes
            volume = volume.astype(np.uint16, copy=False)
            logger.info("Detected mask-like volume; preserving as-is")

        logger.debug(
            f"Volume after resampling: dtype={volume.dtype}, shape={volume.shape}, "
            f"range=[{volume.min()}, {volume.max()}]"
        )

        # Step 2: Apply crop/pad with integrated slice selection
        crop_pad = self.config.processing.get("crop_pad")
        if crop_pad:
            target_size = crop_pad.get("size", [256, 256])  # H, W
            slice_selection = self.config.processing.get("slice_selection", {})
            volume = self._crop_pad_volume(volume, target_size, slice_selection)
        else:
            # Just apply slice selection if no crop/pad
            volume = self._apply_slice_selection(volume)

        logger.debug(f"Processed volume: dtype={volume.dtype}, shape={volume.shape}")
        return volume

    def _resample_volume(
        self, image: sitk.Image, target_spacing: List[float], is_mask: bool = False
    ) -> sitk.Image:
        """Resample volume to target spacing using SimpleITK.
        Uses nearest-neighbor interpolation for masks to preserve labels; linear for others."""
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()

        # Calculate new size based on spacing change
        new_size = [
            int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
            for i in range(3)
        ]

        logger.info(f"Resampling from spacing {original_spacing} to {target_spacing}")
        logger.info(f"Size change: {original_size} → {new_size}")

        # Set up the resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(0)

        # Use nearest-neighbor for masks; linear for continuous MR data
        resampler.SetInterpolator(
            sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
        )

        # Execute resampling
        resampled = resampler.Execute(image)

        return resampled

    def _detect_mask(self, image: sitk.Image, series_info: SeriesInfo) -> bool:
        """Heuristically detect if the series is a segmentation mask.
        Criteria:
          - Series description or path hints (contains 'mask'/'seg'/'label')
          - Non-negative small max intensity (<= 20)
          - Low number of unique values in a shrunk sample (<= 8)
        """
        try:
            desc = (getattr(series_info, "series_description", "") or "").lower()
            name_hint = (getattr(series_info, "series_uid", "") or "").lower()
            path_hint = os.path.basename(series_info.get_dicom_path() or "").lower()
            if (
                any(k in desc for k in ["mask", "seg", "label"])
                or any(k in path_hint for k in ["mask", "seg", "label"])
                or "mask" in name_hint
            ):
                logger.debug("Mask detection: name/description hint matched")
                hint = True
            else:
                hint = False

            # Quick stats
            stats = sitk.StatisticsImageFilter()
            stats.Execute(image)
            min_val = float(stats.GetMinimum())
            max_val = float(stats.GetMaximum())

            # Sample uniqueness via shrink to limit memory
            shrink_factors = [max(1, s // 256) for s in image.GetSize()]
            try:
                sample_img = sitk.Shrink(image, shrink_factors)
            except Exception:
                sample_img = image
            sample_np = sitk.GetArrayFromImage(sample_img)
            try:
                # Use a coarse subsample if still large
                flat = sample_np.ravel()
                step = max(1, flat.size // 200000)
                unique_count = int(np.unique(flat[::step]).size)
            except Exception:
                unique_count = 9999

            is_integer_type = image.GetPixelID() in [
                sitk.sitkUInt8,
                sitk.sitkUInt16,
                sitk.sitkUInt32,
                sitk.sitkInt8,
                sitk.sitkInt16,
                sitk.sitkInt32,
            ]

            is_mask = (
                max_val <= 20.0
                and min_val >= 0.0
                and unique_count <= 8
                and is_integer_type
            ) or hint
            logger.info(
                f"Mask detection: hint={hint}, min={min_val:.3g}, max={max_val:.3g}, unique~{unique_count}, integer_type={is_integer_type} -> is_mask={is_mask}"
            )
            return bool(is_mask)
        except Exception as e:
            logger.warning(f"Mask detection failed, defaulting to non-mask: {e}")
            return False

    def _crop_pad_volume(
        self,
        volume: np.ndarray,
        target_size: List[int],
        slice_selection: Dict[str, Any],
    ) -> np.ndarray:
        """Crop or pad volume to target size with integrated slice selection"""
        original_shape = volume.shape
        logger.info(f"Original volume shape: {original_shape}")

        # Step 1: Handle slice selection (Z dimension)
        if slice_selection.get("enabled", True):
            num_slices = slice_selection.get("slices")
            if num_slices and volume.shape[0] > num_slices:
                # Select middle slices
                start_idx = (volume.shape[0] - num_slices) // 2
                end_idx = start_idx + num_slices
                volume = volume[start_idx:end_idx]
                logger.info(
                    f"Selected middle {num_slices} slices from {original_shape[0]} total slices "
                    f"(indices {start_idx}:{end_idx}) for {self.anatomy} MR"
                )

        after_slice_shape = volume.shape
        logger.info(f"After slice selection: {after_slice_shape}")

        # Step 2: Center crop or pad using simplified logic
        volume = self._center_crop_pad_3d(volume, target_size)

        final_shape = volume.shape
        logger.info(f"Final volume shape after crop/pad: {final_shape}")

        return volume

    def _center_crop_pad_3d(
        self, volume: np.ndarray, target_hw: List[int]
    ) -> np.ndarray:
        """Center crop or pad a 3D volume to target H,W dimensions"""
        # Preserve original dtype and get padding value
        original_dtype = volume.dtype
        pad_value = float(volume.min())

        # Convert to torch tensor (shape: D, H, W)
        volume_tensor = torch.from_numpy(volume).float()

        # Get current and target sizes
        _, current_h, current_w = volume_tensor.shape
        target_h, target_w = target_hw

        # Calculate padding/cropping for each dimension
        h_diff = target_h - current_h
        w_diff = target_w - current_w

        # Apply center crop or pad
        if h_diff < 0:  # Need to crop height
            h_start = (-h_diff) // 2
            volume_tensor = volume_tensor[:, h_start : h_start + target_h, :]
            logger.info(f"Cropped H dimension from {current_h} to {target_h}")
        elif h_diff > 0:  # Need to pad height
            h_pad = (h_diff // 2, h_diff - h_diff // 2)
            # F.pad expects padding in reverse order: (left, right, top, bottom)
            volume_tensor = F.pad(
                volume_tensor, (0, 0, *h_pad), mode="constant", value=pad_value
            )
            logger.info(f"Padded H dimension from {current_h} to {target_h}")

        if w_diff < 0:  # Need to crop width
            w_start = (-w_diff) // 2
            volume_tensor = volume_tensor[:, :, w_start : w_start + target_w]
            logger.info(f"Cropped W dimension from {current_w} to {target_w}")
        elif w_diff > 0:  # Need to pad width
            w_pad = (w_diff // 2, w_diff - w_diff // 2)
            volume_tensor = F.pad(
                volume_tensor, (*w_pad, 0, 0), mode="constant", value=pad_value
            )
            logger.info(f"Padded W dimension from {current_w} to {target_w}")

        # Convert back to numpy with original dtype
        return volume_tensor.numpy().astype(original_dtype)

    def _apply_slice_selection(self, volume: np.ndarray) -> np.ndarray:
        """Apply slice selection based on configuration"""
        # Get slice selection configuration
        slice_selection = self.config.processing.get("slice_selection", {})

        # If slice selection is not enabled, return original volume
        if not slice_selection.get("enabled", True):
            return volume

        # Get number of slices from config
        num_slices = slice_selection.get("slices")
        if not num_slices:
            # No slice count specified, return original volume
            logger.debug(f"No slice count specified for {self.anatomy} MR")
            return volume

        # Get current volume shape
        current_slices = volume.shape[0]

        # If volume has fewer slices than requested, return as is
        if current_slices <= num_slices:
            logger.debug(
                f"Volume has {current_slices} slices, which is <= {num_slices} requested. No selection applied."
            )
            return volume

        # Calculate the middle slice indices
        start_idx = (current_slices - num_slices) // 2
        end_idx = start_idx + num_slices

        # Select the middle slices
        selected_volume = volume[start_idx:end_idx]

        logger.info(
            f"Selected middle {num_slices} slices from {current_slices} total slices "
            f"(indices {start_idx}:{end_idx}) for {self.anatomy} MR"
        )

        return selected_volume
