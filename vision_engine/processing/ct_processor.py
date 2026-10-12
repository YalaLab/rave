"""
CT processor for LZ4/numpy export.
Handles complete pipeline: DICOM → NIfTI → raw numpy slices
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


class CTProcessor(BaseProcessor):
    """CT processor that preserves raw Hounsfield Units"""

    def __init__(self, config):
        super().__init__(config)
        self.anatomy = config.anatomy
        self.orientation_target = orientation_target_from_config(config.processing)
        self._last_orientation: Optional[OrientationInfo] = None

    def process_series(self, series_info: SeriesInfo) -> ProcessedSeries:
        """Complete processing pipeline for one CT series"""
        logger.info(f"Processing CT series: {series_info}")

        try:
            # Step 1: DICOM → SimpleITK image with spacing info
            sitk_image, spacing = self._convert_dicom_to_nifti(series_info)

            # Step 2: Process volume (resampling, slice selection, resizing)
            processed_volume = self._process_volume(sitk_image, spacing)

            # Get processing configuration
            resampling_config = self.config.processing.get("resampling", {})
            target_spacing = resampling_config.get("target_spacing", None)
            crop_pad_config = self.config.processing.get("crop_pad", {})
            target_size = crop_pad_config.get("size", None)

            # Create processing metadata
            processing_metadata = {
                "modality": "CT",
                "anatomy": self.anatomy,
                "original_spacing": spacing.tolist(),
                "target_spacing": target_spacing,
                "original_shape": [
                    int(d) for d in sitk_image.GetSize()[::-1]
                ],  # Convert to Z,Y,X order
                "final_shape": processed_volume.shape,
                "crop_pad_target": target_size,
                "slice_count": processed_volume.shape[0],
                "value_range": "raw_hounsfield_units",
                "orientation": self._last_orientation.to_dict()
                if self._last_orientation is not None
                else None,
                "series_uid": series_info.series_uid,
                "slice_info": getattr(
                    series_info, "slice_info", []
                ),  # Will be populated if available
            }

            return ProcessedSeries(
                series_info=series_info,
                numpy_volume=processed_volume,
                processing_metadata=processing_metadata,
            )

        except Exception as e:
            raise ProcessingError(
                f"Failed to process CT series {series_info.accession}.{series_info.series_number}: {e}"
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

        # Check pixel type - we'll convert to int16 after processing
        pixel_type = image.GetPixelIDTypeAsString()
        logger.debug(f"Pixel type: {pixel_type}")

        return image, spacing

    def _sort_dicom_files(self, dicom_names: List[str]) -> List[str]:
        """Sort DICOM files by instance number and position (from reference code)"""
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
        self, sitk_image: sitk.Image, spacing: np.ndarray
    ) -> np.ndarray:
        """Process volume with 3D resampling and resizing"""
        # Step 1: Apply 3D resampling to target spacing if configured
        resampling_config = self.config.processing.get("resampling", {})
        target_spacing = resampling_config.get("target_spacing")

        if target_spacing:
            sitk_image = self._resample_volume(sitk_image, target_spacing)

        # Put every input (DICOM or NIfTI) in the same patient pose before the
        # array is extracted; see orientation.py for why this matters.
        sitk_image, self._last_orientation = orient_to_canonical(
            sitk_image, self.orientation_target
        )

        # Convert to numpy array after resampling
        volume = sitk.GetArrayFromImage(sitk_image)  # Shape: (Z, Y, X)

        # Ensure int16 dtype for CT (Hounsfield Units)
        if volume.min() >= -32768 and volume.max() <= 32767:
            volume = volume.astype(np.int16)
        else:
            logger.warning(
                f"CT values outside int16 range: [{volume.min()}, {volume.max()}]"
            )

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
        self, image: sitk.Image, target_spacing: List[float]
    ) -> sitk.Image:
        """Resample volume to target spacing using SimpleITK"""
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

        # Use linear interpolation for CT data
        resampler.SetInterpolator(sitk.sitkLinear)

        # Execute resampling
        resampled = resampler.Execute(image)

        return resampled

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
                    f"(indices {start_idx}:{end_idx}) for {self.anatomy} CT"
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
            logger.debug(f"No slice count specified for {self.anatomy} CT")
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
            f"(indices {start_idx}:{end_idx}) for {self.anatomy} CT"
        )

        return selected_volume
