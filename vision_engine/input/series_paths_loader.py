"""
CSV loader for direct series paths (no filtering needed).
"""

import pandas as pd
import logging
import os
from typing import List
from pathlib import Path
from tqdm import tqdm

from ..core.data_structures import SeriesInfo
from ..core.exceptions import InputError

logger = logging.getLogger(__name__)


class SeriesPathsLoader:
    """Load series directly from a CSV containing series paths"""

    def __init__(self, config):
        self.config = config

    def get_series_list(self, csv_path: str) -> List[SeriesInfo]:
        """
        Load series information from a CSV containing direct paths to series directories.

        Expected CSV format:
        - path column (configurable, default 'series_path'): Path to DICOM series directory
        - accession: Accession number (optional, extracted from path if not provided)
        - series_number: Series number (optional, defaults to 1)
        - series_description: Description (optional)

        Args:
            csv_path: Path to CSV file

        Returns:
            List of SeriesInfo objects
        """
        logger.info(f"Loading series paths from CSV: {csv_path}")

        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise InputError(f"CSV file not found: {csv_path}")

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise InputError(f"Failed to read CSV file: {e}")

        if df.empty:
            raise InputError("CSV file is empty")

        # Check required column (configurable)
        path_column = getattr(self.config, "path_column", "series_path")
        if path_column not in df.columns:
            raise InputError(f"CSV must contain '{path_column}' column")

        logger.info(f"Loaded CSV with {len(df)} series paths")

        # Convert to SeriesInfo objects
        series_list = []
        path_column = getattr(self.config, "path_column", "series_path")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc='Gathering series info'):
            try:
                series_path = str(row[path_column])

                # Validate path exists
                if not os.path.exists(series_path):
                    logger.warning(f"Series path not found, skipping: {series_path}")
                    continue

                # Extract series UID from path (last component)
                series_uid = os.path.basename(series_path.rstrip("/"))

                # Get accession from CSV or extract from path
                if "accession" in row and pd.notna(row["accession"]):
                    accession = str(row["accession"])
                else:
                    # Try to extract from path (assuming structure like .../accession/...)
                    path_parts = series_path.split("/")
                    accession = None
                    for part in reversed(path_parts):
                        if part and part[0].isdigit() and len(part) >= 6:
                            accession = part
                            break
                    if not accession:
                        accession = f"unknown_{idx}"

                # Get other fields from CSV or use defaults
                series_number = (
                    int(row.get("series_number", 1))
                    if "series_number" in row and pd.notna(row["series_number"])
                    else 1
                )
                series_description = (
                    str(row.get("series_description", ""))
                    if "series_description" in row
                    and pd.notna(row["series_description"])
                    else ""
                )

                # Check if this is a NIfTI file or directory with medical images
                path_obj = Path(series_path)

                if path_obj.is_file() and path_obj.suffix in [".nii", ".gz"]:
                    # This is a NIfTI file
                    slice_count = 1  # NIfTI files are single volumes
                    logger.debug(f"Detected NIfTI file: {series_path}")
                elif path_obj.is_dir():
                    # Check for NIfTI files in directory
                    nifti_files = list(path_obj.glob("*.nii")) + list(
                        path_obj.glob("*.nii.gz")
                    )
                    if nifti_files:
                        slice_count = len(nifti_files)
                        logger.debug(
                            f"Found {slice_count} NIfTI files in {series_path}"
                        )
                    else:
                        # Count DICOM files in directory
                        dicom_files = [
                            f
                            for f in os.listdir(series_path)
                            if f.endswith((".dcm", ".DCM")) or f.isdigit()
                        ]
                        slice_count = len(dicom_files)

                        if slice_count == 0:
                            logger.warning(
                                f"No medical image files found in {series_path}, skipping"
                            )
                            continue
                else:
                    logger.warning(
                        f"Unknown file type or missing path: {series_path}, skipping"
                    )
                    continue

                # Extract study path (parent of series directory)
                study_path = os.path.dirname(series_path)

                # Create SeriesInfo object
                series = SeriesInfo(
                    accession=accession,
                    series_uid=series_uid,
                    study_path=study_path,
                    series_number=series_number,
                    series_description=series_description,
                    slice_count=slice_count,
                    modality=self.config.modality,
                    anatomy=self.config.anatomy,
                )

                # Override get_dicom_path to return the direct path
                series._direct_path = series_path

                series_list.append(series)

            except Exception as e:
                logger.warning(f"Error processing row {idx}: {e}")
                continue

        if not series_list:
            raise InputError("No valid series found in CSV")

        logger.info(f"Successfully loaded {len(series_list)} series from direct paths")
        return series_list
