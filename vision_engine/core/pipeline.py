"""
Main pipeline orchestrator for vision-engine.
Simplified architecture: Input → Process → Export
"""

import logging
import os

# Set environment variables for thread safety
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .thread_parallel import thread_parallel_map
import pandas as pd

from ..input.series_paths_loader import SeriesPathsLoader
from ..utils.s3_utils import is_s3_url, materialize_s3_to_local
from ..processing.ct_processor import CTProcessor
from ..processing.xray_processor import XRayProcessor
from ..processing.mammogram_processor import MammogramProcessor
from ..processing.mr_processor import MRProcessor
from ..lz4_exporter import LZ4Exporter
from ..video_exporter import VideoExporter
from ..hevc_image_exporter import HEVCImageExporter
from ..torch_exporter import TorchExporter
from .config import Config
from .exceptions import VisionEngineError
from .data_structures import SeriesInfo, ProcessedSeries, ExportResult

logger = logging.getLogger(__name__)


def get_processor_class(modality: str):
    """Get processor class for modality"""
    modality = modality.upper()
    if modality == "CT":
        return CTProcessor
    elif modality in ["XR", "XRAY"]:
        return XRayProcessor
    elif modality in ["MG", "MAMMO", "MAMMOGRAPHY"]:
        return MammogramProcessor
    elif modality == "MR":
        return MRProcessor
    else:
        raise VisionEngineError(f"Unsupported modality: {modality}")


def get_exporter_class(compression: str):
    """Get exporter class for compression type"""
    compression = compression.lower()
    if compression in ["video", "hevc"]:
        return VideoExporter
    elif compression in ["hevc_image", "heic", "heif"]:
        return HEVCImageExporter
    elif compression == "torch":
        return TorchExporter
    else:
        # Default to LZ4 for speed
        return LZ4Exporter


def get_file_extension(compression: str) -> str:
    """Get file extension for compression type"""
    compression = compression.lower()
    if compression in ["video", "hevc"]:
        return ".tar"
    elif compression in ["hevc_image", "heic", "heif"]:
        # hevc image exporter writes a folder named like *.0.tar containing files
        return ""
    elif compression == "torch":
        return ".torch.tar"
    else:
        return ".tar.lz4"


class VisionPipeline:
    """Main pipeline orchestrator - simplified for JPEG tarball output"""

    def __init__(self, config: Config):
        self.config = config

        self.input_handler = SeriesPathsLoader(self.config)
        self.processor = get_processor_class(config.modality)(config)
        self.exporter = get_exporter_class(config.exporter.get("compression", "lz4"))(
            config
        )

        logger.info(f"Initialized pipeline for {config.modality} processing")

    def process(self, input_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Execute: Input → Process → Export as JPEG tarball

        Args:
            input_path: Path to CSV file or DICOM directory
            output_dir: Output directory for tarballs

        Returns:
            Processing summary with statistics
        """
        logger.info(f"Starting pipeline: {input_path} → {output_dir}")

        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Step 1: Get series list from CSV
        logger.info("Step 1: Loading series list from CSV...")
        series_list = self.input_handler.get_series_list(input_path)
        logger.info(f"Loaded {len(series_list)} series")

        # Optional S3 prefetch/materialization
        prefetch_mode = (getattr(self.config, "s3_prefetch", "off") or "off").lower()
        if prefetch_mode not in {"off", "eager", "lazy", "only"}:
            raise VisionEngineError(f"Invalid s3_prefetch mode: {prefetch_mode}")

        if prefetch_mode in {"eager", "only"}:
            logger.info(f"S3 prefetch mode: {prefetch_mode} (materializing inputs)")
            for s in series_list:
                try:
                    src = s.get_dicom_path()
                except Exception:
                    # If something odd happens, skip prefetch for this series and let processing fail loudly.
                    continue
                if is_s3_url(src):
                    local_path, _kind = materialize_s3_to_local(
                        src,
                        cache_dir=getattr(self.config, "s3_cache_dir", None),
                        endpoint_url=getattr(self.config, "s3_endpoint_url", None),
                        region_name=getattr(self.config, "s3_region", None),
                    )
                    # Preserve original source path for mapping/debugging
                    s._original_direct_path = src  # type: ignore[attr-defined]
                    s._direct_path = local_path  # type: ignore[attr-defined]

            if prefetch_mode == "only":
                logger.info("S3 prefetch-only mode: completed downloads; exiting without processing.")
                return {
                    "total_series": len(series_list),
                    "processed_successfully": 0,
                    "failed_series": 0,
                    "output_directory": output_dir,
                    "export_results": [],
                    "failed_details": [],
                    "mapping_file": None,
                }

        # Apply debug mode limiting
        if self.config.debug_mode:
            # Limit by unique accessions (studies) not individual series
            unique_accessions = list(set(s.accession for s in series_list))
            limited_accessions = unique_accessions[: self.config.debug_max_series]
            series_list = [s for s in series_list if s.accession in limited_accessions]
            logger.info(
                f"Debug mode: Limited to {len(limited_accessions)} studies ({len(series_list)} series)"
            )

        logger.info(f"Final series count: {len(series_list)} series to process")

        if not series_list:
            raise VisionEngineError("No series found to process after filtering")

        # Step 2: Process each series (DICOM → NIfTI → JPEG → tarball) - FULLY PARALLELIZED
        logger.info("Step 2: Processing series to tarballs...")
        export_results, failed_series = self._process_series_to_tarballs_parallel(
            series_list, output_dir
        )

        # Create processing summary
        successful_exports = [r for r in export_results if r.success]
        summary = {
            "total_series": len(series_list),
            "processed_successfully": len(successful_exports),
            "failed_series": len(failed_series),
            "output_directory": output_dir,
            "export_results": export_results,
            "failed_details": failed_series,
        }

        # Step 3: Generate path mapping CSV
        logger.info("Step 3: Generating path mapping CSV...")
        mapping_path = self._generate_path_mapping(
            series_list, export_results, output_dir
        )
        summary["mapping_file"] = mapping_path

        logger.info("Pipeline completed successfully")
        return summary

    def _process_series_to_tarballs_parallel(
        self, series_list: List[SeriesInfo], output_dir: str
    ) -> Tuple[List[ExportResult], List[Tuple[SeriesInfo, str]]]:
        """Process series to tarballs in parallel"""

        # Get number of workers from config
        max_workers = self.config.parallel.get("workers", 4)
        logger.info(
            f"Processing {len(series_list)} series to tarballs with {max_workers} parallel workers..."
        )

        start_time = time.time()

        # Create tasks for parallel processing - each task contains all info needed
        tasks = [(series, output_dir, self.config.to_dict()) for series in series_list]

        # Use ThreadPoolExecutor for I/O-bound tasks
        try:
            # Scale up based on available cores for I/O work
            thread_workers = min(max_workers * 2, 32) if max_workers > 1 else 1
            logger.info(f"Using {thread_workers} threads for I/O-bound processing")
            results = thread_parallel_map(
                _process_series_to_tarball,
                tasks,
                max_workers=thread_workers,
                show_progress=True,
                desc="Processing series",
            )
        except Exception as e:
            if self.config.stop_on_error:
                raise
            logger.error(f"Parallel processing failed: {e}")
            results = []

        # Separate successful and failed results
        export_results = []
        failed_series = []

        for i, result in enumerate(results):
            if result is None:
                # Task failed in parallel processing
                logger.error(f"Task {i} returned None (failed in parallel processing)")
                failed_series.append((series_list[i], "Parallel processing failed"))
            elif isinstance(result, ExportResult):
                export_results.append(result)
            else:
                # Result is (series, error_message) tuple
                failed_series.append(result)

        processing_time = time.time() - start_time
        successful_count = len([r for r in export_results if r.success])
        logger.info(
            f"Parallel processing completed: {successful_count} successful, "
            f"{len(failed_series)} failed in {processing_time:.1f}s"
        )

        return export_results, failed_series

    def _generate_path_mapping(
        self,
        series_list: List[SeriesInfo],
        export_results: List[ExportResult],
        output_dir: str,
    ) -> str:
        """
        Generate CSV mapping from source DICOM paths to output tarball paths

        Args:
            series_list: Original list of series to process
            export_results: Results from processing/export
            output_dir: Output directory where tarballs were saved

        Returns:
            Path to the generated mapping.csv file
        """
        mapping_data = []

        # Convert output_dir to absolute path
        output_dir_abs = os.path.abspath(output_dir)

        # Create a dict of successful exports for quick lookup
        success_dict = {
            (r.series_info.accession, r.series_info.series_number): r
            for r in export_results
            if r.success
        }

        # Get compression format and extension (respect folder vs tar settings)
        compression = self.config.exporter.get("compression", "lz4")
        extension = get_file_extension(compression)
        # Special-case video exporter: if configured to not archive, outputs a folder (no extension)
        try:
            if compression.lower() in ["video", "hevc"]:
                video_cfg = (
                    self.config.exporter.get("video", {})
                    if isinstance(self.config.exporter, dict)
                    else {}
                )
                if video_cfg.get("archive", True) is False:
                    extension = ""
        except Exception:
            pass

        # Build mapping for successfully processed series only
        for series in series_list:
            # Check if this series was successfully processed
            export_key = (series.accession, series.series_number)
            export_result = success_dict.get(export_key)

            # Only include successfully processed series
            if export_result:
                # Get source path
                source_path = getattr(series, "_original_direct_path", None) or series.get_dicom_path()

                # Prefer the actual output path reported by exporter (handles masks and unarchived video)
                actual_output_path = getattr(export_result, "output_path", None)
                if actual_output_path:
                    tarball_path = actual_output_path
                else:
                    # Fallback to expected naming
                    tarball_filename = (
                        f"{series.accession}.{series.series_number}.0{extension}"
                    )
                    tarball_path = os.path.join(output_dir_abs, tarball_filename)

                mapping_entry = {
                    "source_path": source_path,
                    "output_path": tarball_path,
                }

                mapping_data.append(mapping_entry)

        # Create DataFrame and save to CSV
        mapping_df = pd.DataFrame(mapping_data)
        mapping_path = os.path.join(output_dir, "mapping.csv")

        try:
            mapping_df.to_csv(mapping_path, index=False)
            logger.info(
                f"Generated simplified mapping CSV with {len(mapping_df)} entries at {mapping_path}"
            )

        except Exception as e:
            logger.error(f"Failed to generate mapping CSV: {e}")
            mapping_path = None

        return mapping_path


def _process_series_to_tarball(task) -> ExportResult:
    """
    Complete pipeline: DICOM → NIfTI → JPEG → tarball

    Args:
        task: Tuple of (series_info, output_dir, config_dict)

    Returns:
        ExportResult with success/failure status
    """
    series_info, output_dir, config_dict = task

    try:
        # Recreate config and components in the worker process
        from .config import Config
        from pathlib import Path

        config = Config(**config_dict)

        # Optional S3 materialization in worker (lazy mode)
        prefetch_mode = (getattr(config, "s3_prefetch", "off") or "off").lower()
        if prefetch_mode == "lazy":
            try:
                src = series_info.get_dicom_path()
            except Exception:
                src = None
            if src and is_s3_url(src):
                local_path, _kind = materialize_s3_to_local(
                    src,
                    cache_dir=getattr(config, "s3_cache_dir", None),
                    endpoint_url=getattr(config, "s3_endpoint_url", None),
                    region_name=getattr(config, "s3_region", None),
                )
                series_info._original_direct_path = src  # type: ignore[attr-defined]
                series_info._direct_path = local_path  # type: ignore[attr-defined]

        # Get the appropriate processor and exporter
        processor_class = get_processor_class(config.modality)
        processor = processor_class(config)

        exporter_class = get_exporter_class(config.exporter.get("compression", "lz4"))
        exporter = exporter_class(config)

        # Step 1: Process series (DICOM → NIfTI → JPEG)
        processed_series = processor.process_series(series_info)

        # Step 2: Export as tarball
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export single series to tarball
        # If processor flagged series as mask, force LZ4 exporter to preserve labels
        try:
            is_mask = bool(processed_series.processing_metadata.get("is_mask", False))
        except Exception:
            is_mask = False
        if is_mask and config.exporter.get("compression", "lz4") != "lz4":
            from ..lz4_exporter import LZ4Exporter

            exporter = LZ4Exporter(config)

        result = exporter.export(processed_series, output_dir)

        return result

    except Exception as e:
        # Return failed result instead of raising
        import traceback

        error_msg = f"Error processing {series_info.accession}.{series_info.series_number}: {str(e)}"
        logger.error(error_msg)
        logger.debug(f"Full traceback:\n{traceback.format_exc()}")

        from .data_structures import ExportResult

        return ExportResult(
            series_info=series_info,
            output_path="",
            file_size_mb=0,
            slice_count=0,
            success=False,
            error_message=str(e),
            dicom_size_mb=0.0,
        )
