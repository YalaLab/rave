"""
Configuration management for vision-engine
"""

import os
import logging
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

from .exceptions import ConfigurationError


@dataclass
class Config:
    """Central configuration management - all values from YAML"""

    # Required fields
    modality: str
    anatomy: str

    # Processing settings
    processing: Dict[str, Any]

    # Exporter configuration
    exporter: Dict[str, Any] = field(default_factory=dict)
    exporter_config: Optional[str] = None

    # Performance settings
    parallel: Dict[str, Any] = field(default_factory=dict)

    # Logging configuration
    logging: Dict[str, Any] = field(default_factory=dict)

    # Runtime options (set via CLI)
    input_series_dirs_csv: Optional[str] = None
    path_column: str = "series_path"
    stop_on_error: bool = False
    overwrite: bool = False
    debug_mode: bool = False
    debug_max_series: int = 10
    dry_run: bool = False

    # Optional S3 input materialization
    # - off: do not download; S3 paths will likely fail in processors
    # - eager: download all S3 paths up front, then process
    # - lazy: download per-series inside workers (on demand)
    # - only: download all S3 paths then exit (no processing)
    s3_prefetch: str = "off"
    s3_cache_dir: Optional[str] = None
    s3_endpoint_url: Optional[str] = None
    s3_region: Optional[str] = None

    def __post_init__(self):
        """Post-initialization processing"""
        # Normalize modality
        self._normalize_modality()

        # Load external exporter config if specified
        if self.exporter_config:
            self._load_exporter_config()

        # Initialize logging
        self._setup_logging()

        # Validate configuration
        self._validate()

    def _normalize_modality(self):
        """Normalize modality names for DICOM compatibility"""
        modality_map = {
            "MAMMO": "MG",
            "MAMMOGRAPHY": "MG",
            "MG": "MG",
            "XRAY": "XR",
            "X-RAY": "XR",
            "XR": "XR",
            "CT": "CT",
            "MR": "MR",
            "MRI": "MR",
        }

        normalized = modality_map.get(self.modality.upper())
        if normalized:
            self.modality = normalized
        else:
            # Keep original if not in mapping
            self.modality = self.modality.upper()

    def _setup_logging(self):
        """Setup logging based on configuration"""
        log_config = self.logging or {}
        level = getattr(logging, log_config.get("level", "INFO").upper())

        handlers = [logging.StreamHandler()]
        if log_config.get("file"):
            handlers.append(logging.FileHandler(log_config["file"]))

        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=handlers,
        )

    def _validate(self):
        """Validate configuration"""
        # Check modality (after normalization)
        valid_modalities = ["CT", "XR", "MG", "MR"]
        if self.modality not in valid_modalities:
            raise ConfigurationError(f"Invalid modality: {self.modality}")

        # Check compression type
        valid_compressions = [
            "lz4",
            "video",
            "hevc",
            "torch",
            "hevc_image",
            "heic",
            "heif",
        ]
        if self.exporter["compression"].lower() not in valid_compressions:
            raise ConfigurationError(
                f"Invalid compression: {self.exporter['compression']}"
            )

    def _load_exporter_config(self):
        """Load exporter configuration from external file"""
        exporter_path = Path(self.exporter_config)

        # Check common locations if not absolute path
        if not exporter_path.is_absolute():
            # Look in configs/exporters directory relative to this file
            base_dir = Path(__file__).parent.parent.parent
            possible_paths = [
                base_dir / "configs" / "exporters" / f"{exporter_path}.yaml",
                base_dir / "configs" / "exporters" / exporter_path,
                base_dir / "configs" / f"{exporter_path}.yaml",
                base_dir / "configs" / exporter_path,
                Path(exporter_path),
            ]

            for path in possible_paths:
                if path.exists():
                    exporter_path = path
                    break
            else:
                raise ConfigurationError(
                    f"Exporter config not found: {self.exporter_config}"
                )

        # Load the exporter config
        try:
            with open(exporter_path, "r") as f:
                external_config = yaml.safe_load(f)

            # Override inline exporter config with external config
            if "exporter" in external_config:
                self.exporter.update(external_config["exporter"])
            else:
                # If the file is just the exporter config directly
                self.exporter.update(external_config)

            logging.info(f"Loaded exporter config from: {exporter_path}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load exporter config: {e}")

    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file"""
        try:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
            return cls(**data)
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {e}")

    @classmethod
    def from_cli_args(cls, cli_args: Dict[str, Any], config_file: str) -> "Config":
        """
        Create configuration from required YAML file with CLI overrides

        Args:
            cli_args: Dictionary of CLI arguments
            config_file: Required YAML config file path

        Returns:
            Config instance
        """
        # Load from YAML file (required)
        config = cls.from_yaml(config_file)

        # Apply CLI overrides for runtime options only
        runtime_fields = [
            "input_series_dirs_csv",
            "path_column",
            "stop_on_error",
            "debug_mode",
            "dry_run",
            "overwrite",
            "s3_prefetch",
            "s3_cache_dir",
            "s3_endpoint_url",
            "s3_region",
        ]

        for field_name in runtime_fields:
            if field_name in cli_args and cli_args[field_name] is not None:
                setattr(config, field_name, cli_args[field_name])

        # Handle debug_limit -> debug_max_series
        if cli_args.get("debug_limit") is not None:
            config.debug_max_series = cli_args["debug_limit"]

        # Workers override
        if cli_args.get("workers") is not None:
            config.parallel["workers"] = cli_args["workers"]

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "modality": self.modality,
            "anatomy": self.anatomy,
            "processing": self.processing,
            "exporter": self.exporter,
            "exporter_config": self.exporter_config,
            "parallel": self.parallel,
            "logging": self.logging,
            "input_series_dirs_csv": self.input_series_dirs_csv,
            "path_column": self.path_column,
            "stop_on_error": self.stop_on_error,
            "debug_mode": self.debug_mode,
            "debug_max_series": self.debug_max_series,
            "dry_run": self.dry_run,
            "overwrite": self.overwrite,
            "s3_prefetch": self.s3_prefetch,
            "s3_cache_dir": self.s3_cache_dir,
            "s3_endpoint_url": self.s3_endpoint_url,
            "s3_region": self.s3_region,
        }
