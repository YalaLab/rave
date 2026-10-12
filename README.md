# Radiology Vision Engine (RAVE)

A high-performance medical image processing engine for converting DICOM and NIfTI files into ML-ready formats with intelligent windowing and compression.

### Installation with uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync
source .venv/bin/activate
```

#### Verify Installation

```python
import rve
print(rve.__version__)  # Should print 1.0.0

# Test basic functionality
from rve import load_sample, apply_windowing
windows = rve.get_available_windows('CT')
print(f"Available CT windows: {len(windows)}")  # Should show 11 windows
```

### Python API Usage

```python
import rve

# Load a medical image (supports .tar, .tar.gz, .tar.lz4)
image = rve.load_sample("path/to/image.tar.gz")

# Load NIfTI files (.nii, .nii.gz)
nifti_volume = rve.load_nifti("path/to/scan.nii.gz")

# Apply windowing
lung_view = rve.apply_windowing(image, 'lung', 'CT')
bone_view = rve.apply_windowing(image, 'bone', 'CT') 

# Apply multiple windows at once
multi = rve.apply_windowing(image, ['lung', 'bone', 'mediastinum'], 'CT')

# Apply all the windows at once

all = rve.apply_windowing(image, 'all', 'CT')

# See available windows
windows = rve.get_available_windows('CT')
print(windows)  # ['lung', 'mediastinum', 'abdomen', 'liver', 'bone', ...]
```

### Command Line Usage

```bash
# Process CT chest series (works with both DICOM and NIfTI paths in CSV)
vision-engine process \
    --config configs/ct_chest.yaml \
    --input-series-csv series_paths.csv \
    --output /output/dir \
    --workers 4

# Process with debug mode (limits to first N studies)
vision-engine process \
    --config configs/xray.yaml \
    --input-series-csv xray_series.csv \
    --output /output/dir \
    --debug --debug-limit 10

# Use the short alias
rve process --config configs/mammogram.yaml ...
```

### MRI (MR) Processing

The engine supports MRI processing (breast and prostate) from either DICOM series directories or NIfTI files listed in a CSV.

```bash
# Breast MRI (CSV contains DICOM directories and/or .nii/.nii.gz files)
vision-engine process \
    --config configs/mr_breast.yaml \
    --input-series-csv test_csvs/breast_mr_series.csv \
    --output test_output_mr \
    --workers 12

# Prostate MRI
vision-engine process \
    --config configs/mr_prostate.yaml \
    --input-series-csv test_csvs/prostate_mr_series.csv \
    --output test_output_prostate_mr \
    --workers 12

# Short alias also works
rve process --config configs/mr_breast.yaml --input-series-csv test_csvs/breast_mr_series.csv --output test_output_mr --workers 12
```

Notes:
- MR uses raw signal intensity (not HU). Intensities are mapped via min-max to 16-bit for video export.
- Resampling uses the modality config (e.g., `processing.resampling.target_spacing`) and preserves geometry.
- Cropping/padding and slice selection are controlled by the YAML (e.g., `processing.crop_pad.size`, `processing.slice_selection`).

Output and metadata:
- Non-mask MR series are exported as HEVC video with 10-bit grayscale and lossless settings appropriate for MR.
- Metadata (`metadata.json`) contains MR-specific processing details, including:
  - `original_spacing` and `target_spacing`
  - `original_shape` (Z,Y,X)
  - `resample_shape` (shape after resample, before crop/pad)
  - `final_shape` (after crop/pad)
  - `is_mask` (auto-detected segmentation mask flag)

Automatic mask handling:
- Segmentation masks are auto-detected (discrete, low unique values). When detected:
  - Nearest-neighbor resampling is used to preserve labels.
  - Export switches automatically to lossless LZ4 tarballs (no video), preserving exact integer values.
  - The output folder (or tarball) still includes a `metadata.json` with `processing_metadata.is_mask: true`.

## Features

- **Multi-modality Support**: CT, MR (MRI), X-ray, Mammography
- **Multiple Input Formats**: DICOM and NIfTI (.nii, .nii.gz)
- **Intelligent Windowing**: Anatomical and percentile-based windows
- **Multiple Export Formats**: 
  - LZ4 tarballs (fast compression for training)
  - HEVC video/HEVC image (10-bit exports for cine review)
  - Torch exports (ready-to-load tensors for downstream pipelines)
- **High Performance**: Parallel processing with configurable workers
- **Flexible Configuration**: YAML-based configuration system
- **Automatic Mask Handling**: Auto-detects segmentation masks, uses nearest-neighbor resampling, and exports losslessly via LZ4 (no HEVC) to preserve labels

## Input Format

Create a CSV file with paths to DICOM series directories or NIfTI files:

```csv
series_path
/path/to/patient1/series1/
/path/to/patient2/series2/
/path/to/patient3/series3/
/path/to/scan1.nii.gz
/path/to/scan2.nii.gz
```

## Configuration

Configuration files are in YAML format. Example for CT chest:

```yaml
# configs/ct_chest.yaml
modality: "CT"
anatomy: "chest"

processing:
  target_size: [512, 512]
  hu_min: -1024
  hu_max: 3071
  slice_thickness: 5.0
  max_slices: 192

exporter_config: "configs/exporters/video_hevc.yaml"
```

## Output orientation

Every 3D volume (CT and MR, DICOM or NIfTI) is reoriented to one patient pose
before export, so models see the same anatomy layout regardless of how the
source was stored. The default pose is `LPI`:

- column index increases toward the patient's **L**eft (radiological view: the
  patient's left is on the right of the image),
- row index increases toward **P**osterior (spine at the bottom of the frame),
- slice index increases toward **I**nferior (slice 0 is the most cranial; a
  chest CT starts near the clavicles and ends in the upper abdomen).

This is a pure axis permutation/flip with no resampling. A DICOM axial series
is usually already in this pose; a NIfTI stored in RAS is reversed on all three
axes and is fixed here, so no `flip` is needed in a training loader. The
applied transform is recorded in `metadata.json` under
`processing_metadata.orientation` (`input`, `output`, `reoriented`).

```yaml
processing:
  orientation: LPI   # any 3-letter code such as RAS; null disables reorientation
```

Set `orientation: null` only to reproduce caches built before this step
existed. The reorientation trusts the direction matrix: a NIfTI whose affine
contradicts its pixels, or a DICOM series without usable
`ImagePositionPatient`, cannot be detected here.

## Python API Reference

### Loading Data

```python
# Load any supported RVE archive (.tar, .tar.gz, .tar.lz4, .tar.hevc)
volume = rve.load_sample("path/to/accession.1.0.tar.lz4")

# Inspect metadata embedded in the archive
metadata = rve.get_export_info("path/to/accession.1.0.tar.lz4")
print(metadata["export_info"]["hu_mapping"])

# Load NIfTI files directly (bypasses tar exports)
nifti_volume = rve.load_nifti("path/to/scan.nii.gz")
```

### Windowing

```python
# Single window
windowed = rve.apply_windowing(image, 'lung', 'CT')

# Multiple windows
multi = rve.apply_windowing(image, ['lung', 'bone'], 'CT')

# All available windows
all_windows = rve.apply_windowing(image, 'all', 'CT')

# Min-max normalization
normalized = rve.apply_windowing(image, 'minmax', 'CT')

# Custom range normalization
custom = rve.apply_windowing(image, 'minmax', 'CT', 
                           min_value=-1000, max_value=1000)
```

### Processing & Export

```python
config = rve.Config.from_yaml("configs/ct_chest.yaml")
processor = rve.get_processor(config.modality, config)

export_format = config.exporter.get("compression", "lz4")
exporter = rve.get_exporter(export_format, config.exporter)
```

## Output Structure

```
output_dir/
├── mapping.csv                    # Maps series to output files
├── 12345.2.0.tar.lz4             # Processed series (LZ4)
├── 12346.1.0.tar                 # Processed series (HEVC video)
└── 12347.3.0.torch.tar           # Processed series (Torch tensors)
```

The `mapping.csv` contains:
```csv
series_path,accession,series_number,output_path,dicom_size_mb,processed_size_mb
/path/to/series1,12345,2,12345.2.0.tar.lz4,125.3,12.8
```

## Available Windows

### CT Windows
- `lung`: (center -600, width 1500) - Emphasizes lung parenchyma
- `mediastinum`: (center 50, width 400) - Soft tissue structures
- `abdomen`: (center 40, width 400) - Abdominal organs
- `liver`: (center 80, width 150) - Liver parenchyma
- `bone`: (center 400, width 1800) - Bone structures
- `brain`: (center 40, width 80) - Brain tissue
- `subdural`: (center 75, width 215) - Subdural spaces
- `stroke`: (center 40, width 40) - Acute stroke detection
- `temporal_bone`: (center 600, width 2800) - Temporal bone detail
- `soft_tissue`: (center 50, width 350) - General soft tissue

### X-ray Windows
Uses percentile-based windows for chest X-rays:
- `lung`: 2-50th percentile
- `mediastinum`: 30-80th percentile  
- `bone`: 70-98th percentile

## Advanced Usage

### Custom Configuration

```python
# Load config and modify
config = rve.Config.from_yaml("base_config.yaml")
config.processing['target_size'] = [256, 256]
config.parallel['workers'] = 8

# Use in processing
processor = rve.get_processor(config.modality, config)
```

### Error Handling

```python
try:
    image = rve.load_sample("path/to/image.tar")
except rve.VisionEngineError as e:
    print(f"Error: {e}")
except rve.ConfigurationError as e:
    print(f"Config error: {e}")
```

### Batch Processing

```python
import rve
from pathlib import Path

# Process multiple files
output_dir = Path("/output")
for tar_file in Path("/data").glob("*.tar.gz"):
    try:
        # Load and process
        image = rve.load_sample(tar_file)
        windowed = rve.apply_multiple_windows(image, 'CT')
        
        # Save results (implement your own logic)
        # ...
        
    except rve.VisionEngineError as e:
        print(f"Failed {tar_file}: {e}")
```

## Performance Tips

1. **Use appropriate number of workers**: 
   - LZ4: 6-8 workers (I/O bound)
   - HEVC video/image: 2-4 workers (encoder-bound)
   - Torch: 2 workers (tensor packaging)

2. **Choose the right format**:
   - Training: LZ4 (fast decompression and numpy compatibility)
   - Review/share: HEVC video or HEVC image exports (compact 10-bit streams)
   - Torch: Torch tarballs for direct dataloader consumption

3. **Debug mode**: Always test with `--debug --debug-limit 10` first

# Citation
If you use this code in your research, please cite the following paper:

```
@article{pillar0,
  title   = {Pillar-0: A New Frontier for Radiology Foundation Models},
  author  = {Agrawal, Kumar Krishna and Liu, Longchao and Lian, Long and Nercessian, Michael and Harguindeguy, Natalia and Wu, Yufu and Mikhael, Peter and Lin, Gigin and Sequist, Lecia V. and Fintelmann, Florian and Darrell, Trevor and Bai, Yutong and Chung, Maggie and Yala, Adam},
  year    = {2025}
}
```
