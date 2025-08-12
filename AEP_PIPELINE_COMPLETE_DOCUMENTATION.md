# AEP 3.0 Air Emissions Prediction Pipeline - Complete Documentation

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Pipeline Components](#pipeline-components)
4. [Data Flow Architecture](#data-flow-architecture)
5. [File Descriptions](#file-descriptions)
6. [Execution Methods](#execution-methods)
7. [Configuration & Setup](#configuration--setup)
8. [Troubleshooting Guide](#troubleshooting-guide)
9. [Output Files](#output-files)
10. [Performance & Optimization](#performance--optimization)

---

## Overview

The **AEP 3.0 Air Emissions Prediction Pipeline** is a comprehensive data processing system that combines satellite imagery data (INSAT) with ground-based air quality measurements (CPCB) to create a unified dataset for PM2.5 prediction modeling.

### Key Objectives:
- Process raw satellite imagery for multiple Indian states
- Extract meaningful features from satellite data
- Clean and validate CPCB air quality station data
- Merge satellite features with ground truth measurements
- Generate a unified dataset for machine learning models

### Supported States:
- **Delhi** (4 stations)
- **Haryana** (5 stations) 
- **Karnataka** (5 stations)
- **Maharashtra** (5 stations)

---

## Project Structure

```
APE_07/
├── data/                           # Main data directory
│   ├── raw/                        # Raw satellite imagery (INSAT)
│   │   ├── Delhi/
│   │   ├── Haryana/
│   │   ├── Karnataka/
│   │   └── Maharashtra/
│   ├── cpcb/                       # CPCB station data
│   │   ├── Delhi/
│   │   │   ├── DL009.csv
│   │   │   ├── DL011.csv
│   │   │   ├── delhi_coordinates.csv
│   │   │   └── ...
│   │   ├── Haryana/
│   │   ├── Karnataka/
│   │   └── Maharashtra/
│   ├── cropped_data/               # Processed satellite images
│   ├── image_features/             # Extracted image features
│   ├── processed_data/             # Merged datasets per station
│   └── unified_dataset.csv         # Final output
├── logs/                           # Pipeline execution logs
├── modules/                        # Pipeline modules (legacy)
├── robust_aep_pipeline_final.py    # MAIN PIPELINE (Recommended)
├── aep_31_complete_pipeline.py     # Alternative pipeline
├── test_cpcb_loading.py            # CPCB data testing utility
├── requirements.txt                # Python dependencies
└── AEP_PIPELINE_COMPLETE_DOCUMENTATION.md  # This file
```

---

## Pipeline Components

### 1. **Data Input Components**
- **Raw Satellite Data**: INSAT imagery files (TIF format)
- **CPCB Station Data**: Air quality measurements (CSV format)
- **Station Metadata**: Coordinates and location information

### 2. **Processing Components**
- **Image Cropping**: Extract station-specific regions from satellite images
- **Feature Extraction**: Calculate statistical features from cropped images
- **Data Cleaning**: Validate and clean CPCB measurements
- **Temporal Alignment**: Match satellite data with ground measurements
- **Data Merging**: Combine features with air quality data

### 3. **Output Components**
- **Cropped Images**: Station-specific satellite image tiles
- **Feature Files**: Statistical summaries of image data
- **Merged Datasets**: Combined satellite + CPCB data per station
- **Unified Dataset**: Final consolidated dataset for modeling

---

## Data Flow Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   Raw INSAT     │    │   CPCB Station  │
│   Satellite     │    │      Data       │
│    Images       │    │   (PM2.5, etc)  │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│  Image Cropping │    │  Data Cleaning  │
│  (Station ROI)  │    │  & Validation   │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          ▼                      │
┌─────────────────┐              │
│    Feature      │              │
│   Extraction    │              │
└─────────┬───────┘              │
          │                      │
          ▼                      ▼
┌─────────────────────────────────────┐
│         Data Merging                │
│    (Temporal Alignment)             │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│       Unified Dataset               │
│      (unified_dataset.csv)          │
└─────────────────────────────────────┘
```

---

## File Descriptions

### **Main Pipeline Files**

#### 1. `robust_aep_pipeline_final.py` ⭐ **RECOMMENDED**
- **Purpose**: Main robust pipeline with comprehensive error handling
- **Features**: 
  - Complete error recovery and validation
  - Skip logic for already processed data
  - Detailed logging with UTF-8 encoding
  - Cross-platform compatibility
  - Memory efficient processing
- **When to Use**: Primary pipeline for production runs
- **Execution**: `python robust_aep_pipeline_final.py`

#### 2. `aep_31_complete_pipeline.py`
- **Purpose**: Alternative complete pipeline (legacy)
- **Features**:
  - Full pipeline functionality
  - Basic error handling
  - May have Unicode encoding issues on Windows
- **When to Use**: Backup option or for comparison
- **Execution**: `python aep_31_complete_pipeline.py`

### **Utility Files**

#### 3. `test_cpcb_loading.py`
- **Purpose**: Test and validate CPCB data loading functionality
- **Features**:
  - Validates data file existence
  - Tests column detection (PM2.5, timestamps)
  - Checks data cleaning processes
  - Reports data quality statistics
- **When to Use**: Debugging CPCB data issues
- **Execution**: `python test_cpcb_loading.py`

#### 4. `create_unified_dataset.py`
- **Purpose**: Standalone dataset creation from processed files
- **Features**:
  - Combines all processed station files
  - Creates final unified dataset
  - Handles missing data gracefully
- **When to Use**: When you have processed data but need to recreate final dataset
- **Execution**: `python create_unified_dataset.py`

### **Configuration Files**

#### 5. `requirements.txt`
- **Purpose**: Python package dependencies
- **Contents**:
  ```
  pandas>=1.3.0
  numpy>=1.21.0
  rasterio>=1.2.0
  pytz>=2021.1
  Pillow>=8.3.0
  ```

### **Legacy Files** (modules/ directory)
- `metadata_parser.py`: Station metadata handling
- `satellite_cropper.py`: Image cropping functionality  
- `data_aligner.py`: Temporal alignment utilities
- `dataset_merger.py`: Data merging operations
- `utils.py`: Common utilities

---

## Execution Methods

### **Method 1: Recommended Robust Pipeline**

```bash
# Navigate to project directory
cd "C:\Users\Lalit Hire\OneDrive\Desktop\APE_07"

# Run the robust pipeline
python robust_aep_pipeline_final.py
```

**Expected Output:**
```
[INIT] Robust AEP Pipeline initialized successfully
================================================================================
[START] ROBUST AEP PIPELINE EXECUTION
================================================================================
[SUCCESS] Found 19 stations across 4 states
[STATE] Processing Delhi (4 stations)
[STATION] Processing station: DL009 (Pusa, Delhi)
[CROPPED] 13620 images for DL009
[CLEANED] 5511 records for DL009
[SUCCESS] DL009 processed
...
[DATASET] Created unified dataset: 45,230 records from 15 stations
================================================================================
[COMPLETE] ROBUST AEP PIPELINE EXECUTION COMPLETE
================================================================================
```

### **Method 2: Alternative Pipeline**

```bash
# Run alternative pipeline
python aep_31_complete_pipeline.py
```

### **Method 3: Step-by-Step Execution**

```bash
# 1. Test CPCB data loading first
python test_cpcb_loading.py

# 2. Run main pipeline
python robust_aep_pipeline_final.py

# 3. If needed, recreate unified dataset only
python create_unified_dataset.py
```

### **Method 4: Custom Execution with Parameters**

```python
# Custom execution in Python
from robust_aep_pipeline_final import RobustAEPPipeline

# Initialize with custom data directory
pipeline = RobustAEPPipeline(data_root="custom_data_path")

# Run pipeline
results = pipeline.run_complete_pipeline()

# Check results
if results['success']:
    print(f"Processed {results['stations_processed']} stations")
    print(f"Output: {results['unified_dataset']['output_file']}")
else:
    print(f"Pipeline failed: {results['error']}")
```

---

## Configuration & Setup

### **Prerequisites**

1. **Python Environment**: Python 3.7+
2. **Required Packages**: Install from requirements.txt
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Structure**: Ensure proper data directory structure
4. **Permissions**: Write access to output directories

### **Data Requirements**

#### **CPCB Data Format**
Each station CSV must contain:
- **PM2.5 Column**: `PM2.5 (ug/m3)` or similar
- **Timestamp Column**: `From Date` or `Date` + `Time`
- **Valid Data Range**: PM2.5 values 0-1000 μg/m³

#### **Satellite Data Format**
- **File Format**: GeoTIFF (.tif, .tiff)
- **Projection**: Geographic coordinate system
- **Organization**: By state and year directories

#### **Coordinate Files**
Each state needs `{state}_coordinates.csv` with:
- `file_name`: Station CSV filename
- `station_location`: Station name
- `latitude`: Station latitude (-90 to 90)
- `longitude`: Station longitude (-180 to 180)

### **Environment Variables** (Optional)
```bash
# Set custom data directory
export AEP_DATA_ROOT="/path/to/data"

# Set log level
export AEP_LOG_LEVEL="DEBUG"
```

---

## Troubleshooting Guide

### **Common Issues & Solutions**

#### **1. Unicode Encoding Errors**
**Problem**: `UnicodeEncodeError: 'charmap' codec can't encode character`
**Solution**: Use `robust_aep_pipeline_final.py` which handles UTF-8 encoding properly

#### **2. CPCB Data Not Found**
**Problem**: `[WARNING] No CPCB data for {station_id}`
**Solution**: 
- Run `python test_cpcb_loading.py` to diagnose
- Check file naming: `{station_id}.csv`
- Verify column names: `PM2.5 (ug/m3)`, `From Date`

#### **3. Image Cropping Failures**
**Problem**: `[WARNING] Failed to crop {image_file}`
**Solutions**:
- Check satellite image file integrity
- Verify coordinate bounds are valid
- Ensure sufficient disk space

#### **4. Memory Issues**
**Problem**: Pipeline crashes with memory errors
**Solutions**:
- Process fewer stations at once
- Clear intermediate files
- Increase system memory/swap

#### **5. Permission Errors**
**Problem**: `PermissionError: Cannot create directory`
**Solutions**:
- Run with administrator privileges
- Check directory write permissions
- Ensure antivirus isn't blocking file operations

### **Debug Commands**

```bash
# Test individual components
python test_cpcb_loading.py                    # Test CPCB data
python -c "from robust_aep_pipeline_final import RobustAEPPipeline; p=RobustAEPPipeline(); print(p._load_station_metadata())"  # Test metadata

# Check data structure
ls -la data/cpcb/Delhi/                        # Linux/Mac
dir data\cpcb\Delhi\                           # Windows

# Monitor pipeline execution
tail -f logs/robust_aep_*.log                  # Linux/Mac
Get-Content logs/robust_aep_*.log -Wait        # Windows PowerShell
```

---

## Output Files

### **Primary Output**

#### `unified_dataset.csv`
**Location**: `data/unified_dataset.csv`
**Description**: Final consolidated dataset ready for ML modeling

**Columns**:
- `state`: State name (Delhi, Haryana, etc.)
- `station_id`: Station identifier (DL009, etc.)
- `station_location`: Human-readable station name
- `latitude`: Station latitude coordinate
- `longitude`: Station longitude coordinate
- `timestamp_utc`: UTC timestamp of measurement
- `PM2.5`: Ground truth PM2.5 concentration (μg/m³)
- `img_mean`: Mean pixel value from satellite image
- `img_std`: Standard deviation of pixel values
- `img_min`: Minimum pixel value
- `img_max`: Maximum pixel value
- `img_median`: Median pixel value

**Sample Data**:
```csv
state,station_id,station_location,latitude,longitude,timestamp_utc,PM2.5,img_mean,img_std,img_min,img_max,img_median
Delhi,DL009,Pusa Delhi,28.6419,77.1419,2021-01-01 02:30:00+00:00,311.95,245.67,45.23,180,320,240
Delhi,DL009,Pusa Delhi,28.6419,77.1419,2021-01-01 03:30:00+00:00,441.33,267.89,52.14,190,340,265
```

### **Intermediate Outputs**

#### `data/cropped_data/{state}/{station_id}/{year}/`
- **Content**: Cropped satellite images for each station
- **Format**: GeoTIFF files with `_cropped.tif` suffix
- **Purpose**: Station-specific satellite image tiles

#### `data/image_features/{state}_{station_id}_features.csv`
- **Content**: Statistical features extracted from cropped images
- **Format**: CSV with image statistics per timestamp
- **Purpose**: Intermediate feature representation

#### `data/processed_data/{state}_{station_id}_merged.csv`
- **Content**: Merged satellite features + CPCB data per station
- **Format**: CSV with all columns for one station
- **Purpose**: Station-level processed dataset

### **Log Files**

#### `logs/robust_aep_{timestamp}.log`
- **Content**: Detailed execution logs with timestamps
- **Format**: Structured log entries with levels (INFO, WARNING, ERROR)
- **Purpose**: Debugging and monitoring pipeline execution

**Sample Log Entry**:
```
2024-01-15 14:23:45,123 - INFO - _process_station_complete:156 - [SUCCESS] DL009 processed
2024-01-15 14:23:45,124 - WARNING - _load_clean_cpcb_data:234 - [WARNING] No CPCB data for DL010
```

---

## Performance & Optimization

### **Performance Metrics**

**Typical Execution Times** (on modern hardware):
- **Image Cropping**: ~40-100 images/second
- **Feature Extraction**: ~200 images/second  
- **CPCB Data Processing**: ~1000 records/second
- **Total Pipeline**: ~15-30 minutes for all states

**Resource Usage**:
- **Memory**: 2-4 GB peak usage
- **Storage**: ~10-20 GB for intermediate files
- **CPU**: Multi-core utilization during image processing

### **Optimization Strategies**

#### **1. Skip Already Processed Data**
```python
# Automatically implemented in robust pipeline
if output_file.exists():
    return str(output_file)  # Skip processing
```

#### **2. Parallel Processing** (Future Enhancement)
```python
# Example for future implementation
from multiprocessing import Pool

def process_station_parallel(args):
    state, station_id, station_info = args
    return pipeline._process_station_complete(state, station_id, station_info)

# Process multiple stations in parallel
with Pool(processes=4) as pool:
    results = pool.map(process_station_parallel, station_args)
```

#### **3. Memory Management**
- Process data in chunks
- Clear intermediate variables
- Use generators for large datasets

#### **4. Storage Optimization**
- Compress intermediate files
- Clean up temporary data
- Use efficient file formats

### **Monitoring & Profiling**

```python
# Add timing to pipeline execution
import time
start_time = time.time()

# Your pipeline code here
results = pipeline.run_complete_pipeline()

execution_time = time.time() - start_time
print(f"Pipeline completed in {execution_time:.2f} seconds")
```

---

## Advanced Usage

### **Custom Data Processing**

```python
from robust_aep_pipeline_final import RobustAEPPipeline

# Initialize with custom parameters
pipeline = RobustAEPPipeline(data_root="custom_data")

# Process specific state only
metadata = pipeline._load_station_metadata()
delhi_stations = metadata.get('Delhi', {})

for station_id, station_info in delhi_stations.items():
    success = pipeline._process_station_complete('Delhi', station_id, station_info)
    print(f"Station {station_id}: {'Success' if success else 'Failed'}")
```

### **Data Quality Analysis**

```python
import pandas as pd

# Load and analyze unified dataset
df = pd.read_csv('data/unified_dataset.csv')

# Data quality report
print("Dataset Overview:")
print(f"Total records: {len(df):,}")
print(f"Date range: {df['timestamp_utc'].min()} to {df['timestamp_utc'].max()}")
print(f"States: {df['state'].unique()}")
print(f"Stations: {df['station_id'].nunique()}")

# PM2.5 statistics
print(f"\nPM2.5 Statistics:")
print(f"Mean: {df['PM2.5'].mean():.2f} μg/m³")
print(f"Range: {df['PM2.5'].min():.2f} - {df['PM2.5'].max():.2f} μg/m³")

# Missing data analysis
print(f"\nMissing Data:")
print(df.isnull().sum())
```

### **Error Recovery**

```python
# Resume pipeline from specific step
pipeline = RobustAEPPipeline()

# Skip to dataset creation if processing is complete
if Path('data/processed_data').exists() and len(list(Path('data/processed_data').glob('*.csv'))) > 0:
    print("Found processed data, creating unified dataset...")
    result = pipeline._create_unified_dataset()
else:
    print("Running complete pipeline...")
    result = pipeline.run_complete_pipeline()
```

---

## Conclusion

This documentation provides complete understanding of the AEP 3.0 Air Emissions Prediction Pipeline. The **`robust_aep_pipeline_final.py`** is the recommended main pipeline file that handles all edge cases and provides reliable execution.

For any issues or questions:
1. Check the troubleshooting guide
2. Review log files in `logs/` directory
3. Test individual components using utility scripts
4. Verify data structure and format requirements

The pipeline is designed to be robust, efficient, and maintainable for long-term air quality prediction research and applications.
