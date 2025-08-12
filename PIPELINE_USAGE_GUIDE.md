# AEP 3.0 Pipeline Usage Guide

## 🌟 Overview

The AEP (Air Emissions Prediction) 3.0 pipeline is a comprehensive system for processing INSAT satellite imagery and CPCB air quality data to create unified datasets for PM2.5 prediction machine learning models.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Main Pipeline
```bash
python robust_aep_pipeline_final.py
```

## 🏗️ Pipeline Architecture

The pipeline follows a modular architecture with these core components:

```
Main Pipeline (robust_aep_pipeline_final.py)
├── 📊 Metadata Parser (modules/metadata_parser.py)
├── ✂️ Satellite Cropper (modules/satellite_cropper.py)
├── 🔗 Data Aligner (modules/data_aligner.py)
├── 🎯 Feature Extractor (enhanced_feature_pipeline_v2.py)
└── 🛠️ Utilities (modules/utils.py)
```

## 📁 Data Structure

Your data should be organized as follows:

```
data/
├── raw/                    # Raw INSAT satellite images (TIR1, WV bands)
│   ├── 3RIMG_01NOV2021_0815_L1B_STD_V01R00_IMG_TIR1.tif
│   └── 3RIMG_01NOV2021_0815_L1B_STD_V01R00_IMG_WV.tif
├── cpcb/                   # CPCB station data by state
│   ├── Delhi/
│   │   ├── DL009.csv       # Station data files
│   │   ├── DL010.csv
│   │   └── Delhi_coordinates.csv  # Station metadata
│   ├── Haryana/
│   │   ├── HR001.csv
│   │   └── Haryana_coordinates.csv
│   ├── Karnataka/
│   └── Maharashtra/
├── cropped_data/           # Generated: Cropped satellite images
│   ├── Delhi/
│   │   └── DL009/
│   │       └── 2021/
├── image_features/         # Generated: Extracted image features
├── processed_data/         # Generated: Processed station datasets
│   ├── Delhi/
│   │   └── DL009_merged.csv
└── unified_dataset.csv     # Generated: Final unified dataset
```

## 🎯 Core Pipeline Files

### Main Pipeline Options

1. **`robust_aep_pipeline_final.py`** ⭐ **RECOMMENDED**
   - Complete, production-ready pipeline
   - Comprehensive error handling
   - Handles "From Date" column issue
   - Skip logic for processed images

2. **`aep_31_complete_pipeline.py`** 🔄 **ALTERNATIVE**
   - Fresh rebuild implementation
   - Similar functionality, different approach
   - Good for comparison/validation

### Feature Extraction Options

1. **`enhanced_feature_pipeline_v2.py`** ⭐ **ADVANCED**
   - Per-band (TIR1/WV) processing
   - Non-lossy approach (preserves all CPCB data)
   - Feature caching for efficiency
   - ±15 minute timestamp matching

2. **`feature_extraction_pipeline.py`** 🔧 **BASIC**
   - Simple statistical features
   - Basic timestamp matching
   - Good for quick processing

## 💻 Usage Examples

### 1. Complete Pipeline Execution
```python
from robust_aep_pipeline_final import RobustAEPPipeline

# Initialize pipeline
pipeline = RobustAEPPipeline(data_root="data")

# Run complete pipeline
results = pipeline.run_complete_pipeline()

print(f"Processed {results['stations_processed']} stations")
print(f"Created unified dataset with {results['unified_dataset']['total_records']} records")
```

### 2. Enhanced Feature Extraction
```python
from enhanced_feature_pipeline_v2 import EnhancedINSATFeatureExtractor

# Initialize enhanced feature extractor
extractor = EnhancedINSATFeatureExtractor("c:/path/to/APE_07")

# Run enhanced pipeline
success = extractor.run_enhanced_pipeline()

if success:
    print("✅ Enhanced feature extraction completed!")
    print(f"📊 Dataset saved as: {extractor.unified_dataset_path}")
```

### 3. Testing CPCB Data Loading
```bash
python test_cpcb_loading.py
```

### 4. Analyze Data Quality
```bash
python analyze_pm25_quality.py
```

## ⚙️ Configuration Options

### Pipeline Parameters
- **`data_root`**: Base directory for all data files (default: "data")
- **`buffer_km`**: Cropping buffer around stations (default: 10.0 km)
- **`time_tolerance`**: Maximum time difference for matching (default: ±15 minutes)
- **`max_hours_diff`**: Maximum hours difference for alignment (default: 1 hour)

### Feature Extraction Parameters
- **`feature_type`**: "statistics", "histogram", "texture"
- **`chunk_size`**: Processing chunk size for memory management
- **`cache_features`**: Enable/disable feature caching

## 📊 Output Files and Formats

### 1. Unified Dataset (`unified_dataset.csv`)
Main training dataset with columns:
- `timestamp_utc`: UTC timestamp
- `PM2.5`: Target variable (μg/m³)
- `state`: State name
- `station_id`: Station identifier
- `img_*_TIR1`: TIR1 band features (mean, std, min, max, etc.)
- `img_*_WV`: Water Vapor band features
- `has_TIR1`, `has_WV`: Indicator columns for image availability

### 2. Station Datasets (`processed_data/State/StationID_merged.csv`)
Individual station datasets with:
- CPCB measurements
- Matched satellite features
- Quality flags
- Metadata

### 3. Cropped Images (`cropped_data/State/StationID/Year/`)
Cropped satellite images:
- Format: TIFF files
- Naming: `original_filename_cropped.tif`
- Organized by state/station/year

## 🔧 Debugging and Testing Tools

### 1. Test CPCB Loading
```bash
python test_cpcb_loading.py
```
**Purpose**: Verify CPCB data loading, timestamp parsing, and PM2.5 data quality

### 2. Analyze PM2.5 Quality
```bash
python analyze_pm25_quality.py
```
**Purpose**: Identify stations with data quality issues (constant values, extreme values, etc.)

### 3. Verify Timestamp Matching
```bash
python verification.py
```
**Purpose**: Debug timestamp alignment between CPCB data and satellite images

### 4. Create Unified Dataset
```bash
python create_unified_dataset.py
```
**Purpose**: Standalone script to recreate unified dataset from processed files

## 🚨 Troubleshooting Guide

### Common Issues and Solutions

#### 1. **"From Date" Column Not Found**
**Error**: CPCB data loading fails
**Solution**: The pipeline now handles "From Date" column automatically
```python
# Fixed in robust_aep_pipeline_final.py
timestamp_cols = ['From Date', 'timestamp', 'datetime', 'Date']
```

#### 2. **Unicode Encoding Errors**
**Error**: Emoji characters in logging cause crashes
**Solution**: Use text-based logging format
```python
# Use [WARNING], [ERROR] instead of emoji characters
self.logger.warning("[WARNING] Processing failed")
```

#### 3. **Memory Issues with Large Datasets**
**Error**: Out of memory during processing
**Solution**: Use chunked processing
```python
# Process in chunks
chunk_size = 10000
for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    # Process chunk
```

#### 4. **Timestamp Alignment Issues**
**Error**: No matches between CPCB and satellite data
**Solution**: Check timezone handling and time tolerance
```python
# Convert IST to UTC properly
ist_tz = pytz.timezone('Asia/Kolkata')
utc_tz = pytz.UTC
df['timestamp_utc'] = df['timestamp_ist'].dt.tz_convert(utc_tz)
```

#### 5. **Missing Satellite Images**
**Error**: No cropped images found
**Solution**: Check raw data directory structure and file patterns
```python
# Verify raw data structure
raw_dir = Path("data/raw")
image_files = list(raw_dir.glob("**/*.tif"))
print(f"Found {len(image_files)} raw images")
```

## 🎯 Performance Optimization

### 1. Hardware Recommendations
- **Storage**: SSD for faster I/O operations
- **RAM**: 16GB+ for processing large datasets
- **CPU**: Multi-core for parallel processing

### 2. Software Optimizations
- **Enable feature caching** to avoid recomputation
- **Use chunked processing** for memory efficiency
- **Process states in parallel** if resources allow
- **Skip already processed images** using built-in logic

### 3. Pipeline Optimization Tips
```python
# Enable skip logic for processed images
if cropped_path.exists():
    cropped_images.append(str(cropped_path))
    continue  # Skip already processed

# Use efficient timestamp matching
merged_df = pd.merge_asof(
    cpcb_sorted, image_sorted,
    on='timestamp_utc',
    tolerance=time_tolerance,
    direction='nearest'
)
```

## 📈 Pipeline Workflow

### Step-by-Step Execution

1. **Initialization**
   - Setup directories and logging
   - Load configuration parameters
   - Initialize timezone objects

2. **Metadata Loading**
   - Parse station coordinate files
   - Validate station metadata
   - Calculate bounding boxes

3. **Image Cropping**
   - Find raw INSAT images
   - Crop images to station bounds
   - Skip already processed images

4. **Feature Extraction**
   - Extract statistical features from cropped images
   - Handle TIR1 and WV bands separately
   - Cache features for efficiency

5. **CPCB Data Processing**
   - Load and clean CPCB station data
   - Handle "From Date" column
   - Filter time window (8-16h IST)
   - Convert to UTC

6. **Data Alignment**
   - Match satellite images to CPCB timestamps
   - Use nearest timestamp within tolerance
   - Preserve all CPCB records (non-lossy)

7. **Dataset Creation**
   - Merge features with CPCB data
   - Create station-specific datasets
   - Generate unified training dataset

## 📚 Advanced Usage

### Custom Feature Extraction
```python
class CustomFeatureExtractor(EnhancedINSATFeatureExtractor):
    def extract_custom_features(self, image_path, band):
        # Add your custom feature extraction logic
        features = super().extract_image_features(image_path, band)
        
        # Add custom features
        features[f'custom_feature_{band}'] = self.calculate_custom_metric(image_path)
        
        return features
```

### Parallel Processing
```python
from concurrent.futures import ProcessPoolExecutor

def process_state_parallel(states):
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_single_state, state) for state in states]
        results = [future.result() for future in futures]
    return results
```

## 🔍 Quality Assurance

### Data Validation Checks
- **PM2.5 Range**: 0-1000 μg/m³ (flags extreme values)
- **Timestamp Validity**: Proper datetime format and timezone
- **Image Quality**: Non-empty, valid TIFF files
- **Feature Completeness**: Minimum required features present
- **Coordinate Validation**: Lat/lon within valid ranges

### Quality Metrics
- **Data Completeness**: % of records with all required fields
- **Temporal Coverage**: Date range and frequency
- **Spatial Coverage**: Number of stations and geographic distribution
- **Feature Coverage**: % of records with satellite features

## 📞 Support and Maintenance

### Log Files
- `robust_aep_YYYYMMDD_HHMMSS.log`: Main pipeline logs
- `enhanced_feature_extraction_v2.log`: Feature extraction logs
- `feature_extraction.log`: Basic feature extraction logs

### Monitoring Pipeline Health
```python
# Check pipeline status
stats = pipeline.stats
print(f"Stations processed: {stats['stations_processed']}")
print(f"Stations failed: {stats['stations_failed']}")
print(f"Images cropped: {stats['images_cropped']}")
print(f"Features extracted: {stats['features_extracted']}")
```

### Regular Maintenance Tasks
1. **Clean up temporary files** in cropped_data if needed
2. **Monitor disk space** usage
3. **Update coordinate files** when new stations are added
4. **Validate data quality** periodically
5. **Update dependencies** as needed

---

## 🎉 Success Indicators

Your pipeline is working correctly when you see:
- ✅ All stations processed without critical errors
- ✅ Unified dataset created with expected record count
- ✅ Feature completeness >80% for both TIR1 and WV bands
- ✅ PM2.5 values in reasonable range (0-500 μg/m³ typically)
- ✅ Temporal coverage matches expected date ranges
- ✅ No Unicode or encoding errors in logs

For additional help, refer to the comprehensive documentation files and use the debugging utilities provided with the pipeline.

## 📋 File Management

### Essential Files to Keep
- `robust_aep_pipeline_final.py` (main pipeline)
- `modules/` directory (all files)
- `enhanced_feature_pipeline_v2.py` (advanced features)
- Testing utilities (`test_cpcb_loading.py`, `analyze_pm25_quality.py`, etc.)

### Files to Remove (See FILE_CLASSIFICATION.md)
- 10 redundant pipeline files that are superseded
- Empty or incomplete files
- Legacy implementations

This will keep your codebase clean and focused on the working components.

## Author : Lalit K Hire (BSc Data Science, Ty @ Department Of Technology, Savitribai Phule Pune University)
