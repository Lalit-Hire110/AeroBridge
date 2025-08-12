# AEP 3.0: Air Emissions Prediction System

A comprehensive Python-based pipeline for predicting PM2.5 levels at CPCB monitoring stations across India using INSAT-3D/3DR satellite images.

## 🌟 Overview

The AEP 3.0 system processes satellite imagery and ground truth air pollution station data to create training datasets for machine learning models. The pipeline handles:

- **Satellite Image Processing**: Crops INSAT satellite images to station-specific regions
- **Data Alignment**: Matches satellite data with CPCB station measurements using timestamps
- **Feature Extraction**: Extracts statistical, histogram, and texture features from satellite images
- **Dataset Creation**: Generates per-station and unified datasets ready for ML training

## 📁 Project Structure

```
AEP_07/
├── aep_pipeline.py          # Main pipeline orchestrator
├── modules/                 # Modular components
│   ├── __init__.py
│   ├── utils.py            # Utility functions
│   ├── metadata_parser.py  # Station metadata parser
│   ├── satellite_cropper.py # Satellite image cropper
│   ├── data_aligner.py     # Data alignment module
│   └── dataset_merger.py   # Dataset merger
├── data/                   # Data directory
│   ├── raw/               # Raw satellite images
│   │   ├── Haryana/
│   │   ├── Maharashtra/
│   │   ├── Karnataka/
│   │   └── Delhi/
│   └── cpcb/              # CPCB station data
│       ├── Haryana/
│       ├── Maharashtra/
│       ├── Karnataka/
│       └── Delhi/
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Git

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd AEP_07
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv aep_env
   source aep_env/bin/activate  # On Windows: aep_env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Data Preparation

Ensure your data follows this structure:

```
data/
├── raw/                    # INSAT satellite images
│   ├── Haryana/
│   │   ├── 2021/
│   │   │   ├── 3RIMG_01NOV2021_0815_L1B_STD_V01R00_IMG_WV.tif
│   │   │   └── ...
│   │   └── 2022/
│   └── ...
└── cpcb/                   # CPCB station data
    ├── Haryana/
    │   ├── haryana_coordinates.csv  # Station metadata
    │   ├── HR001.csv               # Station data
    │   └── ...
    └── ...
```

## 📖 Usage

### Basic Usage

Run the complete pipeline for all available states:

```bash
python aep_pipeline.py
```

### Advanced Usage

```bash
# Process specific states
python aep_pipeline.py --states Haryana Maharashtra

# Use custom buffer distance for cropping
python aep_pipeline.py --buffer-km 15.0

# Skip unified dataset creation
python aep_pipeline.py --no-unified

# Check pipeline status
python aep_pipeline.py --status

# Use custom data directory
python aep_pipeline.py --data-root /path/to/data
```

### Programmatic Usage

```python
from aep_pipeline import AEPPipeline

# Initialize pipeline
pipeline = AEPPipeline(data_root="data", buffer_km=10.0)

# Run pipeline for specific states
results = pipeline.run_pipeline(
    states=["Haryana", "Maharashtra"],
    create_unified_dataset=True
)

# Check pipeline status
status = pipeline.get_pipeline_status()
print(status)
```

## 🔧 Configuration

### Pipeline Parameters

- `data_root`: Root directory containing raw and cpcb data (default: "data")
- `buffer_km`: Buffer distance in kilometers for satellite image cropping (default: 10.0)
- `max_hours_diff`: Maximum allowed difference in hours for timestamp matching (default: 1)
- `feature_type`: Type of satellite features to extract (default: "statistics")

### Feature Extraction Types

1. **statistics**: Basic statistical features (mean, std, min, max, median, quartiles)
2. **histogram**: Histogram-based features (10-bin histograms per band)
3. **texture**: Texture features using Gray-Level Co-occurrence Matrix (GLCM)

## 📊 Output Structure

The pipeline generates the following outputs:

```
data/
├── cropped_data/           # Cropped satellite images
│   ├── Haryana/
│   │   ├── HR001/
│   │   └── ...
│   └── ...
├── processed_data/         # Processed datasets
│   ├── Haryana/
│   │   ├── HR001.csv
│   │   └── ...
│   ├── unified_dataset.csv # Unified dataset
│   └── unified_dataset_metadata.json
└── aep_pipeline.log       # Pipeline logs
```

## 🔍 Data Format

### Input Data

#### Satellite Images
- Format: GeoTIFF (.tif)
- Naming: `3RIMG_01NOV2021_0815_L1B_STD_V01R00_IMG_WV.tif`
- Bands: WV (Water Vapor), TIR1 (Thermal Infrared)

#### CPCB Station Data
- Format: CSV
- Columns: timestamp, PM2.5, PM10, NO, NO2, NOx, NH3, SO2, CO, Ozone, weather data
- Frequency: Hourly data

#### Station Metadata
- Format: CSV
- Columns: file_name, station_location, latitude, longitude, bounding_box_coordinates

### Output Data

#### Processed Datasets
- Format: CSV
- Features: CPCB measurements + satellite features + metadata
- Structure: One row per timestamp with aligned satellite and ground truth data

## 🛠️ Modules

### Metadata Parser (`modules/metadata_parser.py`)
- Parses station metadata files
- Validates coordinate data
- Provides station bounding boxes with buffer

### Satellite Cropper (`modules/satellite_cropper.py`)
- Crops satellite images to station regions
- Handles geospatial operations
- Validates cropped image quality

### Data Aligner (`modules/data_aligner.py`)
- Aligns satellite and CPCB data by timestamps
- Handles timezone conversions (IST ↔ UTC)
- Extracts features from satellite images

### Dataset Merger (`modules/dataset_merger.py`)
- Merges multiple station datasets
- Standardizes column names and data types
- Creates unified datasets for ML training

### Utilities (`modules/utils.py`)
- Common utility functions
- Feature extraction from satellite images
- Data validation and quality checks

## 📈 Performance

### Processing Time
- **Satellite Cropping**: ~2-5 minutes per station per year
- **Data Alignment**: ~1-3 minutes per station
- **Feature Extraction**: ~30-60 seconds per station
- **Dataset Merging**: ~10-30 seconds for unified dataset

### Memory Requirements
- **Minimum**: 4GB RAM
- **Recommended**: 8GB+ RAM for large datasets
- **Storage**: 2-5x original data size (depending on buffer size)

## 🐛 Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Geospatial Libraries**
   ```bash
   # On Windows, install GDAL first
   conda install -c conda-forge gdal
   pip install rasterio geopandas
   ```

3. **Memory Issues**
   - Reduce buffer size: `--buffer-km 5.0`
   - Process fewer states at once
   - Use smaller time windows

4. **No Satellite Images Found**
   - Check file extensions (.tif, .tiff, .img)
   - Verify directory structure
   - Check file permissions

### Logging

The pipeline generates detailed logs in `aep_pipeline.log`. Check this file for:
- Processing progress
- Error messages
- Performance statistics
- Data quality warnings

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the logs for error details

## 🔬 Research

This pipeline is designed for research in:
- Air quality prediction
- Remote sensing applications
- Environmental monitoring
- Machine learning for environmental data

## 📚 References

- INSAT-3D/3DR satellite data documentation
- CPCB air quality monitoring standards
- Geospatial data processing best practices
- Machine learning for environmental applications 