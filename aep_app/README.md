# AEP 3.0 GUI Application

A Windows desktop application for running the Air Emissions Prediction Pipeline with a user-friendly graphical interface.

## Features

- **Simple Interface**: Easy folder selection and one-click pipeline execution
- **Real-time Logging**: View pipeline progress and status in real-time
- **Error Handling**: Comprehensive error reporting and user feedback
- **Portable**: Can be packaged into a single .exe file for distribution
- **Two-Stage Pipeline**: Automated cropping and feature extraction + merging

## Files Structure

```
aep_app/
├── app.py              # Main GUI application
├── build_exe.py        # PyInstaller build script
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Quick Start

### Running from Python

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python app.py
   ```

### Building Executable

1. **Install PyInstaller**:
   ```bash
   pip install pyinstaller
   ```

2. **Build the .exe**:
   ```bash
   python build_exe.py
   ```

3. **Find your executable**:
   - Location: `dist/AEP_3.0_Pipeline.exe`
   - Launcher: `run_aep.bat`

## How to Use

1. **Launch the Application**
   - Run `python app.py` or double-click the .exe file

2. **Select Input Folder**
   - Click "Browse" next to "Raw INSAT Images Folder"
   - Select the folder containing your raw INSAT satellite images
   - Images should be organized in year subfolders

3. **Select Output Folder**
   - Click "Browse" next to "Output Folder"
   - Choose where you want the results saved
   - The app will create necessary subfolders automatically

4. **Run Pipeline**
   - Click "Run Pipeline" to start processing
   - Monitor progress in the log area
   - The pipeline will:
     - Copy CPCB station data to output location
     - Crop satellite images for each station
     - Extract features from cropped images
     - Merge with CPCB air quality data
     - Create final unified dataset

5. **View Results**
   - `final_dataset.csv` - Main output with all merged data
   - `station_datasets/` - Individual station CSV files
   - Log files for debugging

## Pipeline Stages

### Stage 1: Image Cropping
- Processes raw INSAT satellite images
- Crops images to station-specific geographic bounds
- Handles multiple years of data automatically
- Skips already processed images for efficiency

### Stage 2: Feature Extraction & Merging
- Extracts statistical features from cropped images
- Loads and cleans CPCB air quality data
- Merges image features with PM2.5 measurements by timestamp
- Creates unified dataset for machine learning

## Input Requirements

### Raw INSAT Images Folder Structure
```
raw_images/
├── 2021/
│   ├── image_20210101_0800.tif
│   ├── image_20210101_0900.tif
│   └── ...
├── 2022/
│   ├── image_20220101_0800.tif
│   └── ...
└── ...
```

### CPCB Data (Included)
- Station metadata with coordinates
- Historical PM2.5 measurements
- Automatically copied from parent project

## Output Structure

```
output_folder/
├── final_dataset.csv           # Main unified dataset
├── station_datasets/           # Individual station files
│   ├── Delhi_DL001_merged.csv
│   ├── Delhi_DL009_merged.csv
│   └── ...
├── aep_data/                   # Working directory
│   ├── cropped_data/           # Cropped images
│   ├── image_features/         # Extracted features
│   └── processed_data/         # Intermediate files
└── *.log                       # Pipeline log files
```

## System Requirements

- **Operating System**: Windows 10/11
- **Python**: 3.8+ (if running from source)
- **Memory**: 4GB+ RAM recommended
- **Storage**: 2GB+ free space for processing
- **Data Size**: Handles 500MB - 1GB datasets efficiently

## Troubleshooting

### Common Issues

1. **"Pipeline modules not found"**
   - Ensure the app is in the same directory as the main pipeline files
   - Check that `robust_aep_pipeline_final.py` or `aep_31_complete_pipeline.py` exists

2. **"CPCB data directory not found"**
   - Verify `data/cpcb/` folder exists in the parent directory
   - Contains station coordinate files and PM2.5 data

3. **Memory errors with large datasets**
   - Close other applications to free memory
   - Process smaller date ranges if possible

4. **Executable doesn't start**
   - Run from command prompt to see error messages
   - Check Windows Defender/antivirus settings

### Log Analysis

- **Green messages**: Successful operations
- **Orange messages**: Warnings (usually non-critical)
- **Red messages**: Errors requiring attention
- **Blue messages**: General information

## Building for Distribution

The `build_exe.py` script creates a standalone executable that includes:
- All Python dependencies
- Pipeline code and modules
- CPCB station data
- GUI interface

**PyInstaller Command** (automated by build_exe.py):
```bash
pyinstaller --onefile --windowed --name=AEP_3.0_Pipeline 
    --add-data="../data/cpcb;data/cpcb" 
    --add-data="../modules;modules"
    --hidden-import=pandas --hidden-import=rasterio
    app.py
```

## Technical Details

- **GUI Framework**: Tkinter (built into Python)
- **Threading**: Separate thread for pipeline execution
- **Logging**: Real-time log capture and display
- **Error Handling**: Comprehensive try-catch blocks
- **Data Handling**: Efficient memory management for large datasets

## Support

For issues or questions:
1. Check the log output for specific error messages
2. Verify input data format and structure
3. Ensure all dependencies are installed
4. Check system resources (memory, disk space)

---

**AEP 3.0 Team - 2025**
