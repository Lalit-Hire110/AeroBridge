#!/usr/bin/env python3
"""
AEP 3.1: Complete Air Emissions Prediction Pipeline

This is a complete rebuild from scratch that executes the entire pipeline:
1. Station-wise image cropping from raw INSAT data
2. Feature extraction from cropped images
3. CPCB data cleaning and time filtering
4. Merging image features with CPCB data
5. Creating unified dataset

Author: AEP 3.1 Team
Date: 2024
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import json
from PIL import Image
import rasterio
from rasterio.windows import Window
import glob

warnings.filterwarnings('ignore')

class AEP31Pipeline:
    """Complete AEP 3.1 Pipeline - Fresh rebuild from raw data to unified dataset."""
    
    def __init__(self, data_root: str = "data"):
        """Initialize the AEP 3.1 pipeline."""
        self.data_root = Path(data_root)
        
        # Initialize directories
        self.raw_dir = self.data_root / "raw"
        self.cpcb_dir = self.data_root / "cpcb"
        self.cropped_dir = self.data_root / "cropped_data"
        self.features_dir = self.data_root / "image_features"
        self.processed_dir = self.data_root / "processed_data"
        
        # Create output directories
        for directory in [self.cropped_dir, self.features_dir, self.processed_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('aep_31_pipeline.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Timezone objects
        self.ist_tz = pytz.timezone('Asia/Kolkata')
        self.utc_tz = pytz.UTC
        
        # Pipeline statistics
        self.stats = {
            'stations_processed': 0,
            'stations_failed': 0,
            'images_cropped': 0,
            'features_extracted': 0,
            'records_merged': 0,
            'failed_stations': []
        }
        
        self.logger.info("[INIT] AEP 3.1 Pipeline initialized - Fresh rebuild from scratch")
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Execute the complete AEP 3.1 pipeline from raw data to unified dataset."""
        self.logger.info("="*80)
        self.logger.info("[START] STARTING AEP 3.1 COMPLETE PIPELINE EXECUTION")
        self.logger.info("="*80)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Load station metadata
            self.logger.info("[STEP 1] Loading station metadata...")
            metadata = self._load_station_metadata()
            if not metadata:
                raise Exception("No station metadata found")
            
            self.logger.info(f"[SUCCESS] Found {len(metadata)} stations across all states")
            
            # Step 2: Process each station through the complete pipeline
            self.logger.info("[STEP 2] Processing stations through complete pipeline...")
            
            for state, stations in metadata.items():
                self.logger.info(f"[STATE] Processing state: {state}")
                
                for station_id, station_info in stations.items():
                    self.logger.info(f"[STATION] Processing station: {station_id} ({station_info.get('station_location', 'Unknown')})")
                    
                    try:
                        # Execute complete pipeline for this station
                        success = self._process_station_complete(state, station_id, station_info)
                        
                        if success:
                            self.stats['stations_processed'] += 1
                            self.logger.info(f"[SUCCESS] Station {station_id} processed successfully")
                        else:
                            self.stats['stations_failed'] += 1
                            self.stats['failed_stations'].append(f"{state}/{station_id}")
                            self.logger.warning(f"[WARNING] Station {station_id} processing failed")
                    
                    except Exception as e:
                        self.stats['stations_failed'] += 1
                        self.stats['failed_stations'].append(f"{state}/{station_id}")
                        self.logger.error(f"[ERROR] Error processing station {station_id}: {str(e)}")
            
            # Step 3: Create unified dataset
            self.logger.info("[STEP 3] Creating unified dataset...")
            unified_result = self._create_unified_dataset()
            
            # Calculate final statistics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 60
            
            # Final results
            results = {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_minutes': duration,
                'stations_processed': self.stats['stations_processed'],
                'stations_failed': self.stats['stations_failed'],
                'failed_stations': self.stats['failed_stations'],
                'images_cropped': self.stats['images_cropped'],
                'features_extracted': self.stats['features_extracted'],
                'records_merged': self.stats['records_merged'],
                'unified_dataset': unified_result
            }
            
            self._print_final_summary(results)
            return results
            
        except Exception as e:
            self.logger.error(f"💥 Pipeline execution failed: {str(e)}")
            raise
    
    def _load_station_metadata(self) -> Dict[str, Dict[str, Dict]]:
        """Load station metadata from coordinate files."""
        metadata = {}
        
        # Check each state directory for coordinate files
        for state_dir in self.cpcb_dir.iterdir():
            if not state_dir.is_dir():
                continue
            
            state_name = state_dir.name
            coord_file = state_dir / f"{state_name.lower()}_coordinates.csv"
            
            if coord_file.exists():
                try:
                    df = pd.read_csv(coord_file)
                    state_stations = {}
                    
                    for _, row in df.iterrows():
                        station_id = row['file_name']
                        state_stations[station_id] = {
                            'station_location': row['station_location'],
                            'latitude': float(row['latitude']),
                            'longitude': float(row['longitude']),
                            'west_lon': float(row['west_lon']),
                            'east_lon': float(row['east_lon']),
                            'south_lat': float(row['south_lat']),
                            'north_lat': float(row['north_lat'])
                        }
                    
                    metadata[state_name] = state_stations
                    self.logger.info(f"📍 Loaded {len(state_stations)} stations for {state_name}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to load metadata for {state_name}: {e}")
        
        return metadata
    
    def _process_station_complete(self, state: str, station_id: str, station_info: Dict) -> bool:
        """Execute complete pipeline for a single station."""
        try:
            # Step 1: Crop images for this station
            cropped_images = self._crop_station_images(state, station_id, station_info)
            if not cropped_images:
                self.logger.warning(f"⚠️  No images cropped for {station_id}")
                return False
            
            # Step 2: Extract features from cropped images
            features_file = self._extract_image_features(state, station_id, cropped_images)
            if not features_file:
                self.logger.warning(f"⚠️  No features extracted for {station_id}")
                return False
            
            # Step 3: Load and clean CPCB data
            cpcb_data = self._load_clean_cpcb_data(state, station_id)
            if cpcb_data is None or len(cpcb_data) == 0:
                self.logger.warning(f"[WARNING] No CPCB data for {station_id}")
                return False
            
            # Step 4: Merge image features with CPCB data
            merged_data = self._merge_features_with_cpcb(features_file, cpcb_data, station_info, state, station_id)
            if merged_data is None or len(merged_data) == 0:
                self.logger.warning(f"⚠️  No merged data for {station_id}")
                return False
            
            # Step 5: Save processed data
            output_dir = self.processed_dir / state
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{station_id}.csv"
            merged_data.to_csv(output_file, index=False)
            
            self.stats['records_merged'] += len(merged_data)
            return True
            
        except Exception as e:
            self.logger.error(f"💥 Error in complete processing for {station_id}: {e}")
            return False
    
    def _crop_station_images(self, state: str, station_id: str, station_info: Dict) -> List[str]:
        """Crop INSAT images for a specific station."""
        cropped_files = []
        
        # Get station bounding box with buffer
        lat = station_info['latitude']
        lon = station_info['longitude']
        buffer = 0.1  # degrees
        
        min_lat = lat - buffer
        max_lat = lat + buffer
        min_lon = lon - buffer
        max_lon = lon + buffer
        
        # Find raw images for this state
        state_raw_dir = self.raw_dir / state
        if not state_raw_dir.exists():
            self.logger.warning(f"⚠️  No raw images directory for {state}")
            return cropped_files
        
        # Process images by year
        for year_dir in state_raw_dir.iterdir():
            if not year_dir.is_dir():
                continue
            
            year = year_dir.name
            
            # Create output directory for this station/year
            crop_output_dir = self.cropped_dir / state / station_id / year
            crop_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Find all image files
            image_files = []
            for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
                image_files.extend(glob.glob(str(year_dir / "**" / ext), recursive=True))
            
            for image_file in image_files:
                try:
                    # Extract timestamp from filename if possible
                    image_path = Path(image_file)
                    timestamp_str = self._extract_timestamp_from_filename(image_path.name)
                    
                    # Crop the image
                    cropped_file = self._crop_single_image(
                        image_file, crop_output_dir, min_lat, max_lat, min_lon, max_lon, timestamp_str
                    )
                    
                    if cropped_file:
                        cropped_files.append(cropped_file)
                        self.stats['images_cropped'] += 1
                
                except Exception as e:
                    self.logger.warning(f"⚠️  Failed to crop {image_file}: {e}")
        
        self.logger.info(f"[CROPPED] Cropped {len(cropped_files)} images for {station_id}")
        return cropped_files
    
    def _extract_timestamp_from_filename(self, filename: str) -> str:
        """Extract timestamp from image filename."""
        # Common patterns in INSAT filenames
        import re
        
        # Try to find date/time patterns
        patterns = [
            r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})',  # YYYYMMDD_HHMM
            r'(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})',   # YYYYMMDDHHMM
            r'(\d{8})_(\d{4})',                       # YYYYMMDD_HHMM
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                if len(match.groups()) == 5:
                    year, month, day, hour, minute = match.groups()
                    return f"{year}-{month}-{day} {hour}:{minute}:00"
                elif len(match.groups()) == 2:
                    date_part, time_part = match.groups()
                    year = date_part[:4]
                    month = date_part[4:6]
                    day = date_part[6:8]
                    hour = time_part[:2]
                    minute = time_part[2:4]
                    return f"{year}-{month}-{day} {hour}:{minute}:00"
        
        # Default timestamp if pattern not found
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _crop_single_image(self, image_file: str, output_dir: Path, 
                          min_lat: float, max_lat: float, min_lon: float, max_lon: float,
                          timestamp_str: str) -> Optional[str]:
        """Crop a single image to station bounds."""
        try:
            with rasterio.open(image_file) as src:
                # Get image bounds and transform
                bounds = src.bounds
                transform = src.transform
                
                # Check if station bounds intersect with image bounds
                if (max_lon < bounds.left or min_lon > bounds.right or 
                    max_lat < bounds.bottom or min_lat > bounds.top):
                    return None
                
                # Convert geographic coordinates to pixel coordinates
                col_min, row_max = ~transform * (min_lon, min_lat)
                col_max, row_min = ~transform * (max_lon, max_lat)
                
                # Ensure pixel coordinates are within image bounds
                col_min = max(0, int(col_min))
                col_max = min(src.width, int(col_max))
                row_min = max(0, int(row_min))
                row_max = min(src.height, int(row_max))
                
                if col_min >= col_max or row_min >= row_max:
                    return None
                
                # Create window and read data
                window = Window(col_min, row_min, col_max - col_min, row_max - row_min)
                cropped_data = src.read(window=window)
                
                # Create output filename
                base_name = Path(image_file).stem
                output_file = output_dir / f"{base_name}_cropped.tif"
                
                # Skip if already cropped
                if output_file.exists():
                    return str(output_file)
                
                # Write cropped image
                profile = src.profile.copy()
                profile.update({
                    'height': cropped_data.shape[1],
                    'width': cropped_data.shape[2],
                    'transform': rasterio.windows.transform(window, transform)
                })
                
                with rasterio.open(output_file, 'w', **profile) as dst:
                    dst.write(cropped_data)
                
                return str(output_file)
                
        except Exception as e:
            self.logger.warning(f"⚠️  Error cropping {image_file}: {e}")
            return None
    
    def _extract_image_features(self, state: str, station_id: str, cropped_images: List[str]) -> Optional[str]:
        """Extract features from cropped images."""
        features_data = []
        
        for image_file in cropped_images:
            try:
                # Extract timestamp from filename
                timestamp_str = self._extract_timestamp_from_filename(Path(image_file).name)
                
                # Read image and extract features
                with rasterio.open(image_file) as src:
                    data = src.read()
                    
                    # Extract basic features (mean pixel values per band)
                    features = {
                        'timestamp': timestamp_str,
                        'image_file': Path(image_file).name
                    }
                    
                    # Calculate mean pixel values for each band
                    for band_idx in range(data.shape[0]):
                        band_data = data[band_idx]
                        # Mask out invalid values
                        valid_data = band_data[band_data > 0]
                        if len(valid_data) > 0:
                            features[f'band_{band_idx+1}_mean'] = float(np.mean(valid_data))
                            features[f'band_{band_idx+1}_std'] = float(np.std(valid_data))
                            features[f'band_{band_idx+1}_min'] = float(np.min(valid_data))
                            features[f'band_{band_idx+1}_max'] = float(np.max(valid_data))
                        else:
                            features[f'band_{band_idx+1}_mean'] = np.nan
                            features[f'band_{band_idx+1}_std'] = np.nan
                            features[f'band_{band_idx+1}_min'] = np.nan
                            features[f'band_{band_idx+1}_max'] = np.nan
                    
                    features_data.append(features)
                    
            except Exception as e:
                self.logger.warning(f"⚠️  Failed to extract features from {image_file}: {e}")
        
        if not features_data:
            return None
        
        # Save features to CSV
        features_df = pd.DataFrame(features_data)
        features_df['timestamp'] = pd.to_datetime(features_df['timestamp'], errors='coerce')
        
        # Create output directory and file
        output_dir = self.features_dir / state
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{station_id}.csv"
        
        features_df.to_csv(output_file, index=False)
        
        self.stats['features_extracted'] += len(features_data)
        self.logger.info(f"📊 Extracted {len(features_data)} feature records for {station_id}")
        
        return str(output_file)
    
    def _load_clean_cpcb_data(self, state: str, station_id: str) -> Optional[pd.DataFrame]:
        """Load and clean CPCB data for a station."""
        cpcb_file = self.cpcb_dir / state / f"{station_id}.csv"
        
        if not cpcb_file.exists():
            self.logger.warning(f"[WARNING] CPCB file not found: {cpcb_file}")
            return None
        
        try:
            df = pd.read_csv(cpcb_file)
            
            # Find PM2.5 column
            pm25_col = None
            for col in ['PM2.5', 'PM2.5 (ug/m3)', 'pm2.5', 'PM25']:
                if col in df.columns:
                    pm25_col = col
                    break
            
            if pm25_col is None:
                self.logger.warning(f"[WARNING] No PM2.5 column found in {station_id}")
                return None
            
            # Clean PM2.5 data
            df[pm25_col] = pd.to_numeric(df[pm25_col], errors='coerce')
            df = df[df[pm25_col] >= 0]  # Remove negative values
            df = df.dropna(subset=[pm25_col])  # Remove NaN values
            
            # Process timestamps
            timestamp_col = None
            if 'Date' in df.columns and 'Time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
                timestamp_col = 'timestamp'
            else:
                for col in ['From Date', 'timestamp', 'datetime', 'Date']:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        timestamp_col = col
                        break
            
            if timestamp_col is None:
                self.logger.warning(f"[WARNING] No timestamp column found in {station_id}")
                return None
            
            # Remove rows with invalid timestamps
            df = df.dropna(subset=[timestamp_col])
            
            # Convert to IST and filter time window (08:00-16:00 local time)
            df[timestamp_col] = df[timestamp_col].dt.tz_localize(self.ist_tz, ambiguous='infer')
            df['hour'] = df[timestamp_col].dt.hour
            df = df[(df['hour'] >= 8) & (df['hour'] <= 16)]
            df = df.drop('hour', axis=1)
            
            # Convert to UTC
            df['timestamp_utc'] = df[timestamp_col].dt.tz_convert(self.utc_tz)
            
            # Rename PM2.5 column to standard name
            df = df.rename(columns={pm25_col: 'PM2.5'})
            
            self.logger.info(f"[CLEANED] Cleaned CPCB data for {station_id}: {len(df)} records")
            return df
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error loading CPCB data for {station_id}: {e}")
            return None
    
    def _merge_features_with_cpcb(self, features_file: str, cpcb_data: pd.DataFrame, 
                                 station_info: Dict, state: str, station_id: str) -> Optional[pd.DataFrame]:
        """Merge image features with CPCB data by timestamp."""
        try:
            # Load features
            features_df = pd.read_csv(features_file)
            features_df['timestamp'] = pd.to_datetime(features_df['timestamp'], errors='coerce')
            features_df = features_df.dropna(subset=['timestamp'])
            
            if len(features_df) == 0:
                return None
            
            # Convert features timestamp to UTC (assuming they are in UTC already)
            features_df['timestamp_utc'] = features_df['timestamp']
            
            # Merge with tolerance of ±1 hour
            merged_records = []
            
            for _, cpcb_row in cpcb_data.iterrows():
                cpcb_time = cpcb_row['timestamp_utc']
                
                # Find matching features within ±1 hour
                time_diff = abs(features_df['timestamp_utc'] - cpcb_time)
                tolerance = pd.Timedelta(hours=1)
                
                matching_features = features_df[time_diff <= tolerance]
                
                if len(matching_features) > 0:
                    # Use the closest match
                    closest_idx = time_diff.idxmin()
                    feature_row = features_df.loc[closest_idx]
                    
                    # Combine CPCB and feature data
                    merged_row = {}
                    
                    # Add CPCB data
                    for col in cpcb_data.columns:
                        if col not in ['timestamp', 'timestamp_utc']:
                            merged_row[col] = cpcb_row[col]
                    
                    # Add timestamp
                    merged_row['timestamp_utc'] = cpcb_time
                    
                    # Add image features
                    for col in features_df.columns:
                        if col not in ['timestamp', 'timestamp_utc']:
                            merged_row[f'img_{col}'] = feature_row[col]
                    
                    # Add station metadata
                    merged_row['station_id'] = station_id
                    merged_row['station_name'] = station_info.get('station_location', station_id)
                    merged_row['state'] = state
                    merged_row['latitude'] = station_info['latitude']
                    merged_row['longitude'] = station_info['longitude']
                    
                    merged_records.append(merged_row)
            
            if not merged_records:
                return None
            
            merged_df = pd.DataFrame(merged_records)
            self.logger.info(f"🔗 Merged {len(merged_df)} records for {station_id}")
            
            return merged_df
            
        except Exception as e:
            self.logger.error(f"💥 Error merging data for {station_id}: {e}")
            return None
    
    def _create_unified_dataset(self) -> Dict[str, Any]:
        """Create unified dataset from all processed station files."""
        self.logger.info("🔗 Creating unified dataset from all processed stations...")
        
        all_dataframes = []
        processed_stations = 0
        
        # Collect all processed files
        for state_dir in self.processed_dir.iterdir():
            if not state_dir.is_dir():
                continue
            
            for station_file in state_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(station_file)
                    
                    # Drop rows with missing PM2.5 or image features
                    initial_count = len(df)
                    df = df.dropna(subset=['PM2.5'])
                    
                    # Check for at least one image feature
                    img_cols = [col for col in df.columns if col.startswith('img_')]
                    if img_cols:
                        df = df.dropna(subset=img_cols, how='all')
                    
                    if len(df) > 0:
                        all_dataframes.append(df)
                        processed_stations += 1
                        self.logger.info(f"📄 Added {len(df)} records from {station_file.name} (removed {initial_count - len(df)} incomplete records)")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️  Failed to load {station_file}: {e}")
        
        if not all_dataframes:
            return {'success': False, 'error': 'No valid processed data found'}
        
        # Combine all dataframes
        unified_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Final cleaning
        unified_df = unified_df.drop_duplicates()
        unified_df = unified_df.sort_values(['state', 'station_id', 'timestamp_utc'])
        
        # Save unified dataset
        output_file = self.data_root / "unified_dataset.csv"
        unified_df.to_csv(output_file, index=False)
        
        result = {
            'success': True,
            'output_file': str(output_file),
            'total_records': len(unified_df),
            'total_stations': processed_stations,
            'columns': list(unified_df.columns),
            'states': unified_df['state'].unique().tolist(),
            'date_range': {
                'start': str(unified_df['timestamp_utc'].min()),
                'end': str(unified_df['timestamp_utc'].max())
            }
        }
        
        self.logger.info(f"[DATASET] Unified dataset created: {len(unified_df):,} records from {processed_stations} stations")
        return result
    
    def _print_final_summary(self, results: Dict[str, Any]):
        """Print final pipeline execution summary."""
        print("\n" + "="*80)
        print("[COMPLETE] AEP 3.1 PIPELINE EXECUTION COMPLETE")
        print("="*80)
        
        print(f"[TIME] Duration: {results['duration_minutes']:.2f} minutes")
        print(f"[SUCCESS] Stations processed successfully: {results['stations_processed']}")
        print(f"[FAILED] Stations failed: {results['stations_failed']}")
        
        if results['failed_stations']:
            print(f"[ERROR] Failed stations: {', '.join(results['failed_stations'])}")
        
        print(f"[IMAGES] Images cropped: {results['images_cropped']:,}")
        print(f"[FEATURES] Features extracted: {results['features_extracted']:,}")
        print(f"[MERGED] Records merged: {results['records_merged']:,}")
        
        if results['unified_dataset']['success']:
            unified = results['unified_dataset']
            print(f"[DATASET] Final unified dataset: {unified['total_records']:,} records")
            print(f"[STATIONS] Total stations in dataset: {unified['total_stations']}")
            print(f"[STATES] States: {', '.join(unified['states'])}")
            print(f"[DATES] Date range: {unified['date_range']['start']} to {unified['date_range']['end']}")
            print(f"[OUTPUT] Output file: {unified['output_file']}")
            print(f"[COLUMNS] Columns: {len(unified['columns'])}")
        else:
            print(f"[ERROR] Unified dataset creation failed: {results['unified_dataset'].get('error', 'Unknown error')}")
        
        print("="*80)


def main():
    """Main function to run the complete AEP 3.1 pipeline."""
    try:
        pipeline = AEP31Pipeline()
        results = pipeline.run_complete_pipeline()
        return results
    except Exception as e:
        print(f"[ERROR] Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
