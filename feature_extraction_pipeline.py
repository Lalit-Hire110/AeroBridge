"""
INSAT Image Feature Extraction Pipeline
======================================

This module extracts comprehensive features from cropped INSAT-3D/3DR images
and aligns them with CPCB station data to create an enhanced unified dataset.

Features extracted per image:
- Basic statistics: mean, std, min, max, median
- Percentiles: 25th, 75th, 90th, 95th
- Threshold percentages: >240, <50, >200, <100
- Texture features: range, coefficient of variation
- Band-specific naming (TIR1, WV)

Author: AEP 3.0 Pipeline
Date: 2025-08-05
"""

import os
import pandas as pd
import numpy as np
import rasterio
from datetime import datetime, timezone
import pytz
import logging
from pathlib import Path
import glob
from typing import Dict, List, Tuple, Optional
import re
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class INSATFeatureExtractor:
    """Extract comprehensive features from INSAT satellite images."""
    
    def __init__(self, base_path: str):
        """
        Initialize the feature extractor.
        
        Args:
            base_path: Path to the APE_07 project directory
        """
        self.base_path = Path(base_path)
        self.cropped_data_path = self.base_path / "data" / "cropped_data"
        self.processed_data_path = self.base_path / "data" / "processed_data"
        self.unified_dataset_path = self.base_path / "data" / "unified_dataset_enhanced.csv"
        
        # Ensure output directory exists
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        # IST timezone for timestamp conversion
        self.ist_tz = pytz.timezone('Asia/Kolkata')
        self.utc_tz = pytz.UTC
        
    def parse_filename(self, filename: str) -> Tuple[Optional[datetime], Optional[str]]:
        """
        Parse INSAT filename to extract timestamp and band.
        
        Example: 3RIMG_01NOV2021_0815_L1B_STD_V01R00_IMG_TIR1_cropped.tif
        Returns: (datetime object in UTC, band name)
        """
        try:
            # Extract date and time parts
            match = re.search(r'(\d{2}[A-Z]{3}\d{4})_(\d{4})', filename)
            if not match:
                logger.warning(f"Could not parse date/time from filename: {filename}")
                return None, None
                
            date_str, time_str = match.groups()
            
            # Extract band (TIR1 or WV)
            band_match = re.search(r'IMG_([A-Z0-9]+)_cropped', filename)
            if not band_match:
                logger.warning(f"Could not parse band from filename: {filename}")
                return None, None
                
            band = band_match.group(1)
            
            # Parse datetime
            dt_str = f"{date_str}_{time_str}"
            dt_ist = datetime.strptime(dt_str, "%d%b%Y_%H%M")
            
            # Convert to IST timezone, then to UTC
            dt_ist = self.ist_tz.localize(dt_ist)
            dt_utc = dt_ist.astimezone(self.utc_tz)
            
            return dt_utc, band
            
        except Exception as e:
            logger.error(f"Error parsing filename {filename}: {e}")
            return None, None
    
    def extract_image_features(self, image_path: str, band: str) -> Dict[str, float]:
        """
        Extract comprehensive features from a single TIFF image.
        
        Args:
            image_path: Path to the TIFF image
            band: Band name (TIR1 or WV) for feature naming
            
        Returns:
            Dictionary of features with band-specific names
        """
        try:
            with rasterio.open(image_path) as src:
                # Read the image data
                image_data = src.read(1)  # Single band
                
                # Handle nodata values
                if src.nodata is not None:
                    image_data = image_data[image_data != src.nodata]
                
                # Remove any infinite or NaN values
                image_data = image_data[np.isfinite(image_data)]
                
                if len(image_data) == 0:
                    logger.warning(f"No valid pixels in image: {image_path}")
                    return self._get_empty_features(band)
                
                # Basic statistics
                features = {
                    f'mean_{band}': float(np.mean(image_data)),
                    f'std_{band}': float(np.std(image_data)),
                    f'min_{band}': float(np.min(image_data)),
                    f'max_{band}': float(np.max(image_data)),
                    f'median_{band}': float(np.median(image_data)),
                }
                
                # Percentiles
                features.update({
                    f'p25_{band}': float(np.percentile(image_data, 25)),
                    f'p75_{band}': float(np.percentile(image_data, 75)),
                    f'p90_{band}': float(np.percentile(image_data, 90)),
                    f'p95_{band}': float(np.percentile(image_data, 95)),
                })
                
                # Threshold percentages
                total_pixels = len(image_data)
                features.update({
                    f'pct_above_240_{band}': float(np.sum(image_data > 240) / total_pixels * 100),
                    f'pct_below_50_{band}': float(np.sum(image_data < 50) / total_pixels * 100),
                    f'pct_above_200_{band}': float(np.sum(image_data > 200) / total_pixels * 100),
                    f'pct_below_100_{band}': float(np.sum(image_data < 100) / total_pixels * 100),
                })
                
                # Additional texture/statistical features
                features.update({
                    f'range_{band}': float(np.max(image_data) - np.min(image_data)),
                    f'cv_{band}': float(np.std(image_data) / np.mean(image_data)) if np.mean(image_data) != 0 else 0.0,
                    f'skewness_{band}': float(self._calculate_skewness(image_data)),
                    f'kurtosis_{band}': float(self._calculate_kurtosis(image_data)),
                })
                
                return features
                
        except Exception as e:
            logger.error(f"Error extracting features from {image_path}: {e}")
            return self._get_empty_features(band)
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of the data."""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            return np.mean(((data - mean) / std) ** 3)
        except:
            return 0.0
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of the data."""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            return np.mean(((data - mean) / std) ** 4) - 3  # Excess kurtosis
        except:
            return 0.0
    
    def _get_empty_features(self, band: str) -> Dict[str, float]:
        """Return empty/default features for failed extractions."""
        return {
            f'mean_{band}': np.nan,
            f'std_{band}': np.nan,
            f'min_{band}': np.nan,
            f'max_{band}': np.nan,
            f'median_{band}': np.nan,
            f'p25_{band}': np.nan,
            f'p75_{band}': np.nan,
            f'p90_{band}': np.nan,
            f'p95_{band}': np.nan,
            f'pct_above_240_{band}': np.nan,
            f'pct_below_50_{band}': np.nan,
            f'pct_above_200_{band}': np.nan,
            f'pct_below_100_{band}': np.nan,
            f'range_{band}': np.nan,
            f'cv_{band}': np.nan,
            f'skewness_{band}': np.nan,
            f'kurtosis_{band}': np.nan,
        }
    
    def process_station(self, state: str, station: str) -> bool:
        """
        Process all images for a single station and create enhanced CSV.
        
        Args:
            state: State name (e.g., 'Delhi')
            station: Station ID (e.g., 'DL009')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Processing station {state}_{station}")
            
            # Load existing merged data
            merged_file = self.processed_data_path / f"{state}_{station}_merged.csv"
            if not merged_file.exists():
                logger.error(f"Merged file not found: {merged_file}")
                return False
            
            df = pd.read_csv(merged_file)
            df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
            
            # Initialize feature columns
            feature_columns = []
            for band in ['TIR1', 'WV']:
                for feature in ['mean', 'std', 'min', 'max', 'median', 'p25', 'p75', 'p90', 'p95',
                              'pct_above_240', 'pct_below_50', 'pct_above_200', 'pct_below_100',
                              'range', 'cv', 'skewness', 'kurtosis']:
                    col_name = f'{feature}_{band}'
                    feature_columns.append(col_name)
                    df[col_name] = np.nan
            
            # Get all image files for this station
            station_path = self.cropped_data_path / state / station
            if not station_path.exists():
                logger.error(f"Station path not found: {station_path}")
                return False
            
            # Process all years
            image_files = []
            for year_dir in station_path.glob("*"):
                if year_dir.is_dir():
                    image_files.extend(year_dir.glob("*.tif"))
            
            logger.info(f"Found {len(image_files)} images for {state}_{station}")
            
            # Process each image
            features_extracted = 0
            for image_file in tqdm(image_files, desc=f"Processing {state}_{station}"):
                try:
                    # Parse filename
                    timestamp, band = self.parse_filename(image_file.name)
                    if timestamp is None or band is None:
                        continue
                    
                    # Extract features
                    features = self.extract_image_features(str(image_file), band)
                    
                    # Find matching rows in dataframe (within 30 minutes)
                    time_diff = abs(df['timestamp_utc'] - timestamp)
                    matching_rows = df[time_diff <= pd.Timedelta(minutes=30)]
                    
                    if len(matching_rows) > 0:
                        # Update features for matching rows
                        for idx in matching_rows.index:
                            for feature_name, feature_value in features.items():
                                if feature_name in df.columns:
                                    df.at[idx, feature_name] = feature_value
                        features_extracted += 1
                    
                except Exception as e:
                    logger.error(f"Error processing image {image_file}: {e}")
                    continue
            
            logger.info(f"Extracted features from {features_extracted} images for {state}_{station}")
            
            # Save enhanced dataset
            output_file = self.processed_data_path / f"{state}_{station}_enhanced.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"Saved enhanced dataset: {output_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing station {state}_{station}: {e}")
            return False
    
    def process_all_stations(self) -> bool:
        """Process all stations and create enhanced datasets."""
        try:
            success_count = 0
            total_count = 0
            
            # Get all states
            for state_dir in self.cropped_data_path.iterdir():
                if not state_dir.is_dir():
                    continue
                    
                state = state_dir.name
                logger.info(f"Processing state: {state}")
                
                # Get all stations in this state
                for station_dir in state_dir.iterdir():
                    if not station_dir.is_dir():
                        continue
                        
                    station = station_dir.name
                    total_count += 1
                    
                    if self.process_station(state, station):
                        success_count += 1
            
            logger.info(f"Successfully processed {success_count}/{total_count} stations")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error processing all stations: {e}")
            return False
    
    def create_unified_dataset(self) -> bool:
        """Merge all enhanced station datasets into a unified dataset."""
        try:
            logger.info("Creating unified enhanced dataset...")
            
            all_dataframes = []
            
            # Find all enhanced CSV files
            enhanced_files = list(self.processed_data_path.glob("*_enhanced.csv"))
            
            if not enhanced_files:
                logger.error("No enhanced CSV files found!")
                return False
            
            logger.info(f"Found {len(enhanced_files)} enhanced station files")
            
            # Load and combine all files
            for file_path in tqdm(enhanced_files, desc="Merging datasets"):
                try:
                    df = pd.read_csv(file_path)
                    all_dataframes.append(df)
                    logger.info(f"Loaded {len(df)} rows from {file_path.name}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
                    continue
            
            if not all_dataframes:
                logger.error("No dataframes loaded successfully!")
                return False
            
            # Combine all dataframes
            unified_df = pd.concat(all_dataframes, ignore_index=True)
            
            # Sort by timestamp
            unified_df['timestamp_utc'] = pd.to_datetime(unified_df['timestamp_utc'])
            unified_df = unified_df.sort_values(['state', 'station_id', 'timestamp_utc'])
            
            # Save unified dataset
            unified_df.to_csv(self.unified_dataset_path, index=False)
            
            logger.info(f"Created unified dataset with {len(unified_df)} rows")
            logger.info(f"Saved to: {self.unified_dataset_path}")
            
            # Print summary statistics
            logger.info("\nDataset Summary:")
            logger.info(f"Total rows: {len(unified_df)}")
            logger.info(f"Date range: {unified_df['timestamp_utc'].min()} to {unified_df['timestamp_utc'].max()}")
            logger.info(f"States: {unified_df['state'].unique()}")
            logger.info(f"Stations: {unified_df['station_id'].nunique()}")
            
            # Check feature completeness
            feature_cols = [col for col in unified_df.columns if any(band in col for band in ['TIR1', 'WV'])]
            logger.info(f"Feature columns: {len(feature_cols)}")
            
            for col in feature_cols[:5]:  # Show first 5 feature columns
                non_null_pct = (unified_df[col].notna().sum() / len(unified_df)) * 100
                logger.info(f"  {col}: {non_null_pct:.1f}% non-null")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating unified dataset: {e}")
            return False
    
    def run_full_pipeline(self) -> bool:
        """Run the complete feature extraction and dataset creation pipeline."""
        try:
            logger.info("Starting INSAT Feature Extraction Pipeline...")
            
            # Step 1: Process all stations
            if not self.process_all_stations():
                logger.error("Failed to process stations")
                return False
            
            # Step 2: Create unified dataset
            if not self.create_unified_dataset():
                logger.error("Failed to create unified dataset")
                return False
            
            logger.info("Pipeline completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return False


def main():
    """Main execution function."""
    try:
        # Initialize the feature extractor
        base_path = r"c:\Users\Lalit Hire\OneDrive\Desktop\APE_07"
        extractor = INSATFeatureExtractor(base_path)
        
        # Run the full pipeline
        success = extractor.run_full_pipeline()
        
        if success:
            print("\n✅ Feature extraction pipeline completed successfully!")
            print(f"📁 Enhanced datasets saved in: {extractor.processed_data_path}")
            print(f"📊 Unified dataset saved as: {extractor.unified_dataset_path}")
        else:
            print("\n❌ Pipeline failed. Check the logs for details.")
            
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        print(f"\n❌ Pipeline failed: {e}")


if __name__ == "__main__":
    main()
