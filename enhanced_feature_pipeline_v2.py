"""
Enhanced INSAT Image Feature Extraction Pipeline v2.0
====================================================

This improved pipeline addresses low feature completeness by:
- Using per-band (TIR1/WV) nearest timestamp matching (±15 min)
- Preserving ALL CPCB rows (non-lossy approach)
- Adding indicator columns for image presence
- Processing only new/unprocessed images
- Using merge_asof for efficient temporal matching

Author: AEP 3.0 Pipeline Enhanced
Date: 2025-08-05
"""

import os
import pandas as pd
import numpy as np
import rasterio
from datetime import datetime, timezone, timedelta
import pytz
import logging
from pathlib import Path
import glob
from typing import Dict, List, Tuple, Optional, Set
import re
from tqdm import tqdm
import pickle
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_feature_extraction_v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedINSATFeatureExtractor:
    """Enhanced feature extractor with non-lossy per-band matching."""
    
    def __init__(self, base_path: str):
        """
        Initialize the enhanced feature extractor.
        
        Args:
            base_path: Path to the APE_07 project directory
        """
        self.base_path = Path(base_path)
        self.cropped_data_path = self.base_path / "data" / "cropped_data"
        self.processed_data_path = self.base_path / "data" / "processed_data"
        self.cache_path = self.base_path / "data" / "feature_cache"
        self.unified_dataset_path = self.base_path / "data" / "unified_dataset_enhanced_v2.csv"
        
        # Ensure directories exist
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        # IST timezone for timestamp conversion
        self.ist_tz = pytz.timezone('Asia/Kolkata')
        self.utc_tz = pytz.UTC
        
        # Feature cache for avoiding recomputation
        self.feature_cache = {}
        self.load_feature_cache()
        
        # Matching tolerance (±15 minutes)
        self.time_tolerance = pd.Timedelta(minutes=15)
        
    def load_feature_cache(self):
        """Load previously computed features from cache."""
        cache_file = self.cache_path / "feature_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.feature_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.feature_cache)} cached features")
            except Exception as e:
                logger.warning(f"Could not load feature cache: {e}")
                self.feature_cache = {}
        else:
            self.feature_cache = {}
    
    def save_feature_cache(self):
        """Save computed features to cache."""
        cache_file = self.cache_path / "feature_cache.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.feature_cache, f)
            logger.info(f"Saved {len(self.feature_cache)} features to cache")
        except Exception as e:
            logger.warning(f"Could not save feature cache: {e}")
    
    def get_image_hash(self, image_path: str) -> str:
        """Generate a hash for the image file to use as cache key."""
        try:
            # Use file path and modification time for hash
            stat = os.stat(image_path)
            hash_input = f"{image_path}_{stat.st_mtime}_{stat.st_size}"
            return hashlib.md5(hash_input.encode()).hexdigest()
        except:
            return hashlib.md5(image_path.encode()).hexdigest()
    
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
                return None, None
                
            date_str, time_str = match.groups()
            
            # Extract band (TIR1 or WV)
            band_match = re.search(r'IMG_([A-Z0-9]+)_cropped', filename)
            if not band_match:
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
            logger.debug(f"Error parsing filename {filename}: {e}")
            return None, None
    
    def extract_image_features(self, image_path: str, band: str) -> Dict[str, float]:
        """
        Extract comprehensive features from a single TIFF image with caching.
        
        Args:
            image_path: Path to the TIFF image
            band: Band name (TIR1 or WV) for feature naming
            
        Returns:
            Dictionary of features with band-specific names
        """
        # Check cache first
        image_hash = self.get_image_hash(image_path)
        cache_key = f"{image_hash}_{band}"
        
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
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
                    features = self._get_empty_features(band)
                else:
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
                
                # Cache the features
                self.feature_cache[cache_key] = features
                return features
                
        except Exception as e:
            logger.debug(f"Error extracting features from {image_path}: {e}")
            features = self._get_empty_features(band)
            self.feature_cache[cache_key] = features
            return features
    
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
    
    def get_all_feature_columns(self) -> List[str]:
        """Get all possible feature column names."""
        feature_names = ['mean', 'std', 'min', 'max', 'median', 'p25', 'p75', 'p90', 'p95',
                        'pct_above_240', 'pct_below_50', 'pct_above_200', 'pct_below_100',
                        'range', 'cv', 'skewness', 'kurtosis']
        bands = ['TIR1', 'WV']
        
        columns = []
        for band in bands:
            for feature in feature_names:
                columns.append(f'{feature}_{band}')
        
        return columns
    
    def create_image_metadata_df(self, state: str, station: str) -> pd.DataFrame:
        """
        Create a DataFrame with all available images and their metadata.
        
        Args:
            state: State name
            station: Station ID
            
        Returns:
            DataFrame with columns: image_path, timestamp_utc, band
        """
        station_path = self.cropped_data_path / state / station
        if not station_path.exists():
            return pd.DataFrame(columns=['image_path', 'timestamp_utc', 'band'])
        
        image_metadata = []
        
        # Process all years
        for year_dir in station_path.glob("*"):
            if year_dir.is_dir():
                for image_file in year_dir.glob("*.tif"):
                    timestamp, band = self.parse_filename(image_file.name)
                    if timestamp is not None and band is not None:
                        image_metadata.append({
                            'image_path': str(image_file),
                            'timestamp_utc': timestamp,
                            'band': band
                        })
        
        df = pd.DataFrame(image_metadata)
        if len(df) > 0:
            df = df.sort_values('timestamp_utc').reset_index(drop=True)
        
        return df
    
    def match_images_to_cpcb(self, cpcb_df: pd.DataFrame, image_df: pd.DataFrame, band: str) -> pd.DataFrame:
        """
        Match images of a specific band to CPCB data using nearest timestamp within tolerance.
        
        Args:
            cpcb_df: CPCB dataframe with timestamp_utc column
            image_df: Image metadata dataframe filtered for specific band
            band: Band name (TIR1 or WV)
            
        Returns:
            CPCB dataframe with matched image paths and features
        """
        if len(image_df) == 0:
            logger.warning(f"No {band} images found for matching")
            return cpcb_df
        
        # Ensure timestamps are datetime
        cpcb_df = cpcb_df.copy()
        cpcb_df['timestamp_utc'] = pd.to_datetime(cpcb_df['timestamp_utc'])
        image_df = image_df.copy()
        image_df['timestamp_utc'] = pd.to_datetime(image_df['timestamp_utc'])
        
        # Sort both dataframes by timestamp for merge_asof
        cpcb_df = cpcb_df.sort_values('timestamp_utc')
        image_df = image_df.sort_values('timestamp_utc')
        
        # Use merge_asof for nearest timestamp matching
        merged = pd.merge_asof(
            cpcb_df,
            image_df[['timestamp_utc', 'image_path']],
            on='timestamp_utc',
            direction='nearest',
            tolerance=self.time_tolerance,
            suffixes=('', f'_{band}')
        )
        
        # Rename the matched image path column
        if 'image_path' in merged.columns:
            merged = merged.rename(columns={'image_path': f'image_path_{band}'})
        
        return merged
    
    def process_station_enhanced(self, state: str, station: str) -> bool:
        """
        Process a single station with enhanced per-band matching.
        
        Args:
            state: State name
            station: Station ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Processing station {state}_{station} (Enhanced)")
            
            # Load existing merged CPCB data
            merged_file = self.processed_data_path / f"{state}_{station}_merged.csv"
            if not merged_file.exists():
                logger.error(f"Merged file not found: {merged_file}")
                return False
            
            cpcb_df = pd.read_csv(merged_file)
            cpcb_df['timestamp_utc'] = pd.to_datetime(cpcb_df['timestamp_utc'])
            
            logger.info(f"Loaded {len(cpcb_df)} CPCB records for {state}_{station}")
            
            # Create image metadata DataFrame
            image_df = self.create_image_metadata_df(state, station)
            logger.info(f"Found {len(image_df)} total images for {state}_{station}")
            
            if len(image_df) == 0:
                logger.warning(f"No images found for {state}_{station}")
                # Still save the CPCB data with empty feature columns
                enhanced_df = self._add_empty_feature_columns(cpcb_df)
                output_file = self.processed_data_path / f"{state}_{station}_enhanced_v2.csv"
                enhanced_df.to_csv(output_file, index=False)
                return True
            
            # Separate images by band
            tir1_images = image_df[image_df['band'] == 'TIR1'].copy()
            wv_images = image_df[image_df['band'] == 'WV'].copy()
            
            logger.info(f"TIR1 images: {len(tir1_images)}, WV images: {len(wv_images)}")
            
            # Match TIR1 images
            enhanced_df = self.match_images_to_cpcb(cpcb_df, tir1_images, 'TIR1')
            
            # Match WV images
            enhanced_df = self.match_images_to_cpcb(enhanced_df, wv_images, 'WV')
            
            # Initialize all feature columns with NaN
            feature_columns = self.get_all_feature_columns()
            for col in feature_columns:
                enhanced_df[col] = np.nan
            
            # Add indicator columns
            enhanced_df['has_TIR1'] = enhanced_df.get('image_path_TIR1', pd.Series()).notna()
            enhanced_df['has_WV'] = enhanced_df.get('image_path_WV', pd.Series()).notna()
            
            # Extract features for matched images
            features_extracted = {'TIR1': 0, 'WV': 0}
            
            # Process TIR1 features
            if 'image_path_TIR1' in enhanced_df.columns:
                tir1_matches = enhanced_df[enhanced_df['image_path_TIR1'].notna()]
                for idx, row in tqdm(tir1_matches.iterrows(), 
                                   desc=f"Extracting TIR1 features for {state}_{station}",
                                   total=len(tir1_matches)):
                    try:
                        features = self.extract_image_features(row['image_path_TIR1'], 'TIR1')
                        for feature_name, feature_value in features.items():
                            enhanced_df.at[idx, feature_name] = feature_value
                        features_extracted['TIR1'] += 1
                    except Exception as e:
                        logger.debug(f"Error extracting TIR1 features for row {idx}: {e}")
            
            # Process WV features
            if 'image_path_WV' in enhanced_df.columns:
                wv_matches = enhanced_df[enhanced_df['image_path_WV'].notna()]
                for idx, row in tqdm(wv_matches.iterrows(), 
                                   desc=f"Extracting WV features for {state}_{station}",
                                   total=len(wv_matches)):
                    try:
                        features = self.extract_image_features(row['image_path_WV'], 'WV')
                        for feature_name, feature_value in features.items():
                            enhanced_df.at[idx, feature_name] = feature_value
                        features_extracted['WV'] += 1
                    except Exception as e:
                        logger.debug(f"Error extracting WV features for row {idx}: {e}")
            
            # Remove image path columns (keep only features)
            columns_to_drop = [col for col in enhanced_df.columns if col.startswith('image_path_')]
            enhanced_df = enhanced_df.drop(columns=columns_to_drop)
            
            # Save enhanced dataset
            output_file = self.processed_data_path / f"{state}_{station}_enhanced_v2.csv"
            enhanced_df.to_csv(output_file, index=False)
            
            # Log statistics
            total_rows = len(enhanced_df)
            tir1_coverage = enhanced_df['has_TIR1'].sum()
            wv_coverage = enhanced_df['has_WV'].sum()
            both_coverage = (enhanced_df['has_TIR1'] & enhanced_df['has_WV']).sum()
            
            logger.info(f"Station {state}_{station} processed:")
            logger.info(f"  Total CPCB rows: {total_rows}")
            logger.info(f"  TIR1 coverage: {tir1_coverage} ({tir1_coverage/total_rows*100:.1f}%)")
            logger.info(f"  WV coverage: {wv_coverage} ({wv_coverage/total_rows*100:.1f}%)")
            logger.info(f"  Both bands: {both_coverage} ({both_coverage/total_rows*100:.1f}%)")
            logger.info(f"  Features extracted: TIR1={features_extracted['TIR1']}, WV={features_extracted['WV']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing station {state}_{station}: {e}")
            return False
    
    def _add_empty_feature_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add empty feature columns to dataframe."""
        df = df.copy()
        feature_columns = self.get_all_feature_columns()
        for col in feature_columns:
            df[col] = np.nan
        df['has_TIR1'] = False
        df['has_WV'] = False
        return df
    
    def process_all_stations_enhanced(self) -> bool:
        """Process all stations with enhanced matching."""
        try:
            success_count = 0
            total_count = 0
            
            # Get all merged CSV files
            merged_files = list(self.processed_data_path.glob("*_merged.csv"))
            logger.info(f"Found {len(merged_files)} station files to process")
            
            for merged_file in merged_files:
                # Parse state and station from filename
                filename = merged_file.stem  # Remove .csv extension
                if filename.endswith('_merged'):
                    parts = filename[:-7].split('_')  # Remove '_merged' and split
                    if len(parts) >= 2:
                        state = parts[0]
                        station = '_'.join(parts[1:])
                        
                        total_count += 1
                        if self.process_station_enhanced(state, station):
                            success_count += 1
                        
                        # Save cache periodically
                        if total_count % 5 == 0:
                            self.save_feature_cache()
            
            # Final cache save
            self.save_feature_cache()
            
            logger.info(f"Successfully processed {success_count}/{total_count} stations")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error processing all stations: {e}")
            return False
    
    def create_unified_dataset_v2(self) -> bool:
        """Create unified dataset from all enhanced station files."""
        try:
            logger.info("Creating unified enhanced dataset v2...")
            
            all_dataframes = []
            
            # Find all enhanced v2 CSV files
            enhanced_files = list(self.processed_data_path.glob("*_enhanced_v2.csv"))
            
            if not enhanced_files:
                logger.error("No enhanced v2 CSV files found!")
                return False
            
            logger.info(f"Found {len(enhanced_files)} enhanced station files")
            
            # Load and combine all files
            for file_path in tqdm(enhanced_files, desc="Merging enhanced datasets"):
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
            
            # Calculate and log comprehensive statistics
            total_rows = len(unified_df)
            tir1_coverage = unified_df['has_TIR1'].sum()
            wv_coverage = unified_df['has_WV'].sum()
            both_coverage = (unified_df['has_TIR1'] & unified_df['has_WV']).sum()
            either_coverage = (unified_df['has_TIR1'] | unified_df['has_WV']).sum()
            
            logger.info(f"\n=== UNIFIED DATASET V2 STATISTICS ===")
            logger.info(f"Total CPCB rows: {total_rows}")
            logger.info(f"Date range: {unified_df['timestamp_utc'].min()} to {unified_df['timestamp_utc'].max()}")
            logger.info(f"States: {list(unified_df['state'].unique())}")
            logger.info(f"Stations: {unified_df['station_id'].nunique()}")
            logger.info(f"\nImage Feature Coverage:")
            logger.info(f"  TIR1 only: {tir1_coverage} ({tir1_coverage/total_rows*100:.1f}%)")
            logger.info(f"  WV only: {wv_coverage} ({wv_coverage/total_rows*100:.1f}%)")
            logger.info(f"  Both bands: {both_coverage} ({both_coverage/total_rows*100:.1f}%)")
            logger.info(f"  Either band: {either_coverage} ({either_coverage/total_rows*100:.1f}%)")
            logger.info(f"  No images: {total_rows - either_coverage} ({(total_rows - either_coverage)/total_rows*100:.1f}%)")
            
            # Feature completeness by band
            feature_columns = self.get_all_feature_columns()
            tir1_features = [col for col in feature_columns if col.endswith('_TIR1')]
            wv_features = [col for col in feature_columns if col.endswith('_WV')]
            
            logger.info(f"\nFeature Completeness:")
            for feature_set, name in [(tir1_features, 'TIR1'), (wv_features, 'WV')]:
                if feature_set:
                    sample_col = feature_set[0]
                    non_null_count = unified_df[sample_col].notna().sum()
                    logger.info(f"  {name} features: {non_null_count} ({non_null_count/total_rows*100:.1f}%)")
            
            logger.info(f"\nDataset saved to: {self.unified_dataset_path}")
            logger.info(f"File size: {self.unified_dataset_path.stat().st_size / (1024*1024):.1f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating unified dataset v2: {e}")
            return False
    
    def run_enhanced_pipeline(self) -> bool:
        """Run the complete enhanced pipeline."""
        try:
            logger.info("Starting Enhanced INSAT Feature Extraction Pipeline v2...")
            
            # Step 1: Process all stations with enhanced matching
            if not self.process_all_stations_enhanced():
                logger.error("Failed to process stations")
                return False
            
            # Step 2: Create unified dataset
            if not self.create_unified_dataset_v2():
                logger.error("Failed to create unified dataset")
                return False
            
            logger.info("Enhanced pipeline v2 completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Enhanced pipeline failed: {e}")
            return False


def main():
    """Main execution function."""
    try:
        # Initialize the enhanced feature extractor
        base_path = r"c:\Users\Lalit Hire\OneDrive\Desktop\APE_07"
        extractor = EnhancedINSATFeatureExtractor(base_path)
        
        # Run the enhanced pipeline
        success = extractor.run_enhanced_pipeline()
        
        if success:
            print("\n✅ Enhanced feature extraction pipeline v2 completed successfully!")
            print(f"📁 Enhanced datasets saved in: {extractor.processed_data_path}")
            print(f"📊 Unified dataset saved as: {extractor.unified_dataset_path}")
            print("\n🎯 Key Improvements:")
            print("  • Non-lossy: ALL CPCB rows preserved")
            print("  • Per-band matching: TIR1 and WV processed separately")
            print("  • Nearest timestamp matching (±15 min tolerance)")
            print("  • Indicator columns: has_TIR1, has_WV")
            print("  • Feature caching for efficiency")
        else:
            print("\n❌ Enhanced pipeline failed. Check the logs for details.")
            
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        print(f"\n❌ Enhanced pipeline failed: {e}")


if __name__ == "__main__":
    main()
