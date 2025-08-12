"""
Data aligner for AEP 3.0 pipeline.

This module handles alignment of satellite image data with CPCB station data
based on timestamps, accounting for timezone differences.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from tqdm import tqdm

# Import utility functions
from .utils import (
    parse_insat_filename, convert_ist_to_utc, convert_utc_to_ist,
    find_nearest_timestamp, extract_satellite_features
)


class DataAligner:
    """
    Aligner for satellite and CPCB data.
    
    This class handles:
    - Timezone conversion between IST and UTC
    - Matching satellite images to CPCB data timestamps
    - Feature extraction from satellite images
    - Creating aligned datasets
    """
    
    def __init__(self, max_hours_diff: int = 1, feature_type: str = "statistics"):
        """
        Initialize the data aligner.
        
        Args:
            max_hours_diff: Maximum allowed difference in hours for timestamp matching
            feature_type: Type of satellite features to extract
        """
        self.max_hours_diff = max_hours_diff
        self.feature_type = feature_type
        self.logger = logging.getLogger(__name__)
    
    def align_data(self, cpcb_data: pd.DataFrame, cropped_images: List[str], 
                   station_info: Dict[str, Any]) -> pd.DataFrame:
        """
        Align CPCB data with satellite images.
        
        Args:
            cpcb_data: CPCB station data DataFrame
            cropped_images: List of paths to cropped satellite images
            station_info: Station metadata dictionary
            
        Returns:
            DataFrame with aligned CPCB and satellite data
        """
        self.logger.info(f"Aligning data for station: {station_info['station_name']}")
        
        # Step 1: Prepare CPCB data
        cpcb_processed = self._prepare_cpcb_data(cpcb_data)
        
        # Step 2: Prepare satellite data
        satellite_processed = self._prepare_satellite_data(cropped_images)
        
        # Step 3: Align timestamps
        aligned_data = self._align_timestamps(cpcb_processed, satellite_processed)
        
        # Step 4: Extract satellite features
        final_data = self._extract_satellite_features(aligned_data)
        
        self.logger.info(f"Successfully aligned {len(final_data)} records for {station_info['station_name']}")
        return final_data
    
    def _prepare_cpcb_data(self, cpcb_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare CPCB data for alignment.
        
        Args:
            cpcb_data: Raw CPCB data DataFrame
            
        Returns:
            Processed CPCB data DataFrame
        """
        # Create a copy to avoid modifying original data
        cpcb_processed = cpcb_data.copy()
        
        # Convert timestamp column to datetime
        timestamp_col = 'From Date' if 'From Date' in cpcb_processed.columns else cpcb_processed.columns[0]
        cpcb_processed['timestamp'] = pd.to_datetime(cpcb_processed[timestamp_col])
        
        # Convert IST timestamps to UTC
        cpcb_processed['timestamp_utc'] = cpcb_processed['timestamp'].apply(convert_ist_to_utc)
        
        # Round timestamps to nearest hour for better matching
        cpcb_processed['timestamp_hour'] = cpcb_processed['timestamp_utc'].dt.floor('H')
        
        # Remove rows with invalid timestamps
        cpcb_processed = cpcb_processed.dropna(subset=['timestamp_utc'])
        
        # Sort by timestamp
        cpcb_processed = cpcb_processed.sort_values('timestamp_utc')
        
        return cpcb_processed
    
    def _prepare_satellite_data(self, cropped_images: List[str]) -> pd.DataFrame:
        """
        Prepare satellite data for alignment.
        
        Args:
            cropped_images: List of paths to cropped satellite images
            
        Returns:
            DataFrame with satellite image information
        """
        satellite_data = []
        
        for image_path in cropped_images:
            try:
                # Parse filename to extract timestamp
                filename = Path(image_path).name
                filename_info = parse_insat_filename(filename)
                
                if filename_info and 'datetime' in filename_info:
                    satellite_data.append({
                        'image_path': image_path,
                        'timestamp_utc': filename_info['datetime'],
                        'timestamp_hour': filename_info['datetime'].replace(minute=0, second=0, microsecond=0),
                        'band': filename_info.get('band', 'unknown'),
                        'year': filename_info.get('year'),
                        'month': filename_info.get('month'),
                        'day': filename_info.get('day'),
                        'hour': filename_info.get('hour'),
                        'minute': filename_info.get('minute')
                    })
            except Exception as e:
                self.logger.debug(f"Error processing satellite image {image_path}: {e}")
        
        if not satellite_data:
            self.logger.warning("No valid satellite images found")
            return pd.DataFrame()
        
        # Create DataFrame and sort by timestamp
        satellite_df = pd.DataFrame(satellite_data)
        satellite_df = satellite_df.sort_values('timestamp_utc')
        
        return satellite_df
    
    def _align_timestamps(self, cpcb_data: pd.DataFrame, 
                         satellite_data: pd.DataFrame) -> pd.DataFrame:
        """
        Align CPCB and satellite data based on timestamps.
        
        Args:
            cpcb_data: Processed CPCB data
            satellite_data: Processed satellite data
            
        Returns:
            DataFrame with aligned data
        """
        if satellite_data.empty:
            self.logger.warning("No satellite data available for alignment")
            return cpcb_data
        
        # Get unique satellite timestamps
        satellite_timestamps = satellite_data['timestamp_hour'].unique()
        
        # Find matching satellite images for each CPCB timestamp
        aligned_data = []
        
        for _, cpcb_row in tqdm(cpcb_data.iterrows(), 
                               total=len(cpcb_data), 
                               desc="Aligning timestamps"):
            
            cpcb_time = cpcb_row['timestamp_hour']
            
            # Find nearest satellite timestamp
            nearest_sat_time = find_nearest_timestamp(
                cpcb_time, satellite_timestamps, self.max_hours_diff
            )
            
            if nearest_sat_time is not None:
                # Get satellite images for this timestamp
                matching_satellites = satellite_data[
                    satellite_data['timestamp_hour'] == nearest_sat_time
                ]
                
                # Create aligned row
                aligned_row = cpcb_row.to_dict()
                aligned_row['satellite_images'] = matching_satellites['image_path'].tolist()
                aligned_row['satellite_timestamp'] = nearest_sat_time
                aligned_row['time_diff_hours'] = abs(
                    (cpcb_time - nearest_sat_time).total_seconds() / 3600
                )
                
                aligned_data.append(aligned_row)
        
        if not aligned_data:
            self.logger.warning("No aligned data found")
            return pd.DataFrame()
        
        return pd.DataFrame(aligned_data)
    
    def _extract_satellite_features(self, aligned_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from satellite images.
        
        Args:
            aligned_data: DataFrame with aligned CPCB and satellite data
            
        Returns:
            DataFrame with extracted satellite features
        """
        if aligned_data.empty:
            return aligned_data
        
        # Extract features for each row
        feature_data = []
        
        for _, row in tqdm(aligned_data.iterrows(), 
                          total=len(aligned_data), 
                          desc="Extracting satellite features"):
            
            # Get CPCB features (exclude satellite-related columns)
            cpcb_features = {}
            for col in row.index:
                if not col.startswith('satellite_') and col != 'satellite_images':
                    cpcb_features[col] = row[col]
            
            # Extract features from satellite images
            satellite_features = {}
            if 'satellite_images' in row and row['satellite_images']:
                for image_path in row['satellite_images']:
                    features = extract_satellite_features(image_path, self.feature_type)
                    # Add image identifier to feature names
                    image_id = Path(image_path).stem
                    for feature_name, feature_value in features.items():
                        satellite_features[f"{image_id}_{feature_name}"] = feature_value
            
            # Combine CPCB and satellite features
            combined_features = {**cpcb_features, **satellite_features}
            feature_data.append(combined_features)
        
        if not feature_data:
            return aligned_data
        
        # Create final DataFrame
        final_data = pd.DataFrame(feature_data)
        
        # Add metadata columns
        if 'satellite_timestamp' in aligned_data.columns:
            final_data['satellite_timestamp'] = aligned_data['satellite_timestamp']
        if 'time_diff_hours' in aligned_data.columns:
            final_data['time_diff_hours'] = aligned_data['time_diff_hours']
        
        return final_data
    
    def get_alignment_statistics(self, aligned_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about the alignment process.
        
        Args:
            aligned_data: Aligned data DataFrame
            
        Returns:
            Dictionary containing alignment statistics
        """
        if aligned_data.empty:
            return {'error': 'No aligned data available'}
        
        stats = {
            'total_records': len(aligned_data),
            'records_with_satellite': 0,
            'records_without_satellite': 0,
            'time_differences': [],
            'satellite_image_counts': []
        }
        
        for _, row in aligned_data.iterrows():
            if 'satellite_images' in row and row['satellite_images']:
                stats['records_with_satellite'] += 1
                stats['satellite_image_counts'].append(len(row['satellite_images']))
                
                if 'time_diff_hours' in row:
                    stats['time_differences'].append(row['time_diff_hours'])
            else:
                stats['records_without_satellite'] += 1
        
        # Calculate additional statistics
        if stats['time_differences']:
            stats['avg_time_diff'] = np.mean(stats['time_differences'])
            stats['max_time_diff'] = np.max(stats['time_differences'])
            stats['min_time_diff'] = np.min(stats['time_differences'])
        
        if stats['satellite_image_counts']:
            stats['avg_satellite_images'] = np.mean(stats['satellite_image_counts'])
            stats['max_satellite_images'] = np.max(stats['satellite_image_counts'])
            stats['min_satellite_images'] = np.min(stats['satellite_image_counts'])
        
        return stats
    
    def validate_aligned_data(self, aligned_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate aligned data quality.
        
        Args:
            aligned_data: Aligned data DataFrame
            
        Returns:
            Dictionary containing validation results
        """
        if aligned_data.empty:
            return {'error': 'No data to validate'}
        
        validation = {
            'total_records': len(aligned_data),
            'missing_values': {},
            'data_types': {},
            'satellite_coverage': 0,
            'timestamp_coverage': 0
        }
        
        # Check missing values
        for column in aligned_data.columns:
            missing_count = aligned_data[column].isnull().sum()
            missing_pct = (missing_count / len(aligned_data)) * 100
            validation['missing_values'][column] = {
                'count': int(missing_count),
                'percentage': float(missing_pct)
            }
        
        # Check data types
        for column in aligned_data.columns:
            validation['data_types'][column] = str(aligned_data[column].dtype)
        
        # Check satellite coverage
        if 'satellite_images' in aligned_data.columns:
            satellite_coverage = aligned_data['satellite_images'].apply(
                lambda x: len(x) > 0 if isinstance(x, list) else False
            ).sum()
            validation['satellite_coverage'] = int(satellite_coverage)
        
        # Check timestamp coverage
        if 'timestamp_utc' in aligned_data.columns:
            timestamp_coverage = aligned_data['timestamp_utc'].notna().sum()
            validation['timestamp_coverage'] = int(timestamp_coverage)
        
        return validation 