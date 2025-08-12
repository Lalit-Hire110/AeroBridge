"""
Utility functions for AEP 3.0 pipeline.

This module contains common utility functions used across the pipeline.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np


def setup_logging(level: str = "INFO") -> None:
    """
    Setup logging configuration for the pipeline.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('aep_pipeline.log'),
            logging.StreamHandler()
        ]
    )


def create_directories(directories: List[Path]) -> None:
    """
    Create directories if they don't exist.
    
    Args:
        directories: List of directory paths to create
    """
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_state_list(raw_dir: Path) -> List[str]:
    """
    Get list of available states from the raw data directory.
    
    Args:
        raw_dir: Path to raw data directory
        
    Returns:
        List of state names
    """
    if not raw_dir.exists():
        return []
    
    states = [d.name for d in raw_dir.iterdir() if d.is_dir()]
    return sorted(states)


def parse_insat_filename(filename: str) -> Dict[str, Any]:
    """
    Parse INSAT filename to extract date and time information.
    
    Args:
        filename: INSAT filename (e.g., "3RIMG_01NOV2021_0815_L1B_STD_V01R00_IMG_WV.tif")
        
    Returns:
        Dictionary containing parsed information
    """
    try:
        # Extract date and time from filename
        parts = filename.split('_')
        if len(parts) >= 3:
            date_str = parts[1]  # "01NOV2021"
            time_str = parts[2]  # "0815"
            
            # Parse date
            day = int(date_str[:2])
            month_str = date_str[2:5]
            year = int(date_str[5:])
            
            # Parse time
            hour = int(time_str[:2])
            minute = int(time_str[2:])
            
            # Create datetime object (UTC)
            month_map = {
                'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
            }
            
            month = month_map.get(month_str.upper(), 1)
            dt = datetime(year, month, day, hour, minute, tzinfo=pytz.UTC)
            
            return {
                'datetime': dt,
                'date': dt.date(),
                'time': dt.time(),
                'year': year,
                'month': month,
                'day': day,
                'hour': hour,
                'minute': minute,
                'band': parts[-1].split('.')[0] if parts else 'unknown'
            }
    except Exception as e:
        logging.warning(f"Could not parse filename {filename}: {e}")
        return {}
    
    return {}


def convert_ist_to_utc(ist_datetime: datetime) -> datetime:
    """
    Convert IST datetime to UTC.
    
    Args:
        ist_datetime: Datetime object in IST
        
    Returns:
        Datetime object in UTC
    """
    ist_tz = pytz.timezone('Asia/Kolkata')
    utc_tz = pytz.UTC
    
    # If datetime is naive, assume it's in IST
    if ist_datetime.tzinfo is None:
        ist_datetime = ist_tz.localize(ist_datetime)
    
    return ist_datetime.astimezone(utc_tz)


def convert_utc_to_ist(utc_datetime: datetime) -> datetime:
    """
    Convert UTC datetime to IST.
    
    Args:
        utc_datetime: Datetime object in UTC
        
    Returns:
        Datetime object in IST
    """
    utc_tz = pytz.UTC
    ist_tz = pytz.timezone('Asia/Kolkata')
    
    # If datetime is naive, assume it's in UTC
    if utc_datetime.tzinfo is None:
        utc_datetime = utc_tz.localize(utc_datetime)
    
    return utc_datetime.astimezone(ist_tz)


def find_nearest_timestamp(target_time: datetime, available_times: List[datetime], 
                          max_hours_diff: int = 1) -> Optional[datetime]:
    """
    Find the nearest timestamp from a list of available times.
    
    Args:
        target_time: Target timestamp to match
        available_times: List of available timestamps
        max_hours_diff: Maximum allowed difference in hours
        
    Returns:
        Nearest timestamp or None if no match within threshold
    """
    if not available_times:
        return None
    
    min_diff = float('inf')
    nearest_time = None
    
    for time in available_times:
        diff = abs((target_time - time).total_seconds() / 3600)  # hours
        if diff < min_diff and diff <= max_hours_diff:
            min_diff = diff
            nearest_time = time
    
    return nearest_time


def extract_satellite_features(image_path: str, feature_type: str = "statistics") -> Dict[str, float]:
    """
    Extract features from satellite image.
    
    Args:
        image_path: Path to satellite image file
        feature_type: Type of features to extract ("statistics", "histogram", "texture")
        
    Returns:
        Dictionary of extracted features
    """
    try:
        import rasterio
        import numpy as np
        from skimage.feature import graycomatrix, graycoprops
        
        with rasterio.open(image_path) as src:
            # Read all bands
            data = src.read()
            
            features = {}
            
            if feature_type == "statistics":
                # Basic statistical features
                for band_idx in range(data.shape[0]):
                    band_data = data[band_idx]
                    band_data = band_data[band_data != src.nodata] if src.nodata else band_data
                    
                    if len(band_data) > 0:
                        features[f'band_{band_idx}_mean'] = float(np.mean(band_data))
                        features[f'band_{band_idx}_std'] = float(np.std(band_data))
                        features[f'band_{band_idx}_min'] = float(np.min(band_data))
                        features[f'band_{band_idx}_max'] = float(np.max(band_data))
                        features[f'band_{band_idx}_median'] = float(np.median(band_data))
                        features[f'band_{band_idx}_q25'] = float(np.percentile(band_data, 25))
                        features[f'band_{band_idx}_q75'] = float(np.percentile(band_data, 75))
            
            elif feature_type == "histogram":
                # Histogram-based features
                for band_idx in range(data.shape[0]):
                    band_data = data[band_idx]
                    band_data = band_data[band_data != src.nodata] if src.nodata else band_data
                    
                    if len(band_data) > 0:
                        hist, bins = np.histogram(band_data, bins=10)
                        for i, count in enumerate(hist):
                            features[f'band_{band_idx}_hist_{i}'] = float(count)
            
            elif feature_type == "texture":
                # Texture features using GLCM
                for band_idx in range(data.shape[0]):
                    band_data = data[band_idx]
                    band_data = band_data[band_data != src.nodata] if src.nodata else band_data
                    
                    if len(band_data) > 0:
                        # Normalize to 0-255 for GLCM
                        band_norm = ((band_data - band_data.min()) / 
                                   (band_data.max() - band_data.min()) * 255).astype(np.uint8)
                        
                        # Calculate GLCM
                        glcm = graycomatrix(band_norm, [1], [0], levels=256, symmetric=True, normed=True)
                        
                        # Extract texture properties
                        features[f'band_{band_idx}_contrast'] = float(graycoprops(glcm, 'contrast')[0, 0])
                        features[f'band_{band_idx}_homogeneity'] = float(graycoprops(glcm, 'homogeneity')[0, 0])
                        features[f'band_{band_idx}_energy'] = float(graycoprops(glcm, 'energy')[0, 0])
                        features[f'band_{band_idx}_correlation'] = float(graycoprops(glcm, 'correlation')[0, 0])
            
            return features
            
    except ImportError:
        logging.warning("rasterio or skimage not available. Using basic numpy features.")
        # Fallback to basic numpy operations
        try:
            import numpy as np
            data = np.load(image_path) if image_path.endswith('.npy') else None
            if data is not None:
                return {
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data)),
                    'min': float(np.min(data)),
                    'max': float(np.max(data))
                }
        except:
            pass
        
        return {}
    
    except Exception as e:
        logging.error(f"Error extracting features from {image_path}: {e}")
        return {}


def validate_data_quality(cpcb_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate CPCB data quality and return statistics.
    
    Args:
        cpcb_data: CPCB station data DataFrame
        
    Returns:
        Dictionary containing data quality statistics
    """
    stats = {
        'total_records': len(cpcb_data),
        'missing_values': {},
        'data_types': {},
        'value_ranges': {}
    }
    
    # Check missing values
    for column in cpcb_data.columns:
        missing_count = cpcb_data[column].isnull().sum()
        missing_pct = (missing_count / len(cpcb_data)) * 100
        stats['missing_values'][column] = {
            'count': int(missing_count),
            'percentage': float(missing_pct)
        }
    
    # Check data types
    for column in cpcb_data.columns:
        stats['data_types'][column] = str(cpcb_data[column].dtype)
    
    # Check value ranges for numeric columns
    numeric_columns = cpcb_data.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        non_null_data = cpcb_data[column].dropna()
        if len(non_null_data) > 0:
            stats['value_ranges'][column] = {
                'min': float(non_null_data.min()),
                'max': float(non_null_data.max()),
                'mean': float(non_null_data.mean()),
                'std': float(non_null_data.std())
            }
    
    return stats


def save_processing_summary(summary: Dict[str, Any], output_path: Path) -> None:
    """
    Save processing summary to a JSON file.
    
    Args:
        summary: Processing summary dictionary
        output_path: Path to save the summary file
    """
    import json
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str) 