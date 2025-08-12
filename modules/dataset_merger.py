"""
Dataset merger for AEP 3.0 pipeline.

This module handles merging of multiple station datasets into a unified dataset
suitable for machine learning training.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import json


class DatasetMerger:
    """
    Merger for multiple station datasets.
    
    This class handles:
    - Loading multiple station datasets
    - Standardizing column names and data types
    - Merging datasets with proper station identification
    - Creating unified datasets for ML training
    """
    
    def __init__(self):
        """Initialize the dataset merger."""
        self.logger = logging.getLogger(__name__)
    
    def merge_datasets(self, dataset_paths: List[str], output_path: Path) -> pd.DataFrame:
        """
        Merge multiple station datasets into a unified dataset.
        
        Args:
            dataset_paths: List of paths to station datasets
            output_path: Path to save the unified dataset
            
        Returns:
            Unified DataFrame
        """
        self.logger.info(f"Merging {len(dataset_paths)} datasets")
        
        if not dataset_paths:
            self.logger.warning("No datasets provided for merging")
            return pd.DataFrame()
        
        # Load and process each dataset
        datasets = []
        station_info = {}
        
        for dataset_path in dataset_paths:
            try:
                dataset, station_name = self._load_and_process_dataset(dataset_path)
                if not dataset.empty:
                    datasets.append(dataset)
                    station_info[station_name] = {
                        'path': dataset_path,
                        'records': len(dataset),
                        'columns': list(dataset.columns)
                    }
            except Exception as e:
                self.logger.error(f"Error loading dataset {dataset_path}: {e}")
        
        if not datasets:
            self.logger.error("No valid datasets found for merging")
            return pd.DataFrame()
        
        # Merge datasets
        unified_dataset = self._merge_dataframes(datasets)
        
        # Save unified dataset
        self._save_unified_dataset(unified_dataset, output_path)
        
        # Save metadata
        self._save_merge_metadata(station_info, unified_dataset, output_path)
        
        self.logger.info(f"Successfully merged {len(datasets)} datasets into {output_path}")
        return unified_dataset
    
    def _load_and_process_dataset(self, dataset_path: str) -> tuple[pd.DataFrame, str]:
        """
        Load and process a single station dataset.
        
        Args:
            dataset_path: Path to the dataset file
            
        Returns:
            Tuple of (processed DataFrame, station name)
        """
        # Extract station name from file path
        station_name = Path(dataset_path).stem
        
        # Load dataset
        dataset = pd.read_csv(dataset_path)
        
        # Add station identifier
        dataset['station_name'] = station_name
        
        # Standardize column names
        dataset = self._standardize_columns(dataset)
        
        # Convert data types
        dataset = self._convert_data_types(dataset)
        
        return dataset, station_name
    
    def _standardize_columns(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names across datasets.
        
        Args:
            dataset: Input DataFrame
            
        Returns:
            DataFrame with standardized column names
        """
        # Create a copy to avoid modifying original
        standardized = dataset.copy()
        
        # Standardize column names
        column_mapping = {
            'From Date': 'timestamp',
            'timestamp_utc': 'timestamp_utc',
            'timestamp_hour': 'timestamp_hour',
            'PM2.5 (ug/m3)': 'pm25',
            'PM10 (ug/m3)': 'pm10',
            'NO (ug/m3)': 'no',
            'NO2 (ug/m3)': 'no2',
            'NOx (ppb)': 'nox',
            'NH3 (ug/m3)': 'nh3',
            'SO2 (ug/m3)': 'so2',
            'CO (mg/m3)': 'co',
            'Ozone (ug/m3)': 'ozone',
            'Temp (degree C)': 'temperature',
            'RH (%)': 'relative_humidity',
            'WS (m/s)': 'wind_speed',
            'WD (deg)': 'wind_direction',
            'SR (W/mt2)': 'solar_radiation',
            'BP (mmHg)': 'barometric_pressure',
            'RF (mm)': 'rainfall',
            'AT (degree C)': 'apparent_temperature'
        }
        
        # Apply column mapping
        for old_name, new_name in column_mapping.items():
            if old_name in standardized.columns:
                standardized = standardized.rename(columns={old_name: new_name})
        
        # Clean up remaining column names
        standardized.columns = [col.lower().replace(' ', '_').replace('(', '').replace(')', '') 
                              for col in standardized.columns]
        
        return standardized
    
    def _convert_data_types(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Convert data types for consistency across datasets.
        
        Args:
            dataset: Input DataFrame
            
        Returns:
            DataFrame with converted data types
        """
        # Create a copy to avoid modifying original
        converted = dataset.copy()
        
        # Convert timestamp columns
        timestamp_columns = ['timestamp', 'timestamp_utc', 'timestamp_hour', 'satellite_timestamp']
        for col in timestamp_columns:
            if col in converted.columns:
                converted[col] = pd.to_datetime(converted[col], errors='coerce')
        
        # Convert numeric columns
        numeric_columns = ['pm25', 'pm10', 'no', 'no2', 'nox', 'nh3', 'so2', 'co', 'ozone',
                          'temperature', 'relative_humidity', 'wind_speed', 'wind_direction',
                          'solar_radiation', 'barometric_pressure', 'rainfall', 'apparent_temperature',
                          'time_diff_hours']
        
        for col in numeric_columns:
            if col in converted.columns:
                converted[col] = pd.to_numeric(converted[col], errors='coerce')
        
        # Convert categorical columns
        categorical_columns = ['station_name']
        for col in categorical_columns:
            if col in converted.columns:
                converted[col] = converted[col].astype('category')
        
        return converted
    
    def _merge_dataframes(self, datasets: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Merge multiple DataFrames into a single unified dataset.
        
        Args:
            datasets: List of DataFrames to merge
            
        Returns:
            Unified DataFrame
        """
        if len(datasets) == 1:
            return datasets[0]
        
        # Find common columns across all datasets
        all_columns = set()
        for dataset in datasets:
            all_columns.update(dataset.columns)
        
        # Ensure all datasets have the same columns
        standardized_datasets = []
        for dataset in datasets:
            # Add missing columns with NaN values
            for col in all_columns:
                if col not in dataset.columns:
                    dataset[col] = np.nan
            standardized_datasets.append(dataset)
        
        # Concatenate datasets
        unified_dataset = pd.concat(standardized_datasets, ignore_index=True, sort=False)
        
        # Sort by timestamp if available
        if 'timestamp_utc' in unified_dataset.columns:
            unified_dataset = unified_dataset.sort_values('timestamp_utc')
        
        return unified_dataset
    
    def _save_unified_dataset(self, dataset: pd.DataFrame, output_path: Path) -> None:
        """
        Save the unified dataset to file.
        
        Args:
            dataset: Unified dataset DataFrame
            output_path: Path to save the dataset
        """
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save dataset
        dataset.to_csv(output_path, index=False)
        
        self.logger.info(f"Unified dataset saved to {output_path}")
    
    def _save_merge_metadata(self, station_info: Dict[str, Any], 
                           unified_dataset: pd.DataFrame, 
                           output_path: Path) -> None:
        """
        Save metadata about the merge process.
        
        Args:
            station_info: Information about each station dataset
            unified_dataset: Unified dataset DataFrame
            output_path: Path to the unified dataset (for metadata path)
        """
        metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
        
        metadata = {
            'merge_timestamp': datetime.now().isoformat(),
            'total_stations': len(station_info),
            'total_records': len(unified_dataset),
            'stations': station_info,
            'columns': list(unified_dataset.columns),
            'data_types': unified_dataset.dtypes.to_dict(),
            'missing_values': unified_dataset.isnull().sum().to_dict()
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Merge metadata saved to {metadata_path}")
    
    def get_merge_statistics(self, unified_dataset: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about the merged dataset.
        
        Args:
            unified_dataset: Unified dataset DataFrame
            
        Returns:
            Dictionary containing merge statistics
        """
        if unified_dataset.empty:
            return {'error': 'No data available'}
        
        stats = {
            'total_records': len(unified_dataset),
            'total_stations': unified_dataset['station_name'].nunique() if 'station_name' in unified_dataset.columns else 0,
            'date_range': {},
            'station_distribution': {},
            'column_statistics': {}
        }
        
        # Date range statistics
        if 'timestamp_utc' in unified_dataset.columns:
            stats['date_range'] = {
                'start': unified_dataset['timestamp_utc'].min().isoformat(),
                'end': unified_dataset['timestamp_utc'].max().isoformat(),
                'duration_days': (unified_dataset['timestamp_utc'].max() - 
                                unified_dataset['timestamp_utc'].min()).days
            }
        
        # Station distribution
        if 'station_name' in unified_dataset.columns:
            station_counts = unified_dataset['station_name'].value_counts()
            stats['station_distribution'] = station_counts.to_dict()
        
        # Column statistics
        for column in unified_dataset.columns:
            if column in ['timestamp_utc', 'timestamp', 'timestamp_hour', 'satellite_timestamp']:
                continue
            
            col_stats = {
                'dtype': str(unified_dataset[column].dtype),
                'missing_count': int(unified_dataset[column].isnull().sum()),
                'missing_percentage': float(unified_dataset[column].isnull().sum() / len(unified_dataset) * 100)
            }
            
            # Add numeric statistics for numeric columns
            if pd.api.types.is_numeric_dtype(unified_dataset[column]):
                non_null_data = unified_dataset[column].dropna()
                if len(non_null_data) > 0:
                    col_stats.update({
                        'min': float(non_null_data.min()),
                        'max': float(non_null_data.max()),
                        'mean': float(non_null_data.mean()),
                        'std': float(non_null_data.std()),
                        'median': float(non_null_data.median())
                    })
            
            stats['column_statistics'][column] = col_stats
        
        return stats
    
    def create_station_specific_datasets(self, unified_dataset: pd.DataFrame, 
                                       output_dir: Path) -> Dict[str, str]:
        """
        Create separate datasets for each station.
        
        Args:
            unified_dataset: Unified dataset DataFrame
            output_dir: Directory to save station-specific datasets
            
        Returns:
            Dictionary mapping station names to their dataset paths
        """
        if 'station_name' not in unified_dataset.columns:
            self.logger.error("Station name column not found in unified dataset")
            return {}
        
        output_dir.mkdir(parents=True, exist_ok=True)
        station_datasets = {}
        
        for station_name in unified_dataset['station_name'].unique():
            # Filter data for this station
            station_data = unified_dataset[unified_dataset['station_name'] == station_name]
            
            # Save station-specific dataset
            station_file = output_dir / f"{station_name}_dataset.csv"
            station_data.to_csv(station_file, index=False)
            
            station_datasets[station_name] = str(station_file)
            
            self.logger.info(f"Created dataset for {station_name}: {len(station_data)} records")
        
        return station_datasets
    
    def validate_unified_dataset(self, unified_dataset: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the unified dataset quality.
        
        Args:
            unified_dataset: Unified dataset DataFrame
            
        Returns:
            Dictionary containing validation results
        """
        if unified_dataset.empty:
            return {'error': 'No data to validate'}
        
        validation = {
            'total_records': len(unified_dataset),
            'total_stations': 0,
            'missing_values': {},
            'data_quality': {},
            'timestamp_coverage': 0,
            'satellite_coverage': 0
        }
        
        # Station count
        if 'station_name' in unified_dataset.columns:
            validation['total_stations'] = unified_dataset['station_name'].nunique()
        
        # Missing values analysis
        for column in unified_dataset.columns:
            missing_count = unified_dataset[column].isnull().sum()
            missing_pct = (missing_count / len(unified_dataset)) * 100
            validation['missing_values'][column] = {
                'count': int(missing_count),
                'percentage': float(missing_pct)
            }
        
        # Timestamp coverage
        if 'timestamp_utc' in unified_dataset.columns:
            validation['timestamp_coverage'] = int(unified_dataset['timestamp_utc'].notna().sum())
        
        # Satellite coverage
        satellite_columns = [col for col in unified_dataset.columns if 'satellite' in col.lower()]
        if satellite_columns:
            validation['satellite_coverage'] = int(
                unified_dataset[satellite_columns].notna().any(axis=1).sum()
            )
        
        # Data quality checks
        validation['data_quality'] = {
            'duplicate_records': int(unified_dataset.duplicated().sum()),
            'unique_timestamps': int(unified_dataset['timestamp_utc'].nunique()) if 'timestamp_utc' in unified_dataset.columns else 0
        }
        
        return validation 