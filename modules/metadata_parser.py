"""
Metadata parser for AEP 3.0 pipeline.

This module handles parsing of station metadata files containing station locations,
coordinates, and bounding boxes for satellite image cropping.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Any
import re


class MetadataParser:
    """
    Parser for station metadata files.
    
    This class handles parsing of station metadata files that contain:
    - Station names and locations
    - Latitude and longitude coordinates
    - Bounding box coordinates for satellite image cropping
    """
    
    def __init__(self):
        """Initialize the metadata parser."""
        self.logger = logging.getLogger(__name__)
    
    def parse_metadata(self, metadata_path: Path) -> Dict[str, Dict[str, Any]]:
        """
        Parse station metadata file.
        
        Args:
            metadata_path: Path to the metadata CSV file
            
        Returns:
            Dictionary mapping station names to their metadata
        """
        self.logger.info(f"Parsing metadata from {metadata_path}")
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        try:
            # Read CSV with BOM handling
            df = pd.read_csv(metadata_path, encoding='utf-8-sig')
            
            # Clean column names (strip whitespace and remove BOMs)
            df.columns = [col.strip().replace('\ufeff', '').lower() for col in df.columns]
            
            # Parse each station's metadata
            stations_metadata = {}
            
            for _, row in df.iterrows():
                station_metadata = self._parse_station_row(row)
                if station_metadata:
                    station_name = station_metadata['station_name']
                    stations_metadata[station_name] = station_metadata
            
            self.logger.info(f"Successfully parsed metadata for {len(stations_metadata)} stations")
            return stations_metadata
            
        except Exception as e:
            self.logger.error(f"Error parsing metadata file {metadata_path}: {e}")
            raise
    
    def _parse_csv_line(self, line: str) -> List[str]:
        """
        Parse a CSV line with quoted fields.
        
        Args:
            line: CSV line to parse
            
        Returns:
            List of parsed fields
        """
        parts = []
        current_part = ""
        in_quotes = False
        i = 0
        
        while i < len(line):
            char = line[i]
            
            if char == '"':
                if in_quotes and i + 1 < len(line) and line[i + 1] == '"':
                    # Escaped quote (double quote)
                    current_part += '"'
                    i += 2
                    continue
                else:
                    # Toggle quote state
                    in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                # End of field
                parts.append(current_part.strip())
                current_part = ""
            else:
                # Regular character
                current_part += char
            
            i += 1
        
        # Add the last field
        parts.append(current_part.strip())
        
        return parts
    
    def _parse_station_row(self, row: pd.Series) -> Dict[str, Any]:
        """
        Parse a single station row from the metadata file.
        
        Args:
            row: Pandas Series containing station data
            
        Returns:
            Dictionary containing parsed station metadata
        """
        try:
            # Sanitize the row dictionary keys by stripping whitespace and removing BOMs
            row_dict = row.to_dict()
            row_dict = {k.strip().replace('\ufeff', ''): v for k, v in row_dict.items()}
            
            # Safely access file_name with validation
            station_name = str(row_dict.get('file_name', '')).strip()
            if not station_name:
                self.logger.warning(f"[METADATA PARSER] Skipping row with empty file_name: {row_dict}")
                return {}
            
            # Extract basic information
            station_location = str(row_dict.get('station_location', '')).strip()
            
            # Extract coordinates as floats
            latitude = float(row_dict.get('latitude', 0))
            longitude = float(row_dict.get('longitude', 0))
            west_lon = float(row_dict.get('west_lon', 0))
            east_lon = float(row_dict.get('east_lon', 0))
            south_lat = float(row_dict.get('south_lat', 0))
            north_lat = float(row_dict.get('north_lat', 0))
            
            # Validate coordinates
            if not self._validate_coordinates(latitude, longitude, west_lon, east_lon, south_lat, north_lat):
                self.logger.warning(f"Invalid coordinates for station {station_name}")
                return {}
            
            # Create metadata dictionary with all required fields
            metadata = {
                'station_name': station_name,
                'file_name': station_name,  # Include file_name for pipeline use
                'station_location': station_location,
                'latitude': latitude,
                'longitude': longitude,
                'west_lon': west_lon,
                'east_lon': east_lon,
                'south_lat': south_lat,
                'north_lat': north_lat,
                'bounding_box': {
                    'west_lon': west_lon,
                    'east_lon': east_lon,
                    'south_lat': south_lat,
                    'north_lat': north_lat
                },
                'center_point': {
                    'lat': latitude,
                    'lon': longitude
                }
            }
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error parsing station row: {e}")
            return {}
    
    def _validate_coordinates(self, lat: float, lon: float, 
                            west_lon: float, east_lon: float, 
                            south_lat: float, north_lat: float) -> bool:
        """
        Validate coordinate values.
        
        Args:
            lat: Station latitude
            lon: Station longitude
            west_lon: Western longitude boundary
            east_lon: Eastern longitude boundary
            south_lat: Southern latitude boundary
            north_lat: Northern latitude boundary
            
        Returns:
            True if coordinates are valid, False otherwise
        """
        # Check latitude range (-90 to 90)
        if not (-90 <= lat <= 90) or not (-90 <= south_lat <= 90) or not (-90 <= north_lat <= 90):
            return False
        
        # Check longitude range (-180 to 180)
        if not (-180 <= lon <= 180) or not (-180 <= west_lon <= 180) or not (-180 <= east_lon <= 180):
            return False
        
        # Check bounding box consistency
        if west_lon >= east_lon:
            return False
        
        if south_lat >= north_lat:
            return False
        
        # Check if station is within bounding box
        if not (west_lon <= lon <= east_lon):
            return False
        
        if not (south_lat <= lat <= north_lat):
            return False
        
        return True
    
    def get_station_bounds(self, station_metadata: Dict[str, Any], 
                          buffer_km: float = 10.0) -> Dict[str, float]:
        """
        Get station bounding box with buffer.
        
        Args:
            station_metadata: Station metadata dictionary
            buffer_km: Buffer distance in kilometers
            
        Returns:
            Dictionary containing buffered bounding box coordinates
        """
        import math
        
        # Convert buffer from km to degrees (approximate)
        # 1 degree latitude ≈ 111 km
        # 1 degree longitude ≈ 111 km * cos(latitude)
        lat = station_metadata['latitude']
        buffer_deg_lat = buffer_km / 111.0
        buffer_deg_lon = buffer_km / (111.0 * math.cos(math.radians(lat)))
        
        # Get original bounding box
        bbox = station_metadata['bounding_box']
        
        # Apply buffer
        buffered_bounds = {
            'west_lon': bbox['west_lon'] - buffer_deg_lon,
            'east_lon': bbox['east_lon'] + buffer_deg_lon,
            'south_lat': bbox['south_lat'] - buffer_deg_lat,
            'north_lat': bbox['north_lat'] + buffer_deg_lat
        }
        
        # Ensure bounds are within valid ranges
        buffered_bounds['south_lat'] = max(-90, buffered_bounds['south_lat'])
        buffered_bounds['north_lat'] = min(90, buffered_bounds['north_lat'])
        buffered_bounds['west_lon'] = max(-180, buffered_bounds['west_lon'])
        buffered_bounds['east_lon'] = min(180, buffered_bounds['east_lon'])
        
        return buffered_bounds
    
    def validate_metadata_file(self, metadata_path: Path) -> Dict[str, Any]:
        """
        Validate metadata file and return statistics.
        
        Args:
            metadata_path: Path to metadata file
            
        Returns:
            Dictionary containing validation statistics
        """
        if not metadata_path.exists():
            return {'error': 'File not found'}
        
        try:
            df = pd.read_csv(metadata_path)
            
            stats = {
                'total_stations': len(df),
                'valid_stations': 0,
                'invalid_stations': 0,
                'missing_columns': [],
                'coordinate_errors': []
            }
            
            # Check required columns
            required_columns = ['file_name', 'latitude', 'longitude', 
                              'west_lon', 'east_lon', 'south_lat', 'north_lat']
            
            for col in required_columns:
                if col not in df.columns:
                    stats['missing_columns'].append(col)
            
            # Validate each station
            for _, row in df.iterrows():
                station_metadata = self._parse_station_row(row)
                if station_metadata:
                    stats['valid_stations'] += 1
                else:
                    stats['invalid_stations'] += 1
                    stats['coordinate_errors'].append(row.get('file_name', 'unknown'))
            
            return stats
            
        except Exception as e:
            return {'error': str(e)}
    
    def export_metadata_summary(self, stations_metadata: Dict[str, Dict[str, Any]], 
                               output_path: Path) -> None:
        """
        Export metadata summary to a file.
        
        Args:
            stations_metadata: Dictionary of station metadata
            output_path: Path to save the summary
        """
        import json
        
        summary = {
            'total_stations': len(stations_metadata),
            'stations': {}
        }
        
        for station_name, metadata in stations_metadata.items():
            summary['stations'][station_name] = {
                'location': metadata['station_location'],
                'coordinates': metadata['center_point'],
                'bounding_box': metadata['bounding_box']
            }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Metadata summary exported to {output_path}") 