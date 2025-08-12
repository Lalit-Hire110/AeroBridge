#!/usr/bin/env python3
"""
Example usage of AEP 3.0 pipeline.

This script demonstrates how to use the pipeline programmatically.
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from aep_pipeline import AEPPipeline
from modules.utils import setup_logging


def main():
    """Demonstrate pipeline usage."""
    print("🚀 AEP 3.0 Pipeline Example Usage")
    print("=" * 50)
    
    # Setup logging
    setup_logging(level="INFO")
    
    # Initialize pipeline
    print("1. Initializing pipeline...")
    pipeline = AEPPipeline(data_root="data", buffer_km=10.0)
    
    # Check pipeline status
    print("\n2. Checking pipeline status...")
    status = pipeline.get_pipeline_status()
    print(f"   Data root: {status['data_root']}")
    print(f"   Raw data exists: {status['raw_data_exists']}")
    print(f"   CPCB data exists: {status['cpcb_data_exists']}")
    print(f"   Available states: {status['available_states']}")
    
    # Process a single state (Haryana)
    print("\n3. Processing Haryana state...")
    try:
        results = pipeline.run_pipeline(
            states=["Haryana"],
            create_unified_dataset=False  # Skip unified dataset for demo
        )
        
        print("   Processing completed!")
        print("   Results:")
        for state, state_data in results.items():
            if isinstance(state_data, dict) and "error" not in state_data:
                print(f"     {state}:")
                for station, station_data in state_data.items():
                    if isinstance(station_data, dict):
                        print(f"       {station}:")
                        for key, value in station_data.items():
                            print(f"         {key}: {value}")
            else:
                print(f"     {state}: Error - {state_data}")
                
    except Exception as e:
        print(f"   Error during processing: {e}")
        print("   This is expected if dependencies are not installed.")
    
    print("\n4. Pipeline demonstration completed!")
    print("\nTo run the full pipeline:")
    print("   pip install -r requirements.txt")
    print("   python aep_pipeline.py --states Haryana")
    
    print("\nFor more options:")
    print("   python aep_pipeline.py --help")


if __name__ == "__main__":
    main() 