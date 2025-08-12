#!/usr/bin/env python3
"""
AEP 3.0 GUI Application
A Windows desktop application for running the Air Emissions Prediction Pipeline
with a user-friendly interface.

Author: Lalit Hire (BSc Data Science @ Department of technology, Savitribai Phule Pune University)
Date: 2025
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys
import shutil
from pathlib import Path
import logging
from datetime import datetime
import traceback

# Add parent directory to path to import pipeline modules
sys.path.append(str(Path(__file__).parent.parent))

try:
    from robust_aep_pipeline_final import RobustAEPPipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    try:
        from aep_31_complete_pipeline import AEP31Pipeline
        PIPELINE_AVAILABLE = True
    except ImportError:
        PIPELINE_AVAILABLE = False

# Import enhanced feature pipeline
try:
    from enhanced_feature_pipeline_v2 import EnhancedINSATFeatureExtractor
    ENHANCED_PIPELINE_AVAILABLE = True
except ImportError:
    ENHANCED_PIPELINE_AVAILABLE = False


class AEPGUIApp:
    """Main GUI Application for AEP 3.0 Pipeline"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("AEP 3.0 - Air Emissions Prediction Pipeline by Lalit Hire")
        self.root.geometry("800x700")
        self.root.resizable(True, True)
        
        # Application state
        self.input_folder = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.is_running = False
        self.pipeline_thread = None
        
        # Setup GUI
        self.setup_gui()
        self.setup_logging()
        
        # Check pipeline availability
        if not PIPELINE_AVAILABLE:
            self.log_message("ERROR: Pipeline modules not found! Please ensure pipeline files are in the parent directory.", "error")
        elif not ENHANCED_PIPELINE_AVAILABLE:
            self.log_message("WARNING: Enhanced feature pipeline not found. Only basic pipeline will run.", "warning")
        else:
            self.log_message("SUCCESS: Both main and enhanced pipelines available!", "success")
    
    def setup_gui(self):
        """Setup the main GUI interface"""
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="AEP 3.0 Air Emissions Prediction Pipeline", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Input folder selection
        ttk.Label(main_frame, text="Raw INSAT Images Folder:", font=("Arial", 10, "bold")).grid(
            row=1, column=0, sticky=tk.W, pady=(0, 5))
        
        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        input_frame.columnconfigure(0, weight=1)
        
        self.input_entry = ttk.Entry(input_frame, textvariable=self.input_folder, width=60)
        self.input_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        ttk.Button(input_frame, text="Browse", command=self.browse_input_folder).grid(
            row=0, column=1)
        
        # Output folder selection
        ttk.Label(main_frame, text="Output Folder:", font=("Arial", 10, "bold")).grid(
            row=3, column=0, sticky=tk.W, pady=(0, 5))
        
        output_frame = ttk.Frame(main_frame)
        output_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        output_frame.columnconfigure(0, weight=1)
        
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_folder, width=60)
        self.output_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        ttk.Button(output_frame, text="Browse", command=self.browse_output_folder).grid(
            row=0, column=1)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=3, pady=(0, 15))
        
        self.run_button = ttk.Button(button_frame, text="Run Pipeline", 
                                    command=self.run_pipeline, style="Accent.TButton")
        self.run_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text="Stop", 
                                     command=self.stop_pipeline, state="disabled")
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT)
        
        # Progress bar
        self.progress_var = tk.StringVar(value="Ready")
        ttk.Label(main_frame, text="Status:", font=("Arial", 10, "bold")).grid(
            row=6, column=0, sticky=tk.W, pady=(0, 5))
        
        self.status_label = ttk.Label(main_frame, textvariable=self.progress_var, 
                                     foreground="green")
        self.status_label.grid(row=7, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        
        self.progress_bar = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress_bar.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # Log area
        ttk.Label(main_frame, text="Pipeline Log:", font=("Arial", 10, "bold")).grid(
            row=9, column=0, sticky=tk.W, pady=(0, 5))
        
        # Create log text area with scrollbar
        self.log_text = scrolledtext.ScrolledText(main_frame, height=20, width=80, 
                                                 font=("Consolas", 9))
        self.log_text.grid(row=10, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), 
                          pady=(0, 10))
        
        # Configure text tags for colored logging
        self.log_text.tag_configure("error", foreground="red")
        self.log_text.tag_configure("warning", foreground="orange")
        self.log_text.tag_configure("success", foreground="green")
        self.log_text.tag_configure("info", foreground="blue")
        
        # Configure grid weights for resizing
        main_frame.rowconfigure(10, weight=1)
        
        # Initial log message
        self.log_message("AEP 3.0 Pipeline GUI initialized successfully!", "success")
        self.log_message("Two-stage pipeline: (1) Cropping + Merging, (2) Enhanced Feature Extraction", "info")
        self.log_message("Select input folder with raw INSAT images and output folder, then click 'Run Pipeline'", "info")
    
    def setup_logging(self):
        """Setup logging to capture pipeline output"""
        # Create a custom handler to redirect pipeline logs to GUI
        class GUILogHandler(logging.Handler):
            def __init__(self, gui_app):
                super().__init__()
                self.gui_app = gui_app
            
            def emit(self, record):
                msg = self.format(record)
                # Determine log level for coloring
                if record.levelno >= logging.ERROR:
                    tag = "error"
                elif record.levelno >= logging.WARNING:
                    tag = "warning"
                elif "SUCCESS" in msg or "COMPLETE" in msg:
                    tag = "success"
                else:
                    tag = "info"
                
                self.gui_app.log_message(msg, tag)
        
        # Setup the handler
        self.gui_handler = GUILogHandler(self)
        self.gui_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    def browse_input_folder(self):
        """Browse for input folder containing raw INSAT images"""
        folder = filedialog.askdirectory(title="Select Raw INSAT Images Folder")
        if folder:
            self.input_folder.set(folder)
            self.log_message(f"Input folder selected: {folder}", "info")
    
    def browse_output_folder(self):
        """Browse for output folder"""
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder.set(folder)
            self.log_message(f"Output folder selected: {folder}", "info")
    
    def log_message(self, message, tag="info"):
        """Add message to log with timestamp and coloring"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        # Thread-safe GUI update
        def update_log():
            self.log_text.insert(tk.END, formatted_message, tag)
            self.log_text.see(tk.END)
        
        if threading.current_thread() == threading.main_thread():
            update_log()
        else:
            self.root.after(0, update_log)
    
    def clear_log(self):
        """Clear the log text area"""
        self.log_text.delete(1.0, tk.END)
        self.log_message("Log cleared", "info")
    
    def validate_inputs(self):
        """Validate user inputs before running pipeline"""
        if not self.input_folder.get():
            messagebox.showerror("Error", "Please select an input folder with raw INSAT images")
            return False
        
        if not self.output_folder.get():
            messagebox.showerror("Error", "Please select an output folder")
            return False
        
        if not os.path.exists(self.input_folder.get()):
            messagebox.showerror("Error", "Input folder does not exist")
            return False
        
        if not os.path.exists(self.output_folder.get()):
            try:
                os.makedirs(self.output_folder.get(), exist_ok=True)
            except Exception as e:
                messagebox.showerror("Error", f"Cannot create output folder: {e}")
                return False
        
        if not PIPELINE_AVAILABLE:
            messagebox.showerror("Error", "Pipeline modules not available. Please check installation.")
            return False
        
        # Enhanced pipeline is optional but recommended
        if not ENHANCED_PIPELINE_AVAILABLE:
            response = messagebox.askyesno(
                "Enhanced Pipeline Missing", 
                "Enhanced feature extraction pipeline not found.\n\n"
                "The app will run only the basic pipeline (cropping + merging).\n\n"
                "Do you want to continue?"
            )
            if not response:
                return False
        
        return True
    
    def run_pipeline(self):
        """Run the AEP pipeline in a separate thread"""
        if not self.validate_inputs():
            return
        
        if self.is_running:
            messagebox.showwarning("Warning", "Pipeline is already running!")
            return
        
        # Start pipeline in separate thread
        self.pipeline_thread = threading.Thread(target=self._run_pipeline_worker, daemon=True)
        self.pipeline_thread.start()
    
    def _run_pipeline_worker(self):
        """Worker thread for running the pipeline"""
        try:
            self.is_running = True
            self._update_ui_running(True)
            
            input_path = Path(self.input_folder.get())
            output_path = Path(self.output_folder.get())
            
            self.log_message("="*60, "info")
            self.log_message("STARTING AEP 3.0 PIPELINE EXECUTION", "success")
            self.log_message("="*60, "info")
            self.log_message(f"Input folder: {input_path}", "info")
            self.log_message(f"Output folder: {output_path}", "info")
            
            # Check if cropped_data already exists in input folder
            input_cropped_data = input_path / "cropped_data"
            skip_cropping = input_cropped_data.exists() and any(input_cropped_data.iterdir())
            
            if skip_cropping:
                self.log_message("🔍 DETECTED: Existing cropped_data directory in input folder", "success")
                self.log_message("⚡ OPTIMIZATION: Skipping cropping step, using existing cropped images", "info")
            else:
                self.log_message("📁 No existing cropped_data found - will perform full pipeline", "info")
            
            # Setup data structure in output folder
            data_root = output_path / "aep_data"
            data_root.mkdir(exist_ok=True)
            
            # Copy CPCB data to output location
            source_cpcb = Path(__file__).parent.parent / "data" / "cpcb"
            target_cpcb = data_root / "cpcb"
            
            if source_cpcb.exists() and not target_cpcb.exists():
                self.log_message("Copying CPCB station data...", "info")
                shutil.copytree(source_cpcb, target_cpcb)
                self.log_message("CPCB data copied successfully", "success")
            
            # Setup raw data and cropped data directories
            raw_data_dir = data_root / "raw"
            cropped_data_dir = data_root / "cropped_data"
            
            if skip_cropping:
                # Use existing cropped data directly
                if cropped_data_dir.exists():
                    shutil.rmtree(cropped_data_dir)
                shutil.copytree(input_cropped_data, cropped_data_dir)
                self.log_message(f"Existing cropped data copied to working directory: {cropped_data_dir}", "success")
                
                # Still need to set up raw data directory for any remaining raw images
                if raw_data_dir.exists():
                    shutil.rmtree(raw_data_dir)
                
                # Copy only non-cropped content to raw directory
                raw_data_dir.mkdir(exist_ok=True)
                for item in input_path.iterdir():
                    if item.name != "cropped_data":
                        if item.is_dir():
                            shutil.copytree(item, raw_data_dir / item.name)
                        else:
                            shutil.copy2(item, raw_data_dir / item.name)
                self.log_message("Raw data (excluding cropped_data) copied to working directory", "info")
            else:
                # Normal workflow - copy all raw data
                if raw_data_dir.exists():
                    shutil.rmtree(raw_data_dir)
                
                # Create symbolic link or copy raw data
                try:
                    if os.name == 'nt':  # Windows
                        shutil.copytree(input_path, raw_data_dir)
                        self.log_message("Raw data copied to working directory", "info")
                    else:
                        os.symlink(input_path, raw_data_dir)
                        self.log_message("Raw data linked to working directory", "info")
                except Exception as e:
                    self.log_message(f"Warning: Could not link raw data: {e}", "warning")
                    self.log_message("Attempting to copy raw data...", "info")
                    shutil.copytree(input_path, raw_data_dir)
                    self.log_message("Raw data copied successfully", "success")
            
            # Initialize and run pipeline
            self.log_message("Initializing AEP pipeline...", "info")
            
            # Try robust pipeline first, then fallback to regular pipeline
            try:
                pipeline = RobustAEPPipeline(data_root=str(data_root))
                self.log_message("Using Robust AEP Pipeline", "success")
            except NameError:
                pipeline = AEP31Pipeline(data_root=str(data_root))
                self.log_message("Using AEP 3.1 Pipeline", "success")
            
            # Add GUI logging handler to pipeline logger
            pipeline.logger.addHandler(self.gui_handler)
            pipeline.logger.setLevel(logging.INFO)
            
            # Run Stage 1 conditionally based on whether cropping is needed
            if skip_cropping:
                self.log_message("Starting Stage 1: Processing existing cropped data (merging only)...", "info")
                # Create a minimal results structure since we're skipping cropping
                results = {
                    'stations_processed': 1,  # Will be updated by actual processing
                    'stations_failed': 0,
                    'images_cropped': 0,  # No new cropping done
                    'features_extracted': 0,
                    'records_merged': 0,
                    'failed_stations': [],
                    'duration_minutes': 0
                }
                
                # Still need to run the merging part of the pipeline
                # The pipeline will find existing cropped data and process it
                try:
                    pipeline_results = pipeline.run_complete_pipeline()
                    if pipeline_results:
                        results.update(pipeline_results)
                except Exception as e:
                    self.log_message(f"Error in processing existing cropped data: {e}", "error")
                    results['stations_failed'] = 1
            else:
                self.log_message("Starting Stage 1: Full pipeline execution (cropping + merging)...", "info")
                results = pipeline.run_complete_pipeline()
            
            # Process Stage 1 results
            if results and results.get('stations_processed', 0) > 0:
                self.log_message("="*60, "success")
                self.log_message("STAGE 1 COMPLETED SUCCESSFULLY!", "success")
                self.log_message("="*60, "success")
                
                # Run Stage 2: Enhanced Feature Extraction (if available)
                enhanced_results = None
                if ENHANCED_PIPELINE_AVAILABLE:
                    self.log_message("\n" + "="*60, "info")
                    self.log_message("Starting Stage 2: Enhanced feature extraction...", "info")
                    self.log_message("="*60, "info")
                    
                    try:
                        # Initialize enhanced feature extractor with correct paths
                        enhanced_extractor = EnhancedINSATFeatureExtractor(str(data_root))
                        
                        # Ensure the enhanced pipeline can find the cropped data
                        expected_cropped_path = data_root / "cropped_data"
                        if not expected_cropped_path.exists():
                            # Try to find cropped data in alternative locations
                            alt_cropped_path = data_root / "data" / "cropped_data"
                            if alt_cropped_path.exists():
                                self.log_message(f"Found cropped data at: {alt_cropped_path}", "info")
                            else:
                                self.log_message(f"WARNING: No cropped data found at expected location: {expected_cropped_path}", "warning")
                        else:
                            self.log_message(f"Cropped data found at: {expected_cropped_path}", "success")
                        
                        # Override paths in the enhanced extractor to match our structure
                        enhanced_extractor.cropped_data_path = expected_cropped_path
                        enhanced_extractor.processed_data_path = data_root / "processed_data"
                        enhanced_extractor.unified_dataset_path = data_root / "unified_dataset_enhanced_v2.csv"
                        
                        # Ensure processed_data directory exists
                        enhanced_extractor.processed_data_path.mkdir(parents=True, exist_ok=True)
                        
                        # Add GUI logging handler to enhanced pipeline logger
                        enhanced_extractor.logger = pipeline.logger  # Use same logger
                        
                        # Run enhanced pipeline
                        enhanced_success = enhanced_extractor.run_enhanced_pipeline()
                        
                        if enhanced_success:
                            self.log_message("="*60, "success")
                            self.log_message("STAGE 2 COMPLETED SUCCESSFULLY!", "success")
                            self.log_message("Enhanced feature-enriched dataset created!", "success")
                            self.log_message("="*60, "success")
                            
                            enhanced_results = {
                                'success': True,
                                'enhanced_dataset_path': str(enhanced_extractor.unified_dataset_path),
                                'feature_cache_size': len(enhanced_extractor.feature_cache)
                            }
                        else:
                            self.log_message("Stage 2 failed - using basic dataset", "warning")
                            enhanced_results = {'success': False}
                    
                    except Exception as e:
                        self.log_message(f"Stage 2 failed: {str(e)}", "error")
                        enhanced_results = {'success': False, 'error': str(e)}
                else:
                    self.log_message("Stage 2 skipped - enhanced pipeline not available", "warning")
                
                # Copy final outputs to main output folder
                self._copy_final_outputs(data_root, output_path, results, enhanced_results)
                
                # Show completion message
                success_msg = self._build_completion_message(results, enhanced_results)
                self.root.after(0, lambda: messagebox.showinfo("Success", success_msg))
            else:
                self.log_message("Stage 1 completed with errors or no data processed", "warning")
                self.root.after(0, lambda: messagebox.showwarning(
                    "Warning", 
                    "Stage 1 completed but no stations were successfully processed.\n"
                    "Check the log for details."
                ))
        
        except Exception as e:
            error_msg = f"Pipeline execution failed: {str(e)}"
            self.log_message(error_msg, "error")
            self.log_message(f"Error details: {traceback.format_exc()}", "error")
            
            self.root.after(0, lambda: messagebox.showerror(
                "Error", 
                f"Pipeline execution failed:\n{str(e)}\n\nCheck the log for details."
            ))
        
        finally:
            self.is_running = False
            self._update_ui_running(False)
    
    def _build_completion_message(self, results, enhanced_results=None):
        """Build completion message for both pipelines"""
        msg = "AEP 3.0 Pipeline completed successfully!\n\n"
        
        # Stage 1 results
        msg += "📊 STAGE 1 - Main Pipeline:\n"
        msg += f"  • Stations processed: {results.get('stations_processed', 0)}\n"
        
        # Show different message based on whether cropping was done
        images_cropped = results.get('images_cropped', 0)
        if images_cropped > 0:
            msg += f"  • Images cropped: {images_cropped:,}\n"
        else:
            msg += "  • Used existing cropped images (cropping skipped)\n"
        
        msg += f"  • Records merged: {results.get('records_merged', 0):,}\n\n"
        
        # Stage 2 results
        if enhanced_results:
            if enhanced_results.get('success'):
                msg += "🚀 STAGE 2 - Enhanced Features:\n"
                msg += "  • Feature extraction: SUCCESS\n"
                msg += "  • Enhanced dataset: CREATED\n"
                msg += f"  • Features cached: {enhanced_results.get('feature_cache_size', 0)}\n\n"
                msg += "📁 MAIN OUTPUT: Enhanced feature-enriched dataset\n"
            else:
                msg += "⚠️ STAGE 2 - Enhanced Features:\n"
                msg += "  • Feature extraction: FAILED\n"
                msg += "  • Using basic dataset only\n\n"
                msg += "📁 MAIN OUTPUT: Basic merged dataset\n"
        else:
            msg += "📁 MAIN OUTPUT: Basic merged dataset\n"
        
        msg += "\nCheck output folder for all results!"
        return msg
    
    def _copy_final_outputs(self, data_root, output_path, results, enhanced_results=None):
        """Copy final pipeline outputs to the main output folder"""
        try:
            # Determine which dataset to use as main output
            main_dataset_copied = False
            
            # Copy enhanced dataset if available (this becomes the main output)
            if enhanced_results and enhanced_results.get('success'):
                enhanced_file = Path(enhanced_results['enhanced_dataset_path'])
                if enhanced_file.exists():
                    shutil.copy2(enhanced_file, output_path / "final_dataset_enhanced.csv")
                    # Also copy as main final dataset
                    shutil.copy2(enhanced_file, output_path / "final_dataset.csv")
                    self.log_message(f"Enhanced final dataset saved: {output_path / 'final_dataset_enhanced.csv'}", "success")
                    self.log_message(f"Main output: {output_path / 'final_dataset.csv'} (enhanced version)", "success")
                    main_dataset_copied = True
            
            # Copy basic unified dataset (always available as backup)
            unified_file = data_root / "unified_dataset.csv"
            if unified_file.exists():
                shutil.copy2(unified_file, output_path / "final_dataset_basic.csv")
                self.log_message(f"Basic dataset saved: {output_path / 'final_dataset_basic.csv'}", "success")
                
                # If enhanced dataset wasn't copied, use basic as main
                if not main_dataset_copied:
                    shutil.copy2(unified_file, output_path / "final_dataset.csv")
                    self.log_message(f"Main output: {output_path / 'final_dataset.csv'} (basic version)", "success")
            
            # Copy processed station files
            processed_dir = data_root / "processed_data"
            if processed_dir.exists():
                output_processed = output_path / "station_datasets"
                if output_processed.exists():
                    shutil.rmtree(output_processed)
                shutil.copytree(processed_dir, output_processed)
                self.log_message(f"Station datasets saved: {output_processed}", "success")
            
            # Copy cropped images (optional - can be large)
            cropped_dir = data_root / "cropped_data"
            if cropped_dir.exists():
                # Only copy a sample or create a summary
                try:
                    output_cropped = output_path / "cropped_images_sample"
                    output_cropped.mkdir(exist_ok=True)
                    
                    # Copy first few images from each station as samples
                    sample_count = 0
                    max_samples = 50  # Limit to avoid huge copies
                    
                    for station_dir in cropped_dir.iterdir():
                        if station_dir.is_dir() and sample_count < max_samples:
                            station_sample = output_cropped / station_dir.name
                            station_sample.mkdir(exist_ok=True)
                            
                            # Copy first 3 images from this station
                            for img_file in list(station_dir.glob("*.tif"))[:3]:
                                shutil.copy2(img_file, station_sample / img_file.name)
                                sample_count += 1
                                if sample_count >= max_samples:
                                    break
                    
                    self.log_message(f"Cropped images sample saved: {output_cropped} ({sample_count} files)", "info")
                except Exception as e:
                    self.log_message(f"Could not copy cropped images sample: {e}", "warning")
            
            # Copy log files
            log_files = list(data_root.glob("*.log"))
            if log_files:
                for log_file in log_files:
                    shutil.copy2(log_file, output_path / log_file.name)
                self.log_message(f"Log files copied to output folder", "info")
            
            # Create summary file
            self._create_output_summary(output_path, results, enhanced_results)
        
        except Exception as e:
            self.log_message(f"Warning: Could not copy some output files: {e}", "warning")
    
    def _create_output_summary(self, output_path, results, enhanced_results=None):
        """Create a summary file with pipeline execution details"""
        try:
            summary_file = output_path / "AEP_Pipeline_Summary.txt"
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("AEP 3.0 Air Emissions Prediction Pipeline - Execution Summary\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Stage 1 Summary
                f.write("STAGE 1 - Main Pipeline (Cropping + Merging):\n")
                f.write("-" * 50 + "\n")
                f.write(f"Stations processed: {results.get('stations_processed', 0)}\n")
                f.write(f"Stations failed: {results.get('stations_failed', 0)}\n")
                f.write(f"Images cropped: {results.get('images_cropped', 0):,}\n")
                f.write(f"Features extracted: {results.get('features_extracted', 0):,}\n")
                f.write(f"Records merged: {results.get('records_merged', 0):,}\n")
                
                if results.get('failed_stations'):
                    f.write(f"Failed stations: {', '.join(results['failed_stations'])}\n")
                
                f.write(f"Duration: {results.get('duration_minutes', 0):.2f} minutes\n\n")
                
                # Stage 2 Summary
                f.write("STAGE 2 - Enhanced Feature Extraction:\n")
                f.write("-" * 50 + "\n")
                
                if enhanced_results:
                    if enhanced_results.get('success'):
                        f.write("Status: SUCCESS\n")
                        f.write("Enhanced dataset: CREATED\n")
                        f.write(f"Features cached: {enhanced_results.get('feature_cache_size', 0)}\n")
                        f.write(f"Enhanced dataset path: {enhanced_results.get('enhanced_dataset_path', 'N/A')}\n")
                    else:
                        f.write("Status: FAILED\n")
                        f.write(f"Error: {enhanced_results.get('error', 'Unknown error')}\n")
                else:
                    f.write("Status: SKIPPED (Enhanced pipeline not available)\n")
                
                f.write("\n")
                
                # Output Files Summary
                f.write("OUTPUT FILES:\n")
                f.write("-" * 50 + "\n")
                f.write("final_dataset.csv - Main output dataset\n")
                
                if enhanced_results and enhanced_results.get('success'):
                    f.write("  └─ Enhanced version with additional image features\n")
                    f.write("final_dataset_enhanced.csv - Enhanced dataset (same as main)\n")
                    f.write("final_dataset_basic.csv - Basic dataset (backup)\n")
                else:
                    f.write("  └─ Basic version (cropping + merging only)\n")
                    f.write("final_dataset_basic.csv - Basic dataset (same as main)\n")
                
                f.write("station_datasets/ - Individual station CSV files\n")
                f.write("cropped_images_sample/ - Sample of cropped images\n")
                f.write("*.log - Pipeline execution logs\n")
                f.write("AEP_Pipeline_Summary.txt - This summary file\n\n")
                
                # Usage Instructions
                f.write("USAGE INSTRUCTIONS:\n")
                f.write("-" * 50 + "\n")
                f.write("1. Use 'final_dataset.csv' as your main dataset for analysis\n")
                f.write("2. Individual station data is available in 'station_datasets/'\n")
                f.write("3. Check log files for detailed execution information\n")
                f.write("4. Cropped images sample shows the processing results\n\n")
                
                # Technical Details
                f.write("TECHNICAL DETAILS:\n")
                f.write("-" * 50 + "\n")
                f.write("Pipeline Version: AEP 3.0\n")
                f.write("Stage 1: Image cropping, feature extraction, CPCB data merging\n")
                
                if enhanced_results and enhanced_results.get('success'):
                    f.write("Stage 2: Enhanced feature extraction with per-band matching\n")
                    f.write("  - TIR1 and WV bands processed separately\n")
                    f.write("  - Nearest timestamp matching (±15 min tolerance)\n")
                    f.write("  - Non-lossy approach (all CPCB rows preserved)\n")
                    f.write("  - Feature caching for efficiency\n")
                else:
                    f.write("Stage 2: Not executed (enhanced pipeline unavailable/failed)\n")
            
            self.log_message(f"Pipeline summary saved: {summary_file}", "info")
            
        except Exception as e:
            self.log_message(f"Could not create summary file: {e}", "warning")
    
    def _update_ui_running(self, running):
        """Update UI elements based on running state"""
        def update():
            if running:
                self.run_button.config(state="disabled")
                self.stop_button.config(state="normal")
                self.progress_var.set("Pipeline running...")
                self.status_label.config(foreground="orange")
                self.progress_bar.start()
            else:
                self.run_button.config(state="normal")
                self.stop_button.config(state="disabled")
                self.progress_var.set("Ready")
                self.status_label.config(foreground="green")
                self.progress_bar.stop()
        
        if threading.current_thread() == threading.main_thread():
            update()
        else:
            self.root.after(0, update)
    
    def stop_pipeline(self):
        """Stop the running pipeline (graceful termination)"""
        if self.is_running:
            self.log_message("Stop requested - pipeline will finish current operation...", "warning")
            # Note: Graceful stopping would require pipeline modification
            # For now, just update UI
            self.is_running = False
            self._update_ui_running(False)


def main():
    """Main function to run the GUI application"""
    try:
        # Create and run the GUI
        root = tk.Tk()
        
        # Set application icon (if available)
        try:
            # You can add an icon file here if desired
            # root.iconbitmap('icon.ico')
            pass
        except:
            pass
        
        app = AEPGUIApp(root)
        
        # Center window on screen
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
        y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
        root.geometry(f"+{x}+{y}")
        
        # Start the GUI event loop
        root.mainloop()
        
    except Exception as e:
        print(f"Failed to start GUI application: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
