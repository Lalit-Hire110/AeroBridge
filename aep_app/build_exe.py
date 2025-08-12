#!/usr/bin/env python3
"""
Build script for creating AEP 3.0 GUI executable using PyInstaller
This script automates the process of building a standalone .exe file
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def build_executable():
    """Build the AEP GUI application as a standalone executable"""
    
    print("="*60)
    print("AEP 3.0 GUI - Building Executable")
    print("="*60)
    
    # Get current directory
    app_dir = Path(__file__).parent
    parent_dir = app_dir.parent
    
    print(f"App directory: {app_dir}")
    print(f"Parent directory: {parent_dir}")
    
    # Check if required files exist
    app_file = app_dir / "app.py"
    if not app_file.exists():
        print("ERROR: app.py not found!")
        return False
    
    # Check for pipeline files in parent directory
    pipeline_files = [
        parent_dir / "robust_aep_pipeline_final.py",
        parent_dir / "aep_31_complete_pipeline.py"
    ]
    
    pipeline_found = any(f.exists() for f in pipeline_files)
    if not pipeline_found:
        print("ERROR: No pipeline files found in parent directory!")
        return False
    
    # Check for modules directory
    modules_dir = parent_dir / "modules"
    if not modules_dir.exists():
        print("ERROR: modules directory not found in parent directory!")
        return False
    
    # Check for CPCB data
    cpcb_dir = parent_dir / "data" / "cpcb"
    if not cpcb_dir.exists():
        print("WARNING: CPCB data directory not found. App may not work properly.")
    
    # PyInstaller command
    cmd = [
        "pyinstaller",
        "--onefile",                    # Create single executable
        "--windowed",                   # No console window (GUI only)
        "--name=AEP_3.0_Pipeline",      # Executable name
        "--icon=NONE",                  # No icon for now
        "--clean",                      # Clean cache
        "--noconfirm",                  # Overwrite without asking
        
        # Add parent directory to Python path
        f"--add-data={parent_dir};.",
        
        # Add modules directory
        f"--add-data={modules_dir};modules",
        
        # Add CPCB data if exists
        f"--add-data={cpcb_dir};data/cpcb" if cpcb_dir.exists() else "",
        
        # Add pipeline files
        f"--add-data={parent_dir / 'robust_aep_pipeline_final.py'};." if (parent_dir / 'robust_aep_pipeline_final.py').exists() else "",
        f"--add-data={parent_dir / 'aep_31_complete_pipeline.py'};." if (parent_dir / 'aep_31_complete_pipeline.py').exists() else "",
        f"--add-data={parent_dir / 'enhanced_feature_pipeline_v2.py'};." if (parent_dir / 'enhanced_feature_pipeline_v2.py').exists() else "",
        
        # Hidden imports
        "--hidden-import=pandas",
        "--hidden-import=numpy",
        "--hidden-import=rasterio",
        "--hidden-import=PIL",
        "--hidden-import=pytz",
        "--hidden-import=tkinter",
        "--hidden-import=tkinter.ttk",
        "--hidden-import=tkinter.filedialog",
        "--hidden-import=tkinter.messagebox",
        "--hidden-import=tkinter.scrolledtext",
        
        # Main script
        str(app_file)
    ]
    
    # Remove empty strings from command
    cmd = [arg for arg in cmd if arg]
    
    print("\nRunning PyInstaller...")
    print("Command:", " ".join(cmd))
    print()
    
    try:
        # Run PyInstaller
        result = subprocess.run(cmd, cwd=app_dir, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ PyInstaller completed successfully!")
            
            # Check if executable was created
            exe_path = app_dir / "dist" / "AEP_3.0_Pipeline.exe"
            if exe_path.exists():
                print(f"✓ Executable created: {exe_path}")
                print(f"✓ File size: {exe_path.stat().st_size / (1024*1024):.1f} MB")
                
                # Create a simple launcher script
                launcher_script = app_dir / "run_aep.bat"
                with open(launcher_script, 'w') as f:
                    f.write(f'@echo off\n')
                    f.write(f'cd /d "{exe_path.parent}"\n')
                    f.write(f'"{exe_path.name}"\n')
                    f.write(f'pause\n')
                
                print(f"✓ Launcher script created: {launcher_script}")
                
                print("\n" + "="*60)
                print("BUILD COMPLETED SUCCESSFULLY!")
                print("="*60)
                print(f"Executable location: {exe_path}")
                print(f"Launcher script: {launcher_script}")
                print("\nYou can now distribute the .exe file to other computers.")
                print("The executable is self-contained and doesn't require Python installation.")
                return True
            else:
                print("ERROR: Executable not found after build!")
                return False
        else:
            print("ERROR: PyInstaller failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except FileNotFoundError:
        print("ERROR: PyInstaller not found!")
        print("Please install PyInstaller: pip install pyinstaller")
        return False
    except Exception as e:
        print(f"ERROR: Build failed: {e}")
        return False

def clean_build():
    """Clean build directories"""
    app_dir = Path(__file__).parent
    
    dirs_to_clean = ["build", "dist", "__pycache__"]
    files_to_clean = ["*.spec"]
    
    print("Cleaning build directories...")
    
    for dir_name in dirs_to_clean:
        dir_path = app_dir / dir_name
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"✓ Removed {dir_path}")
    
    # Clean spec files
    for spec_file in app_dir.glob("*.spec"):
        spec_file.unlink()
        print(f"✓ Removed {spec_file}")
    
    print("✓ Build directories cleaned")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        clean_build()
    else:
        success = build_executable()
        if not success:
            sys.exit(1)
