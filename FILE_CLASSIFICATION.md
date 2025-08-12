# AEP 3.0 Pipeline File Classification

## 🟢 ESSENTIAL FILES (KEEP - 7 files)

### Main Pipeline
- `robust_aep_pipeline_final.py` ⭐ **RECOMMENDED MAIN PIPELINE**
- `aep_31_complete_pipeline.py` ⭐ **ALTERNATIVE MAIN PIPELINE**

### Core Modules
- `modules/metadata_parser.py` ⭐ **ESSENTIAL**
- `modules/satellite_cropper.py` ⭐ **ESSENTIAL**  
- `modules/data_aligner.py` ⭐ **ESSENTIAL**
- `modules/utils.py` ⭐ **ESSENTIAL**
- `modules/__init__.py` ⭐ **ESSENTIAL**

---

## 🟡 USEFUL FILES (KEEP - 6 files)

### Feature Extraction
- `enhanced_feature_pipeline_v2.py` 🟡 **ADVANCED FEATURES**
- `feature_extraction_pipeline.py` 🟡 **BASIC FEATURES**

### Utilities & Testing
- `test_cpcb_loading.py` 🟡 **TESTING UTILITY**
- `analyze_pm25_quality.py` 🟡 **ANALYSIS UTILITY**
- `create_unified_dataset.py` 🟡 **DATASET UTILITY**
- `verification.py` 🟡 **DEBUGGING UTILITY**

---

## 🔴 REDUNDANT FILES (REMOVE - 10 files)

### Duplicate Pipeline Implementations
- `aep_31_simple.py` 🔴 **REMOVE - Redundant (29,782 lines)**
- `robust_aep_complete.py` 🔴 **REMOVE - Superseded**
- `robust_aep_pipeline.py` 🔴 **REMOVE - Old version**
- `run_robust_pipeline.py` 🔴 **REMOVE - Wrapper script**
- `robust_pipeline_methods.py` 🔴 **REMOVE - Methods integrated**
- `fixed_robust_pipeline.py` 🔴 **REMOVE - Superseded**

### Legacy/Incomplete Files
- `aep_pipeline.py` 🔴 **REMOVE - Legacy**
- `test_pipeline.py` 🔴 **REMOVE - Incomplete**
- `full_pipeline_final_version.py` 🔴 **REMOVE - Misleading name**
- `debug_parser.py` 🔴 **REMOVE - Empty file**

---

## 🔵 OPTIONAL FILES (KEEP IF NEEDED - 7 files)

### Documentation & Examples
- `example_usage.py` 🔵 **KEEP - Documentation**

### Module Extensions (Check if used)
- `modules/dataset_merger.py` 🔵 **Check usage**
- `modules/aep_feature_extractor.py` 🔵 **Check usage**

### INSAT Analysis Tools
- `insat_image_report/visualize_insat_gaps.py` 🔵 **Analysis tool**
- `insat_image_report/simulate_files_from_Extras.py` 🔵 **Utility**
- `insat_image_report/scan_extra.py` 🔵 **Utility**
- `insat_image_report/clean_INSAT.py` 🔵 **Utility**

---

## 📋 SUMMARY

- **Total Files**: 30 Python files
- **Keep (Essential + Useful)**: 13 files (43%)
- **Remove (Redundant)**: 10 files (33%)
- **Optional**: 7 files (24%)

## 🚀 RECOMMENDED ACTIONS

### 1. Immediate Cleanup (Remove these 10 files):
```bash
rm aep_31_simple.py
rm robust_aep_complete.py
rm robust_aep_pipeline.py
rm run_robust_pipeline.py
rm robust_pipeline_methods.py
rm fixed_robust_pipeline.py
rm aep_pipeline.py
rm test_pipeline.py
rm full_pipeline_final_version.py
rm debug_parser.py
```

### 2. Main Pipeline Usage:
- **Primary**: `robust_aep_pipeline_final.py`
- **Alternative**: `aep_31_complete_pipeline.py`

### 3. Feature Extraction:
- **Advanced**: `enhanced_feature_pipeline_v2.py`
- **Basic**: `feature_extraction_pipeline.py`

### 4. Testing & Debugging:
- `test_cpcb_loading.py` - Test CPCB data loading
- `analyze_pm25_quality.py` - Analyze data quality
- `verification.py` - Debug timestamp matching

This cleanup will reduce your codebase from 30 to 20 files, eliminating confusion and focusing on the essential components.
