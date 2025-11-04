# Tomato Disease Classification

A deep learning project for classifying tomato plant diseases using the PlantVillage dataset. This project focuses specifically on the tomato image subset (36,320 images with 10 different health conditions) from the full PlantVillage dataset.

## Repository Structure

```
clasificiacion_hortalizas/
├── data/
│   ├── plantvillage dataset/          # Raw image dataset (created after extraction)
│   │   ├── color/                     # Color images (54,305)
│   │   ├── grayscale/                 # Grayscale images (54,305)
│   │   └── segmented/                 # Segmented images
│   ├── plantvillage_images_metadata.parquet  # Image metadata catalog (108,610 rows)
│   ├── splits/                        # Train/validation/test splits
│   │   ├── train/                     # Training images by class
│   │   ├── validation/                # Validation images by class
│   │   └── test/                      # Test images by class
│   └── preprocessed_variants_2/       # Preprocessed images at different resolutions/qualities
├── preprocessing/
│   ├── extract_data.ipynb             # Step 1: Download dataset from Kaggle
│   ├── process_data.ipynb             # Step 2: Generate metadata parquet file
│   ├── eda.ipynb                      # Step 3: Exploratory data analysis
│   └── preprocess.ipynb               # Step 4: Create preprocessed variants
├── training/
│   ├── model-selection.ipynb          # Experiment 1: Benchmark 6 models (fixed resolution)
│   └── general_model.ipynb            # Experiment 2: Resolution evaluation (selected model)
├── pyproject.toml                     # Python project configuration
├── uv.lock                            # Dependency lock file
└── README.md                          # This file
```

## Project Workflow

```
1. DATA PREPROCESSING
   └─> Extract → Process → EDA → Preprocess Variants
   
2. MODEL TRAINING
   ├─> Experiment 1: Model Selection (6 architectures)
   └─> Experiment 2: Resolution Optimization (best model)
```

## Getting Started

### Data Preprocessing

Run the preprocessing notebooks in order:

1. **`preprocessing/extract_data.ipynb`** - Downloads the PlantVillage dataset from Kaggle
2. **`preprocessing/process_data.ipynb`** - Scans all images and generates the metadata parquet file
3. **`preprocessing/eda.ipynb`** - Performs exploratory data analysis with visualizations
4. **`preprocessing/preprocess.ipynb`** - Creates preprocessed image variants at different resolutions and quality levels

### Model Training

#### Experiment 1: Model Selection (`training/model-selection.ipynb`)
Benchmark of **6 different CNN architectures** at a fixed resolution and image quality:
- Objective: Identify the best-performing model architecture
- Fixed parameters: Resolution and JPEG quality
- Models evaluated: 6 different architectures
- Metrics: Accuracy, Precision, Recall, F1-Score

#### Experiment 2: Resolution Evaluation (`training/general_model.ipynb`)
Evaluation of the **selected model** across different image resolutions:
- Objective: Determine optimal resolution for the best model
- Variable parameters: Image resolution and quality combinations
- Uses the winning architecture from Experiment 1
- Metrics: Performance vs. computational cost trade-offs

## Metadata Table: `plantvillage_images_metadata.parquet`

This file contains metadata for all 108,610 images in the full PlantVillage dataset. **This project uses only the 36,320 tomato images** filtered from this metadata file.

### Columns

| Column Name | Data Type | Description | Example |
|-------------|-----------|-------------|---------|
| `image_path` | string | Relative path to image file | `data/plantvillage dataset/color/Tomato___healthy/image001.JPG` |
| `image_type` | string | Image processing type | `color`, `grayscale` |
| `plant_type` | string | Plant species | `Tomato`, `Apple`, `Corn_(maize)`, `Pepper,_bell` |
| `condition` | string | Health condition or disease | `healthy`, `Bacterial_spot`, `Early_blight`, `Powdery_mildew` |
| `file_size_bytes` | int64 | File size in bytes | Range: 3,284 - 121,650 |
| `width` | int64 | Image width in pixels | 256 (all images) |
| `height` | int64 | Image height in pixels | 256 (all images) |
| `file_size_kb` | float64 | File size in kilobytes | Mean: 14.99 KB |
| `file_size_mb` | float64 | File size in megabytes | Calculated from bytes |

### Usage Example

```python
import pandas as pd

# Load metadata
df = pd.read_parquet('data/plantvillage_images_metadata.parquet')

# View basic info
print(f"Total images: {len(df):,}")
print(f"Plant types: {df['plant_type'].nunique()}")
print(f"Conditions: {df['condition'].nunique()}")
```

## Dataset Overview

### Full PlantVillage Dataset (Kaggle)
- **Total Images**: 108,610
- **Plant Types**: 14 (Tomato, Orange, Soybean, Grape, Corn, Apple, Peach, Pepper, Potato, Cherry, Squash, Strawberry, Blueberry, Raspberry)
- **Conditions**: 21 (1 healthy + 20 disease types)
- **Image Dimensions**: 256×256 pixels (standardized)
- **Healthy vs Diseased**: 27.8% healthy, 72.2% diseased

### Tomato Subset (Used in This Project)
- **Total Images**: 36,320 (33% of full dataset)
- **Classes**: 10 conditions
  - Healthy
  - Bacterial_spot
  - Early_blight
  - Late_blight
  - Leaf_Mold
  - Septoria_leaf_spot
  - Spider_mites (Two-spotted spider mite)
  - Target_Spot
  - Tomato_mosaic_virus
  - Tomato_Yellow_Leaf_Curl_Virus
- **Split**: Train (70%), Validation (15%), Test (15%)

## Requirements

```bash
uv sync
```


## License

PlantVillage dataset: [Kaggle Dataset License](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
