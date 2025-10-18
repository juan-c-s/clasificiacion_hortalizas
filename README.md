# Plant Disease Classification - PlantVillage Dataset

A deep learning project for classifying plant diseases using the PlantVillage dataset (108,610 images of 14 plant species with various health conditions).

## Repository Structure

```
clasificiacion_hortalizas/
├── data/
│   ├── plantvillage dataset/          # Raw image dataset (created after extraction)
│   │   ├── color/                     # Color images (54,305)
│   │   ├── grayscale/                 # Grayscale images (54,305)
│   │   └── segmented/                 # Segmented images
│   └── plantvillage_images_metadata.parquet  # Image metadata catalog (108,610 rows)
├── preprocessing/
│   ├── extract_data.ipynb             # Step 1: Download dataset from Kaggle
│   ├── process_data.ipynb             # Step 2: Generate metadata parquet file
│   ├── eda.ipynb                      # Step 3: Exploratory data analysis
│   └── generate_data_split.ipynb      # Step 4: Create train/val/test splits
├── pyproject.toml                     # Python project configuration
├── uv.lock                            # Dependency lock file
└── README.md                          # This file
```

## Getting Started

**Important:** Run the notebooks in order:

1. **`extract_data.ipynb`** - Downloads the PlantVillage dataset from Kaggle and places it in `data/plantvillage dataset/`
2. **`process_data.ipynb`** - Scans all images and generates the metadata parquet file
3. **`eda.ipynb`** - Performs exploratory data analysis with visualizations
4. **`generate_data_split.ipynb`** - Creates train/validation/test splits

## Metadata Table: `plantvillage_images_metadata.parquet`

This file contains metadata for all 108,610 images in the dataset.

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

- **Total Images**: 108,610
- **Plant Types**: 14 (Tomato, Orange, Soybean, Grape, Corn, Apple, Peach, Pepper, Potato, Cherry, Squash, Strawberry, Blueberry, Raspberry)
- **Conditions**: 21 (1 healthy + 20 disease types)
- **Image Dimensions**: 256×256 pixels (standardized)
- **Healthy vs Diseased**: 27.8% healthy, 72.2% diseased

## Requirements

```bash
uv sync
```


## License

PlantVillage dataset: [Kaggle Dataset License](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
