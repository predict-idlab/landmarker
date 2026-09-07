# 🚀 Getting Started

Landmarker is a Python package built on PyTorch that provides a comprehensive toolkit for anatomical landmark localization in 2D/3D medical images. This guide will help you get started with the basic functionality of the package.

## Table of Contents
- [🚀 Getting Started](#-getting-started)
  - [Table of Contents](#table-of-contents)
  - [⚙️ Installation](#️-installation)
  - [Basic Usage](#basic-usage)
    - [1. Loading Data](#1-loading-data)
      - [Option 1: Using LandmarkDataset directly](#option-1-using-landmarkdataset-directly)
      - [Option 2: Using Built-in Datasets](#option-2-using-built-in-datasets)
    - [2. Setting Up Heatmap Generation](#2-setting-up-heatmap-generation)
    - [3. Creating and Training a Model](#3-creating-and-training-a-model)
    - [4. Visualization and Evaluation](#4-visualization-and-evaluation)
  - [Supported Features](#supported-features)
  - [Tips for Best Results](#tips-for-best-results)
  - [Common Issues and Solutions](#common-issues-and-solutions)
  - [Next Steps](#next-steps)

---

(installation)=
## ⚙️ Installation
You can install landmarker using pip:
`````{tab-set}
````{tab-item} pip
```bash
pip install landmarker
```
````
`````
The package requires Python 3.10 or higher.

## Basic Usage

### 1. Loading Data

There are two main ways to load your data into landmarker:

#### Option 1: Using LandmarkDataset directly

```python
from landmarker.data import LandmarkDataset

# Initialize dataset
dataset = LandmarkDataset(
    imgs=image_paths,          # List of paths to your images
    landmarks=landmarks_array, # NumPy array of shape (N, C, D)
                             # N = number of samples
                             # C = number of landmark classes
                             # D = spatial dimensions (2 or 3)
    spatial_dims=2,          # 2 for 2D images, 3 for 3D
    transform=transforms,    # MONAI transforms for preprocessing
    dim_img=(512, 512),     # Target image dimensions
    class_names=names       # List of landmark class names
)
```

#### Option 2: Using Built-in Datasets

```python
from landmarker.dataset import get_cepha_landmark_datasets

# Load the ISBI2015 cephalometric dataset
data_dir = "path/to/data"
train_ds, test1_ds, test2_ds = get_cepha_landmark_datasets(data_dir)
```

### 2. Setting Up Heatmap Generation

For heatmap-based landmark detection, you'll need to set up a heatmap generator:

```python
from landmarker.heatmap import GaussianHeatmapGenerator

generator = GaussianHeatmapGenerator(
    nb_landmarks=19,        # Number of landmarks
    sigmas=3,              # Standard deviation for Gaussian distribution
    learnable=True,        # Enable adaptive heatmap parameters
    heatmap_size=(512, 512) # Output heatmap dimensions
)
```

***Note:** you could also use the heatmapdataset to generate static heatmaps.*

### 3. Creating and Training a Model

Here's an example using the SpatialConfigurationNetwork:

```python
import torch
from landmarker.models import OriginalSpatialConfigurationNet
from landmarker.losses import GaussianHeatmapL2Loss
from torch.utils.data import DataLoader

# Initialize model
model = OriginalSpatialConfigurationNet(
    in_channels=1,    # Number of input channels
    out_channels=19   # Number of landmarks
)

# Set up optimizer
optimizer = torch.optim.SGD([
    {'params': model.parameters(), "weight_decay": 1e-3},
    {'params': heatmap_generator.sigmas},
    {'params': heatmap_generator.rotation}
], lr=1e-6, momentum=0.99, nesterov=True)

# Define loss function
criterion = GaussianHeatmapL2Loss(alpha=5)

# Create data loader
train_loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0
)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(100):
    model.train()
    for batch in train_loader:
        images = batch["image"].to(device)
        landmarks = batch["landmark"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        heatmaps = heatmap_generator(landmarks)
        loss = criterion(outputs, heatmap_generator.sigmas, heatmaps)

        loss.backward()
        optimizer.step()
```

### 4. Visualization and Evaluation

Landmarker provides tools for visualizing your data and model predictions:

```python
from landmarker.visualize import inspection_plot, prediction_inspect_plot

# Visualize dataset samples
inspection_plot(dataset, range(3), heatmap_generator=generator)

# Visualize model predictions
prediction_inspect_plot(test_dataset, model, test_dataset.indices[:3])
```

## Supported Features

- Multiple dataset types: `LandmarkDataset`, `HeatmapDataset`, `MaskDataset`, `PatchDataset`
- Various image formats: NIfTI, DICOM, PNG, JPG, BMP, NPY/NPZ
- Preprocessing and data augmentation through MONAI transformations
- Multiple heatmap generation methods and decoding operations
- Built-in models and loss functions
- Comprehensive evaluation metrics and visualization tools

## Tips for Best Results

1. **Data Preprocessing**: Use MONAI's transformations to normalize and augment your data appropriately for your specific use case.

2. **Model Selection**: Choose between coordinate regression and heatmap regression based on your requirements. Heatmap regression generally yields better performance.

3. **Hyperparameter Tuning**: Experiment with different heatmap parameters (e.g., sigma values) and learning rates to optimize performance.

4. **Validation**: Use the visualization tools regularly during training to ensure your model is learning correctly.

## Common Issues and Solutions

- If your images have different dimensions, make sure to specify `dim_img` in the dataset initialization to resize them consistently.
- For 3D images, remember to set `spatial_dims=3` in the dataset initialization.
- When using learnable heatmap parameters, ensure they're included in the optimizer parameter groups.

## Next Steps

- Explore the [documentation](../reference/index) for detailed API references
- Check out the examples directory in the GitHub repository
- Join the community and contribute to the project

For questions or issues, please contact jef.jonkers@ugent.be or visit the [GitHub repository](https://github.com/predict-idlab/landmarker).
```