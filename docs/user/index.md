# 🚀 Getting Started

(installation)=
## ⚙️ Installation
`````{tab-set}
````{tab-item} pip
```bash
pip install landmarker
```
````
`````

## Overview
Landmarker provides a simple API for training and evaluating landmark detection models. The API is designed to be easy to use and to be flexible such that it can be used intertwined with other libraries or custom code. Landmarker can also act as utility library for certain components of landmark detection algorithms. For example, it provides a set of loss functions and heatmap decoding operations that can be used in combination with other PyTorch-based libraries.

In the following sections, we provide a brief overview of the library and the API. For more specific details, we refer to the [API reference](../reference/index).

### 📦 Modules

#### Data Loading, Preprocessing, and Augmentation
The [**landmarker.data**](../reference/data) module contains classes for loading and preprocessing
data. All classes inherit from the
[`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) class such that they
can be used with PyTorch's
[`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) class. There
are three types of datasets:
* [`LandmarkDataset`](../reference/data/#landmark_dataset): A dataset of images and corresponding
    landmarks (landmark coordinates). Is constructed from images and landmarks pairs.
* [`HeatmapDataset`](../reference/data/#heatmap_dataset): A dataset of images and corresponding
    heatmaps representing the associated landmark. Is constructed from images and landmarks pairs.
* [`MaskDataset`](../reference/data/#mask_dataset): A dataset of images and corresponding masks
    (i.e., binary segmentation masks indiciating the location of the landmarks). Can be
    constructed from specified image and landmarks pairs, or from images and masks pairs, because
    often that is how the data is distributed.

Both the `HeatmapDataset` and `MaskDataset` inherit from the `LandmarkDataset` class, and thus also
contain information about the landmarks. The `MaskDataset` can be constructed from specified image
and landmarks pairs, or from images and masks pairs, because often that is how the data is
distributed. The `HeatmapDataset` can be constructed from images and landmarks pairs.

For all three types of datasets images can be provided as a list of paths to stored images, or as a a numpy
arary, torch tensor, list of numpy  arrays or list of torch tensors. Landmarks can be as numpy arrays or torch tensors.
The landmarks are assumed to be in the range of the dimensions of the image. For example, if the image is 256x256, the values of the landmarks are assumed to
be in the range [0, 256].
These landmarks can be provided in three different shapes:
1) (N, D) where N is the number of samples and D is the number of dimensions
2) (N, C, D) where C is the number of landmark classes
3) (N, C, I, D) where I is the number of instances per landmark class, if less than I instances are
    provided, the remaining instances are filled with NaNs.

##### Preprocessing
If the images are provided as a list of paths, the images loaded from these paths and their values
are normalized to the range [0, 1]. If the images are provided as numpy arrays or torch tensors,
they are assumed to be normalized to the range [0, 1]. When a dim argument is provided, the images
and landmarks are rescaled to the specified dimensions. The original dimensions and original
landmarksare stored in the `original_dims` and `original_landmarks` attributes of the dataset. By
default the rescalling is done such that the aspect ratio is preserved, i.e., padding is added to
the image such that the aspect ratio is preserved. If `resize_pad` is set to `False`, the image is
rescaled without preserving the aspect ratio.

##### Augmentation
All datasets can be augmented using the `transforms` argument. Currently the transforms needs to
follow the [monai](https://docs.monai.io/en/latest/transforms.html), and more specifically they need
to be in a compose function.

#### Datasets
Comming soon...

#### Heatmap Generation and Decoding
Comming soon...

#### Loss Functions
Comming soon...

#### Model Architectures
Comming soon...

#### Training and Evaluation
Comming soon...

#### Utilities
Comming soon...
