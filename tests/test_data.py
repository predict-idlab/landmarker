"""Test data module."""

import os

import cv2
import numpy as np
import pytest
import torch
from monai.transforms import (
    Compose,
    RandAdjustContrastd,
    RandAffined,
    RandGaussianNoised,
    RandHistogramShiftd,
    RandScaleIntensityd,
    RandStdShiftIntensityd,
)

from src.landmarker.data import LandmarkDataset
from src.landmarker.data.landmark_dataset import HeatmapDataset, MaskDataset

NSIM = 10  # number of simulated images


@pytest.fixture(scope="session", autouse=True)
def setup_data():
    """Create data for tests."""
    # Check if data folder exists and create it if not
    if not os.path.exists("tests/data"):
        os.makedirs("tests/data")
    all_landmarks = []
    all_img_uint8 = []
    for i in range(NSIM):
        img_uint8 = (np.random.rand(64, 64) * 256).astype(np.uint8)
        img_uint16 = (np.random.random((64, 64)) * 65535).astype(np.uint16)
        img_color = (np.random.rand(64, 64, 3) * 256).astype(np.uint8)
        landmarks = np.random.rand(8, 2)
        landmarks = landmarks * 64
        cv2.imwrite(f"tests/data/{i}_img_uint8.png", img_uint8)
        cv2.imwrite(
            f"tests/data/{i}_img_uint16.png",
            img_uint16,
        )
        cv2.imwrite(f"tests/data/{i}_img_color.png", img_color)
        all_landmarks.append(landmarks)
        all_img_uint8.append(img_uint8.reshape((1, 64, 64)))

    all_landmarks = np.stack(all_landmarks)

    np.save("tests/data/landmarks.npy", all_landmarks)
    np.save("all_img_uint8.npy", all_img_uint8)

    # define global variables
    pytest.landmarks = all_landmarks
    pytest.all_img_uint8 = all_img_uint8

    def clear_data():
        """Remove data for tests."""
        for i in range(NSIM):
            os.remove(f"tests/data/{i}_img_uint8.png")
            os.remove(f"tests/data/{i}_img_uint16.png")
            os.remove(f"tests/data/{i}_img_color.png")
        os.remove("tests/data/landmarks.npy")
        os.remove("all_img_uint8.npy")

    yield
    clear_data()


@pytest.fixture(scope="session", autouse=True)
def setup_data_3d():
    """Create data 3d for tests."""
    # Check if data folder exists and create it if not
    if not os.path.exists("tests/data"):
        os.makedirs("tests/data")
    all_landmarks = []
    all_img_uint8 = []
    for i in range(NSIM):
        img_uint8 = (np.random.rand(64, 64, 64) * 256).astype(np.uint8)
        img_uint16 = (np.random.random((64, 64, 64)) * 65535).astype(np.uint16)
        np.save(f"tests/data/{i}_img_3d_uint8.npy", img_uint8)
        np.save(f"tests/data/{i}_img_3d_uint16.npy", img_uint16)
        landmarks = np.random.rand(8, 3)
        landmarks = landmarks * 64
        all_landmarks.append(landmarks)

    all_landmarks = np.stack(all_landmarks)

    np.save("tests/data/landmarks_3d.npy", all_landmarks)

    # define global variables
    pytest.landmarks_3d = all_landmarks
    pytest.all_img_uint8_3d = all_img_uint8

    def clear_data():
        """Remove data for tests."""
        for i in range(NSIM):
            os.remove(f"tests/data/{i}_img_3d_uint8.npy")
            os.remove(f"tests/data/{i}_img_3d_uint16.npy")
        os.remove("tests/data/landmarks_3d.npy")

    yield
    clear_data()


def test_landmark_dataset_stored_imgs():
    """Test LandmarkDatase with store_imgs=True."""
    dataset = LandmarkDataset(
        imgs=[f"tests/data/{i}_img_uint8.png" for i in range(NSIM)],
        landmarks=pytest.landmarks,
        class_names=[f"class_{i}" for i in range(8)],
    )
    assert len(dataset) == len(pytest.landmarks)
    assert dataset[0]["image"].shape == torch.Size([1, 64, 64])  # image
    assert dataset[0]["landmark"].shape == torch.Size([pytest.landmarks.shape[1], 2])  # landmark
    assert dataset[len(pytest.landmarks) - 1]["image"].shape == torch.Size([1, 64, 64])  # image
    assert dataset[len(pytest.landmarks) - 1]["landmark"].shape == torch.Size(
        [pytest.landmarks.shape[1], 2]
    )  # landmark

    dataset = LandmarkDataset(
        imgs=[f"tests/data/{i}_img_uint16.png" for i in range(NSIM)],
        landmarks=pytest.landmarks,
        dim_img=(32, 20),
        pixel_spacing=(0.1, 0.1),
    )
    assert len(dataset) == len(pytest.landmarks)
    assert dataset[0]["image"].shape == torch.Size([1, 32, 20])  # image
    assert dataset[0]["landmark"].shape == torch.Size([pytest.landmarks.shape[1], 2])  # landmark
    assert dataset[len(pytest.landmarks) - 1]["image"].shape == torch.Size([1, 32, 20])  # image
    assert dataset[len(pytest.landmarks) - 1]["landmark"].shape == torch.Size(
        [pytest.landmarks.shape[1], 2]
    )  # landmark

    dataset = LandmarkDataset(
        imgs=[f"tests/data/{i}_img_uint16.png" for i in range(NSIM)],
        landmarks=pytest.landmarks,
        dim_img=(32, 32),
        resize_pad=False,
        pixel_spacing=torch.Tensor([[0.1, 0.1]]).repeat(NSIM, 1),
    )
    assert len(dataset) == len(pytest.landmarks)
    assert dataset[0]["image"].shape == torch.Size([1, 32, 32])  # image
    assert dataset[0]["landmark"].shape == torch.Size([pytest.landmarks.shape[1], 2])  # landmark
    assert dataset[len(pytest.landmarks) - 1]["image"].shape == torch.Size([1, 32, 32])  # image
    assert dataset[len(pytest.landmarks) - 1]["landmark"].shape == torch.Size(
        [pytest.landmarks.shape[1], 2]
    )  # landmark

    dataset = LandmarkDataset(
        imgs=pytest.all_img_uint8,
        landmarks=pytest.landmarks,
    )
    assert len(dataset) == len(pytest.landmarks)
    assert dataset[0]["image"].shape == torch.Size([1, 64, 64])  # image
    assert dataset[0]["landmark"].shape == torch.Size([pytest.landmarks.shape[1], 2])  # landmark
    assert dataset[len(pytest.landmarks) - 1]["image"].shape == torch.Size([1, 64, 64])  # image
    assert dataset[len(pytest.landmarks) - 1]["landmark"].shape == torch.Size(
        [pytest.landmarks.shape[1], 2]
    )  # landmark

    all_img_uint8_tensor_list = [torch.Tensor(img) for img in pytest.all_img_uint8]
    dataset = LandmarkDataset(
        imgs=all_img_uint8_tensor_list,
        landmarks=pytest.landmarks,
    )
    assert len(dataset) == len(pytest.landmarks)
    assert dataset[0]["image"].shape == torch.Size([1, 64, 64])  # image
    assert dataset[0]["landmark"].shape == torch.Size([pytest.landmarks.shape[1], 2])  # landmark
    assert dataset[len(pytest.landmarks) - 1]["image"].shape == torch.Size([1, 64, 64])  # image
    assert dataset[len(pytest.landmarks) - 1]["landmark"].shape == torch.Size(
        [pytest.landmarks.shape[1], 2]
    )  # landmark

    all_img_uint8_tensor_list = [torch.Tensor(img) for img in pytest.all_img_uint8]
    dataset = LandmarkDataset(
        imgs=torch.stack(all_img_uint8_tensor_list),
        landmarks=torch.Tensor(pytest.landmarks),
    )
    assert len(dataset) == len(pytest.landmarks)
    assert dataset[0]["image"].shape == torch.Size([1, 64, 64])  # image
    assert dataset[0]["landmark"].shape == torch.Size([pytest.landmarks.shape[1], 2])  # landmark
    assert dataset[len(pytest.landmarks) - 1]["image"].shape == torch.Size([1, 64, 64])  # image
    assert dataset[len(pytest.landmarks) - 1]["landmark"].shape == torch.Size(
        [pytest.landmarks.shape[1], 2]
    )  # landmark

    dataset = LandmarkDataset(
        imgs=pytest.all_img_uint8,
        landmarks=pytest.landmarks,
        dim_img=(32, 20),
    )
    assert len(dataset) == len(pytest.landmarks)
    assert dataset[0]["image"].shape == torch.Size([1, 32, 20])  # image
    assert dataset[0]["landmark"].shape == torch.Size([pytest.landmarks.shape[1], 2])  # landmark
    assert dataset[len(pytest.landmarks) - 1]["image"].shape == torch.Size([1, 32, 20])  # image
    assert dataset[len(pytest.landmarks) - 1]["landmark"].shape == torch.Size(
        [pytest.landmarks.shape[1], 2]
    )  # landmark

    dataset = LandmarkDataset(
        imgs=pytest.all_img_uint8, landmarks=pytest.landmarks, dim_img=(32, 32), resize_pad=False
    )
    assert len(dataset) == len(pytest.landmarks)
    assert dataset[0]["image"].shape == torch.Size([1, 32, 32])  # image
    assert dataset[0]["landmark"].shape == torch.Size([pytest.landmarks.shape[1], 2])  # landmark
    assert dataset[len(pytest.landmarks) - 1]["image"].shape == torch.Size([1, 32, 32])  # image
    assert dataset[len(pytest.landmarks) - 1]["landmark"].shape == torch.Size(
        [pytest.landmarks.shape[1], 2]
    )  # landmark


def test_landmark_dataset_not_store_img():
    """Test LandmarkDataset with images and store_imgs=False."""
    dataset = LandmarkDataset(
        imgs=[f"tests/data/{i}_img_uint8.png" for i in range(NSIM)],
        landmarks=pytest.landmarks,
        store_imgs=False,
        dim_img=(32, 20),
        resize_pad=False,
    )
    assert len(dataset) == len(pytest.landmarks)
    assert dataset[0]["image"].shape == torch.Size([1, 32, 20])  # image
    assert dataset[0]["landmark"].shape == torch.Size([pytest.landmarks.shape[1], 2])  # landmark
    assert dataset[len(pytest.landmarks) - 1]["image"].shape == torch.Size([1, 32, 20])  # image
    assert dataset[len(pytest.landmarks) - 1]["landmark"].shape == torch.Size(
        [pytest.landmarks.shape[1], 2]
    )  # landmark

    dataset = LandmarkDataset(
        imgs=[f"tests/data/{i}_img_uint8.png" for i in range(NSIM)],
        landmarks=pytest.landmarks,
        store_imgs=False,
        dim_img=(30, 15),
        resize_pad=True,
    )
    assert len(dataset) == len(pytest.landmarks)
    assert dataset[0]["image"].shape == torch.Size([1, 30, 15])  # image
    assert dataset[0]["landmark"].shape == torch.Size([pytest.landmarks.shape[1], 2])  # landmark
    assert dataset[len(pytest.landmarks) - 1]["image"].shape == torch.Size([1, 30, 15])  # image
    assert dataset[len(pytest.landmarks) - 1]["landmark"].shape == torch.Size(
        [pytest.landmarks.shape[1], 2]
    )  # landmark
    assert torch.allclose(dataset[0]["padding"], torch.Tensor((32, 0)))  # padding


def test_landmark_dataset():
    """Test LandmarkDataset with images."""
    dataset = LandmarkDataset(
        imgs=[f"tests/data/{i}_img_color.png" for i in range(NSIM)],
        landmarks=pytest.landmarks,
    )
    assert len(dataset) == len(pytest.landmarks)
    assert dataset[0]["image"].shape == torch.Size([3, 64, 64])  # image
    assert dataset[0]["landmark"].shape == torch.Size([pytest.landmarks.shape[1], 2])  # landmark

    dataset = LandmarkDataset(
        imgs=[f"tests/data/{i}_img_color.png" for i in range(NSIM)],
        landmarks=pytest.landmarks,
        dim_img=(32, 20),
    )
    assert len(dataset) == len(pytest.landmarks)
    assert dataset[0]["image"].shape == torch.Size([3, 32, 20])  # image
    assert dataset[0]["landmark"].shape == torch.Size([pytest.landmarks.shape[1], 2])  # landmark

    dataset = LandmarkDataset(
        imgs=[f"tests/data/{i}_img_color.png" for i in range(NSIM)],
        landmarks=pytest.landmarks,
        store_imgs=False,
    )
    assert len(dataset) == len(pytest.landmarks)
    assert dataset[0]["image"].shape == torch.Size([3, 64, 64])  # image
    assert dataset[0]["landmark"].shape == torch.Size([pytest.landmarks.shape[1], 2])  # landmark

    dataset = LandmarkDataset(
        imgs=[f"tests/data/{i}_img_color.png" for i in range(NSIM)],
        landmarks=pytest.landmarks,
        dim_img=(32, 20),
        store_imgs=False,
    )
    assert len(dataset) == len(pytest.landmarks)
    assert dataset[0]["image"].shape == torch.Size([3, 32, 20])  # image
    assert dataset[0]["landmark"].shape == torch.Size([pytest.landmarks.shape[1], 2])  # landmark


def test_landmark_dataset_transforms():
    """Test LandmarkDataset with transforms."""
    fn_keys = ("image",)
    spatial_transformd = [
        RandAffined(
            fn_keys,
            prob=0.75,
            rotate_range=(-np.pi / 12, np.pi / 12),
            translate_range=(-100, 100),
            scale_range=(-0.25, 0.25),
        )
    ]

    composed_transformd = Compose(
        spatial_transformd
        +
        # Add gaussian noise
        [
            RandGaussianNoised(("image",), prob=0.1, mean=0, std=0.005),
            # Add random intensity shift
            RandStdShiftIntensityd(("image",), prob=0.1, factors=0.1),
            # Add random intensity scaling
            RandScaleIntensityd(("image",), factors=0.25, prob=0.1),
            RandAdjustContrastd(("image",), prob=0.1, gamma=(0.5, 2)),  # Randomly adjust contrast
            # Randomly shift histogram
            RandHistogramShiftd(("image",), prob=0.1),
        ]
    )
    dataset = LandmarkDataset(
        imgs=[f"tests/data/{i}_img_uint8.png" for i in range(NSIM)],
        landmarks=pytest.landmarks,
        transform=composed_transformd,
    )
    assert len(dataset) == len(pytest.landmarks)
    assert dataset[0]["image"].shape == torch.Size([1, 64, 64])  # image
    assert dataset[0]["landmark"].shape == torch.Size([pytest.landmarks.shape[1], 2])  # landmark
    assert dataset[len(pytest.landmarks) - 1]["image"].shape == torch.Size([1, 64, 64])  # image
    assert dataset[len(pytest.landmarks) - 1]["landmark"].shape == torch.Size(
        [pytest.landmarks.shape[1], 2]
    )  # landmark

    dataset = LandmarkDataset(
        imgs=[f"tests/data/{i}_img_uint8.png" for i in range(NSIM)],
        landmarks=pytest.landmarks,
        transform=composed_transformd,
        dim_img=(32, 20),
    )
    assert len(dataset) == len(pytest.landmarks)
    assert dataset[0]["image"].shape == torch.Size([1, 32, 20])  # image
    assert dataset[0]["landmark"].shape == torch.Size([pytest.landmarks.shape[1], 2])  # landmark
    assert dataset[len(pytest.landmarks) - 1]["image"].shape == torch.Size([1, 32, 20])  # image
    assert dataset[len(pytest.landmarks) - 1]["landmark"].shape == torch.Size(
        [pytest.landmarks.shape[1], 2]
    )  # landmark

    dataset = LandmarkDataset(
        imgs=[f"tests/data/{i}_img_uint8.png" for i in range(NSIM)],
        landmarks=pytest.landmarks,
        transform=composed_transformd,
        store_imgs=False,
    )
    assert len(dataset) == len(pytest.landmarks)
    assert dataset[0]["image"].shape == torch.Size([1, 64, 64])  # image
    assert dataset[0]["landmark"].shape == torch.Size([pytest.landmarks.shape[1], 2])  # landmark
    assert dataset[len(pytest.landmarks) - 1]["image"].shape == torch.Size([1, 64, 64])  # image
    assert dataset[len(pytest.landmarks) - 1]["landmark"].shape == torch.Size(
        [pytest.landmarks.shape[1], 2]
    )  # landmark

    dataset = LandmarkDataset(
        imgs=[f"tests/data/{i}_img_uint8.png" for i in range(NSIM)],
        landmarks=pytest.landmarks,
        transform=composed_transformd,
        dim_img=(32, 20),
        store_imgs=False,
    )
    assert len(dataset) == len(pytest.landmarks)
    assert dataset[0]["image"].shape == torch.Size([1, 32, 20])  # image
    assert dataset[0]["landmark"].shape == torch.Size([pytest.landmarks.shape[1], 2])  # landmark
    assert dataset[len(pytest.landmarks) - 1]["image"].shape == torch.Size([1, 32, 20])  # image
    assert dataset[len(pytest.landmarks) - 1]["landmark"].shape == torch.Size(
        [pytest.landmarks.shape[1], 2]
    )  # landmark


def test_mask_landmark_dataset():
    """Test MaskDataset."""
    dataset = MaskDataset(
        imgs=[f"tests/data/{i}_img_uint8.png" for i in range(NSIM)],
        landmarks=pytest.landmarks,
    )
    assert len(dataset) == len(pytest.landmarks)
    assert dataset[0]["image"].shape == torch.Size([1, 64, 64])  # image
    assert dataset[0]["mask"].shape == torch.Size([pytest.landmarks.shape[1], 64, 64])  # mask
    assert dataset[0]["landmark"].shape == torch.Size([pytest.landmarks.shape[1], 2])  # landmark
    assert dataset[len(pytest.landmarks) - 1]["image"].shape == torch.Size([1, 64, 64])  # image
    assert dataset[len(pytest.landmarks) - 1]["mask"].shape == torch.Size(
        [pytest.landmarks.shape[1], 64, 64]
    )  # mask
    assert dataset[len(pytest.landmarks) - 1]["landmark"].shape == torch.Size(
        [pytest.landmarks.shape[1], 2]
    )  # landmark

    for i, batch in enumerate(dataset):
        cv2.imwrite(f"tests/data/{i}_mask.png", batch["mask"].sum(dim=0).numpy() * 255)
    dataset = MaskDataset(
        imgs=[f"tests/data/{i}_img_uint8.png" for i in range(NSIM)],
        mask_paths=[f"tests/data/{i}_mask.png" for i in range(NSIM)],
    )
    assert len(dataset) == len(pytest.landmarks)
    assert dataset[0]["image"].shape == torch.Size([1, 64, 64])  # image
    assert dataset[0]["mask"].shape == torch.Size([1, 64, 64])  # mask
    assert dataset[0]["landmark"].shape == torch.Size([1, 8, 2])  # landmark
    assert dataset[len(pytest.landmarks) - 1]["image"].shape == torch.Size([1, 64, 64])  # image
    assert dataset[len(pytest.landmarks) - 1]["mask"].shape == torch.Size([1, 64, 64])  # mask
    assert dataset[len(pytest.landmarks) - 1]["landmark"].shape == torch.Size([1, 8, 2])  # landmark

    dataset = MaskDataset(
        imgs=[f"tests/data/{i}_img_uint8.png" for i in range(NSIM)],
        landmarks=pytest.landmarks,
        dim_img=(32, 20),
    )
    assert len(dataset) == len(pytest.landmarks)
    assert dataset[0]["image"].shape == torch.Size([1, 32, 20])  # image
    assert dataset[0]["mask"].shape == torch.Size([pytest.landmarks.shape[1], 32, 20])  # mask
    assert dataset[0]["landmark"].shape == torch.Size([pytest.landmarks.shape[1], 2])  # landmark
    assert dataset[len(pytest.landmarks) - 1]["image"].shape == torch.Size([1, 32, 20])  # image
    assert dataset[len(pytest.landmarks) - 1]["mask"].shape == torch.Size(
        [pytest.landmarks.shape[1], 32, 20]
    )  # mask
    assert dataset[len(pytest.landmarks) - 1]["landmark"].shape == torch.Size(
        [pytest.landmarks.shape[1], 2]
    )  # landmark

    dataset = MaskDataset(
        imgs=[f"tests/data/{i}_img_uint8.png" for i in range(NSIM)],
        mask_paths=[f"tests/data/{i}_mask.png" for i in range(NSIM)],
        dim_img=(32, 20),
    )
    assert len(dataset) == len(pytest.landmarks)
    assert dataset[0]["image"].shape == torch.Size([1, 32, 20])  # image
    assert dataset[0]["mask"].shape == torch.Size([1, 32, 20])  # mask
    assert dataset[0]["landmark"].shape == torch.Size([1, 8, 2])  # landmark
    assert dataset[len(pytest.landmarks) - 1]["image"].shape == torch.Size([1, 32, 20])  # image
    assert dataset[len(pytest.landmarks) - 1]["mask"].shape == torch.Size([1, 32, 20])  # mask
    assert dataset[len(pytest.landmarks) - 1]["landmark"].shape == torch.Size([1, 8, 2])  # landmark

    dataset = MaskDataset(
        imgs=[f"tests/data/{i}_img_uint8.png" for i in range(NSIM)],
        mask_paths=[f"tests/data/{i}_mask.png" for i in range(NSIM)],
        dim_img=(32, 20),
        resize_pad=False,
    )
    assert len(dataset) == len(pytest.landmarks)
    assert dataset[0]["image"].shape == torch.Size([1, 32, 20])  # image
    assert dataset[0]["mask"].shape == torch.Size([1, 32, 20])  # mask
    assert dataset[0]["landmark"].shape == torch.Size([1, 8, 2])  # landmark
    assert dataset[len(pytest.landmarks) - 1]["image"].shape == torch.Size([1, 32, 20])  # image
    assert dataset[len(pytest.landmarks) - 1]["mask"].shape == torch.Size([1, 32, 20])  # mask
    assert dataset[len(pytest.landmarks) - 1]["landmark"].shape == torch.Size([1, 8, 2])  # landmark

    # remove masks
    for i in range(NSIM):
        os.remove(f"tests/data/{i}_mask.png")


def test_mask_landmark_transforms():
    """Test MaskDataset with transforms."""
    fn_keys = ("image",)
    spatial_transformd = [
        RandAffined(
            fn_keys,
            prob=0.75,
            rotate_range=(-np.pi / 12, np.pi / 12),
            translate_range=(-100, 100),
            scale_range=(-0.25, 0.25),
        )
    ]

    composed_transformd = Compose(
        spatial_transformd
        +
        # Add gaussian noise
        [
            RandGaussianNoised(("image",), prob=0.1, mean=0, std=0.005),
            # Add random intensity shift
            RandStdShiftIntensityd(("image",), prob=0.1, factors=0.1),
            # Add random intensity scaling
            RandScaleIntensityd(("image",), factors=0.25, prob=0.1),
            RandAdjustContrastd(("image",), prob=0.1, gamma=(0.5, 2)),  # Randomly adjust contrast
            # Randomly shift histogram
            RandHistogramShiftd(("image",), prob=0.1),
        ]
    )

    dataset = MaskDataset(
        imgs=[f"tests/data/{i}_img_uint8.png" for i in range(NSIM)],
        landmarks=pytest.landmarks,
        transform=composed_transformd,
    )
    assert len(dataset) == len(pytest.landmarks)
    assert dataset[0]["image"].shape == torch.Size([1, 64, 64])  # image
    assert dataset[0]["mask"].shape == torch.Size([pytest.landmarks.shape[1], 64, 64])  # mask
    assert dataset[0]["landmark"].shape == torch.Size([pytest.landmarks.shape[1], 2])  # landmark
    assert dataset[len(pytest.landmarks) - 1]["image"].shape == torch.Size([1, 64, 64])  # image
    assert dataset[len(pytest.landmarks) - 1]["mask"].shape == torch.Size(
        [pytest.landmarks.shape[1], 64, 64]
    )  # mask
    assert dataset[len(pytest.landmarks) - 1]["landmark"].shape == torch.Size(
        [pytest.landmarks.shape[1], 2]
    )  # landmark


def test_heatmap_dataset():
    """Test HeatmapDataset."""
    dataset = HeatmapDataset(
        imgs=[f"tests/data/{i}_img_uint8.png" for i in range(NSIM)],
        landmarks=pytest.landmarks,
        heatmap_fun="gaussian",
    )
    assert len(dataset) == len(pytest.landmarks)
    assert dataset[0]["image"].shape == torch.Size([1, 64, 64])  # image
    assert dataset[0]["mask"].shape == torch.Size([pytest.landmarks.shape[1], 64, 64])  # heatmap
    assert dataset[0]["landmark"].shape == torch.Size([pytest.landmarks.shape[1], 2])  # landmark
    assert dataset[len(pytest.landmarks) - 1]["image"].shape == torch.Size([1, 64, 64])  # image
    assert dataset[len(pytest.landmarks) - 1]["mask"].shape == torch.Size(
        [pytest.landmarks.shape[1], 64, 64]
    )  # heatmap
    assert dataset[len(pytest.landmarks) - 1]["landmark"].shape == torch.Size(
        [pytest.landmarks.shape[1], 2]
    )  # landmark
    dataset = HeatmapDataset(
        imgs=[f"tests/data/{i}_img_uint8.png" for i in range(NSIM)],
        landmarks=pytest.landmarks,
        heatmap_fun="laplacian",
        dim_img=(20, 32),
        batch_size=0,
    )
    assert len(dataset) == len(pytest.landmarks)
    assert dataset[0]["image"].shape == torch.Size([1, 20, 32])  # image
    assert dataset[0]["mask"].shape == torch.Size([pytest.landmarks.shape[1], 20, 32])  # heatmap
    assert dataset[0]["landmark"].shape == torch.Size([pytest.landmarks.shape[1], 2])  # landmark
    assert dataset[len(pytest.landmarks) - 1]["image"].shape == torch.Size([1, 20, 32])  # image
    assert dataset[len(pytest.landmarks) - 1]["mask"].shape == torch.Size(
        [pytest.landmarks.shape[1], 20, 32]
    )  # heatmap
    assert dataset[len(pytest.landmarks) - 1]["landmark"].shape == torch.Size(
        [pytest.landmarks.shape[1], 2]
    )  # landmark


def test_heatmap_dataset_transforms():
    """Test HeatmapDataset with transforms."""
    fn_keys = ("image", "mask")
    spatial_transformd = [
        RandAffined(
            fn_keys,
            prob=0.75,
            rotate_range=(-np.pi / 12, np.pi / 12),
            translate_range=(-100, 100),
            scale_range=(-0.25, 0.25),
        )
    ]

    composed_transformd = Compose(
        spatial_transformd
        +
        # Add gaussian noise
        [
            RandGaussianNoised(("image",), prob=0.1, mean=0, std=0.005),
            # Add random intensity shift
            RandStdShiftIntensityd(("image",), prob=0.1, factors=0.1),
            # Add random intensity scaling
            RandScaleIntensityd(("image",), factors=0.25, prob=0.1),
            RandAdjustContrastd(("image",), prob=0.1, gamma=(0.5, 2)),  # Randomly adjust contrast
            # Randomly shift histogram
            RandHistogramShiftd(("image",), prob=0.1),
        ]
    )

    dataset = HeatmapDataset(
        imgs=[f"tests/data/{i}_img_uint8.png" for i in range(NSIM)],
        landmarks=pytest.landmarks,
        heatmap_fun="gaussian",
        transform=composed_transformd,
    )
    assert len(dataset) == len(pytest.landmarks)
    assert dataset[0]["image"].shape == torch.Size([1, 64, 64])  # image
    assert dataset[0]["mask"].shape == torch.Size([pytest.landmarks.shape[1], 64, 64])  # heatmap
    assert dataset[0]["landmark"].shape == torch.Size([pytest.landmarks.shape[1], 2])  # landmark
    assert dataset[len(pytest.landmarks) - 1]["image"].shape == torch.Size([1, 64, 64])  # image
    assert dataset[len(pytest.landmarks) - 1]["mask"].shape == torch.Size(
        [pytest.landmarks.shape[1], 64, 64]
    )  # heatmap
    assert dataset[len(pytest.landmarks) - 1]["landmark"].shape == torch.Size(
        [pytest.landmarks.shape[1], 2]
    )  # landmark
    dataset = HeatmapDataset(
        imgs=[f"tests/data/{i}_img_uint8.png" for i in range(NSIM)],
        landmarks=pytest.landmarks,
        heatmap_fun="laplacian",
        dim_img=(20, 32),
        batch_size=0,
        transform=composed_transformd,
    )
    assert len(dataset) == len(pytest.landmarks)
    assert dataset[0]["image"].shape == torch.Size([1, 20, 32])  # image
    assert dataset[0]["mask"].shape == torch.Size([pytest.landmarks.shape[1], 20, 32])  # heatmap
    assert dataset[0]["landmark"].shape == torch.Size([pytest.landmarks.shape[1], 2])  # landmark
    assert dataset[len(pytest.landmarks) - 1]["image"].shape == torch.Size([1, 20, 32])  # image
    assert dataset[len(pytest.landmarks) - 1]["mask"].shape == torch.Size(
        [pytest.landmarks.shape[1], 20, 32]
    )  # heatmap
    assert dataset[len(pytest.landmarks) - 1]["landmark"].shape == torch.Size(
        [pytest.landmarks.shape[1], 2]
    )  # landmark
