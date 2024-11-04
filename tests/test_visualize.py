"""Test the visualize module."""

import os

import cv2
import numpy as np
import pytest
from matplotlib import pyplot as plt

from landmarker.heatmap.generator import GaussianHeatmapGenerator
from src.landmarker.data import LandmarkDataset
from src.landmarker.data.landmark_dataset import HeatmapDataset, MaskDataset
from src.landmarker.models import ProbSpatialConfigurationNet
from src.landmarker.visualize import inspection_plot, prediction_inspect_plot

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
        cv2.imwrite(f"tests/data/visualize_{i}_img_uint8.png", img_uint8)
        cv2.imwrite(
            f"tests/data/visualize_{i}_img_uint16.png",
            img_uint16,
        )
        cv2.imwrite(f"tests/data/visualize_{i}_img_color.png", img_color)
        all_landmarks.append(landmarks)
        all_img_uint8.append(img_uint8.reshape((1, 64, 64)))

    all_landmarks = np.stack(all_landmarks)

    np.save("tests/data/visualize_landmarks.npy", all_landmarks)
    np.save("visualize_all_img_uint8.npy", all_img_uint8)

    # define global variables
    pytest.landmarks = all_landmarks
    pytest.all_img_uint8 = all_img_uint8

    def clear_data():
        """Remove data for tests."""
        for i in range(NSIM):
            os.remove(f"tests/data/visualize_{i}_img_uint8.png")
            os.remove(f"tests/data/visualize_{i}_img_uint16.png")
            os.remove(f"tests/data/visualize_{i}_img_color.png")
        os.remove("tests/data/visualize_landmarks.npy")
        os.remove("visualize_all_img_uint8.npy")

    yield
    clear_data()


def test_inspection_plot():
    ds = LandmarkDataset(
        imgs=[f"tests/data/visualize_{i}_img_uint8.png" for i in range(NSIM)],
        landmarks=pytest.landmarks,
        class_names=[f"class_{i}" for i in range(8)],
        dim_img=(32, 32),
    )
    ds_heatmap = HeatmapDataset(
        imgs=[f"tests/data/visualize_{i}_img_uint8.png" for i in range(NSIM)],
        landmarks=pytest.landmarks,
        class_names=[f"class_{i}" for i in range(8)],
        dim_img=(32, 32),
    )

    ds_mask = MaskDataset(
        imgs=[f"tests/data/visualize_{i}_img_uint8.png" for i in range(NSIM)],
        landmarks=pytest.landmarks,
        class_names=[f"class_{i}" for i in range(8)],
        dim_img=(32, 32),
    )

    generator = GaussianHeatmapGenerator(ds.nb_landmarks, heatmap_size=(32, 32))

    idx = [2, 3]
    plt.ion()
    inspection_plot(ds, idx=idx, heatmap_generator=None, save_path=None)
    inspection_plot(ds, idx=idx, heatmap_generator=generator, save_path=None)

    idxs = [2, 3, 4]
    inspection_plot(ds, idx=idxs, heatmap_generator=generator, save_path=None)

    inspection_plot(ds, idx=idx, heatmap_generator=generator, save_path="tests/inspection_plot.png")

    os.remove("tests/inspection_plot.png")

    inspection_plot(ds_heatmap, idx=idx, save_path=None)
    inspection_plot(ds_heatmap, idx=idxs, save_path=None)
    inspection_plot(ds_heatmap, idx=idx, save_path="tests/inspection_plot.png")
    os.remove("tests/inspection_plot.png")

    inspection_plot(ds_mask, idx=idx, save_path=None)
    inspection_plot(ds_mask, idx=idxs, save_path=None)
    inspection_plot(ds_mask, idx=idx, save_path="tests/inspection_plot.png")
    os.remove("tests/inspection_plot.png")
    plt.close("all")


def test_prediction_inspect_plot():
    ds = LandmarkDataset(
        imgs=[f"tests/data/visualize_{i}_img_uint8.png" for i in range(NSIM)],
        landmarks=pytest.landmarks,
        class_names=[f"class_{i}" for i in range(8)],
        dim_img=(32, 32),
    )

    model = ProbSpatialConfigurationNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=8,
        sp_image_input=False,
        la_channels=(32, 32, 32),
        la_strides=(2, 2),
    )

    idx = [2, 3]
    plt.ion()
    prediction_inspect_plot(ds, idx=idx, model=model, save_path=None)
    prediction_inspect_plot(ds, idx=idx, model=model, save_path="tests/prediction_inspect_plot.png")
    os.remove("tests/prediction_inspect_plot.png")
    plt.close("all")
