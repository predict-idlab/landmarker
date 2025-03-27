"""
Utility functions for visualizing data.
"""

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from torch import nn

from landmarker.data import HeatmapDataset, LandmarkDataset, MaskDataset
from landmarker.heatmap.decoder import heatmap_to_coord, heatmap_to_multiple_coord
from landmarker.heatmap.generator import HeatmapGenerator
from landmarker.utils.utils import pixel_to_unit


def inspection_plot(
    ds: LandmarkDataset,
    idx: int | Sequence[int],
    heatmap_generator: Optional[HeatmapGenerator] = None,
    save_path: Optional[str] = None,
    fig_title: str = "Landmark Dataset Inspection Plot",
):
    """
    Plots the transformed image, heatmap, and original image with landmarks for the given dataset
    indices.

    Args:
        ds (LandmarkDataset): Dataset to inspect.
        idx (int | Iterable[int]): Indices of the dataset to inspect.
        heatmap_generator (HeatmapGenerator, optional): Heatmap generator to use. Defaults to None.
        save_path (str, optional): Path to save the plot to. If None, the plot is not saved.
            Defaults to None.
        fig_title (str, optional): Title of the figure. Defaults to "Landmark Dataset Inspection
            Plot".
    """
    if isinstance(idx, int):
        idx = [idx]

    fig = plt.figure(figsize=(15, 5 * len(idx)))
    # fig.suptitle(fig_title)

    subfigs = fig.subfigures(len(idx), 1, squeeze=False).flatten()
    for row, subfig in enumerate(subfigs):
        ds_idx = idx[row]
        batch = ds[ds_idx]
        landmark = batch["landmark"]
        landmarks_original = ds.landmarks_original[ds_idx]
        img_t = batch["image"]
        img = ds.image_loader(ds.img_paths[ds_idx])  # type: ignore
        if not isinstance(ds, (HeatmapDataset, MaskDataset)):
            batch = ds[ds_idx]
            if heatmap_generator:
                heatmap = heatmap_generator(landmark.unsqueeze(0)).squeeze(0)
                assert heatmap is not None, "Heatmap generator must return a heatmap."
            else:
                heatmap = None
        else:
            heatmap = batch["mask"]
        if img_t.shape[0] > 3:  # If more than 3 channels, remove other channels
            img_t = img_t[:3]
            img = img[:3]
        if img_t.shape[0] == 1:
            img_t = img_t[0]
            img = img[0]
            img = (img - img.min()) / (img.max() - img.min()) * 255
        else:
            img_t = img_t.permute(1, 2, 0)
            img = img.permute(1, 2, 0)
        img = img.detach().numpy().astype(np.uint8)
        if heatmap is not None:
            axs = subfig.subplots(nrows=1, ncols=3)
            axs[0].imshow(img_t)
            axs[1].imshow(img_t)
            axs[1].imshow(heatmap.detach().numpy().sum(axis=0), cmap="jet", alpha=0.5)
            if len(img.shape) == 2:
                axs[2].imshow(img, cmap="gray")
            else:
                axs[2].imshow(img)
            if len(landmark.shape) == 3:  # If multiple instances
                for i in range(landmark.shape[0]):
                    axs[0].scatter(landmark[i, :, 1], landmark[i, :, 0], c="r", s=5)
                for i in range(landmarks_original.shape[0]):
                    axs[2].scatter(
                        landmarks_original[i, :, 1], landmarks_original[i, :, 0], c="r", s=5
                    )
            else:
                axs[0].scatter(landmark[:, 1], landmark[:, 0], c="r", s=5)
                axs[2].scatter(landmarks_original[:, 1], landmarks_original[:, 0], c="r", s=5)
            axs[0].set_title("Transformed w/ landmarks")
            axs[1].set_title("Transformed w/ heatmap")
            axs[2].set_title("Original w/ landmarks")

        else:
            axs = subfig.subplots(nrows=1, ncols=2)
            axs[0].imshow(img_t)
            axs[1].imshow(img)
            if len(landmark.shape) == 3:
                for i in range(landmark.shape[0]):
                    axs[0].scatter(landmark[i, :, 1], landmark[i, :, 0], c="r", s=5)
                for i in range(landmarks_original.shape[0]):
                    axs[1].scatter(
                        landmarks_original[i, :, 1], landmarks_original[i, :, 0], c="r", s=5
                    )
            else:
                axs[0].scatter(landmark[:, 1], landmark[:, 0], c="r", s=5)
                axs[1].scatter(landmarks_original[:, 1], landmarks_original[:, 0], c="r", s=5)
            axs[0].set_title("Transformed w/ landmarks")
            axs[1].set_title("Original w/ landmarks")
        # subfig.suptitle(f"Image {ds_idx}")
    if save_path:
        plt.savefig(save_path)
    plt.show()


def prediction_inspect_plot(
    ds: LandmarkDataset,
    model: nn.Module,
    idx: int | Sequence[int],
    activation: nn.Module = nn.Identity(),
    save_path: Optional[str] = None,
    fig_title: str = "Landmark Prediction Model Inspection Plot",
):
    """
    Plots the transformed image, predicted heatmap, and original image with predicted and true
    landmarks for the given dataset indices.

    Args:
        ds (LandmarkDataset): Dataset to inspect.
        model (nn.Module): Model to use for prediction.
        idx (int | Sequence[int]): Indices of the dataset to inspect.
        activation (nn.Module, optional): Activation function to use. Defaults to nn.Identity().
        save_path (str, optional): Path to save the plot to. If None, the plot is not saved.
            Defaults to None.
        fig_title (str, optional): Title of the figure. Defaults to "Landmark Prediction Model
            Inspection Plot".
    """
    if isinstance(idx, int):
        idx = [idx]

    fig = plt.figure(figsize=(15, 5 * len(idx)))
    # fig.suptitle(fig_title)

    # Create len(idx)x1 subfigures
    subfigs = fig.subfigures(len(idx), 1, squeeze=False).flatten()
    for row, subfig in enumerate(subfigs):
        ds_idx = idx[row]
        batch = ds[ds_idx]
        img = ds.image_loader(ds.img_paths[ds_idx])  # type: ignore
        img_t = batch["image"]
        heatmap = activation(model(img_t.unsqueeze(0)))
        landmark_t = batch["landmark"].view((-1, batch["landmark"].shape[-1]))
        landmarks_original = ds.landmarks_original[ds_idx]
        landmarks_original = landmarks_original.view((-1, landmarks_original.shape[-1]))
        pred_landmarks_t = heatmap_to_coord(heatmap, method="local_soft_argmax")
        pred_landmarks = pixel_to_unit(
            pred_landmarks_t,
            pixel_spacing=None,
            dim=img_t.shape[-2:],
            dim_orig=batch["dim_original"],
            padding=batch["padding"],
        ).squeeze(0)
        pred_landmarks_t = pred_landmarks_t.squeeze(0)
        if img_t.shape[0] > 3:  # If more than 3 channels, remove other channels
            img_t = img_t[:3]
            img = img[:3]
        if img_t.shape[0] == 1:
            img_t = img_t[0]
            img = img[0]
            img = (img - img.min()) / (img.max() - img.min()) * 255
        else:
            img_t = img_t.permute(1, 2, 0)
            img = img.permute(1, 2, 0)
        img = img.float()
        img_t = img_t.float()
        heatmap = heatmap.squeeze(0)
        axs = subfig.subplots(nrows=1, ncols=3)
        axs[0].imshow(img_t)
        axs[0].scatter(landmark_t[:, 1], landmark_t[:, 0], c="b", s=5)
        axs[0].scatter(
            pred_landmarks_t.detach().numpy()[:, 1],
            pred_landmarks_t.detach().numpy()[:, 0],
            c="r",
            s=5,
        )
        axs[1].imshow(img_t)
        axs[1].imshow(heatmap.detach().numpy().sum(axis=0), cmap="jet", alpha=0.5)
        if len(img.shape) == 2:
            axs[2].imshow(img, cmap="gray")
        else:
            axs[2].imshow(img)
        axs[2].scatter(landmarks_original[:, 1], landmarks_original[:, 0], c="b", s=5)
        axs[2].scatter(
            pred_landmarks.detach().numpy()[:, 1], pred_landmarks.detach().numpy()[:, 0], c="r", s=5
        )

        axs[0].set_title("Transformed w/ landmarks")
        axs[1].set_title("Transformed w/ heatmap")
        axs[2].set_title("Original w/ landmarks")
        # subfig.suptitle(f"Image {ds_idx}")
    fig.legend(["True", "Predicted"])
    if save_path:
        plt.savefig(save_path)
    plt.show()


def prediction_inspect_plot_multi_instance(
    ds: LandmarkDataset,
    model: nn.Module,
    idx: int | Sequence[int],
    threshold: float = 0.5,
    window: int = 5,
    activation: nn.Module = nn.Identity(),
    save_path: Optional[str] = None,
    fig_title: str = "Landmark Prediction Model Inspection Plot",
):
    if isinstance(idx, int):
        idx = [idx]

    fig = plt.figure(figsize=(15, 5 * len(idx)))
    fig.suptitle(fig_title)

    # Create len(idx)x1 subfigures
    subfigs = fig.subfigures(len(idx), 1, squeeze=False).flatten()
    for row, subfig in enumerate(subfigs):
        ds_idx = idx[row]
        img = ds.image_loader(ds.img_paths[ds_idx])  # type: ignore
        batch = ds[ds_idx]
        img_t = batch["image"]
        landmark_t = batch["landmark"]
        landmarks_original = pixel_to_unit(
            landmark_t,
            pixel_spacing=None,
            dim=img_t.shape[-2:],
            dim_orig=batch["dim_original"],
            padding=batch["padding"],
        )
        dim_orig = batch["dim_original"]
        padding = batch["padding"]
        heatmap = activation(model(img_t.unsqueeze(0)))
        pred_landmarks_t_batch, _ = heatmap_to_multiple_coord(
            heatmap, method="argmax", threshold=threshold, window=window
        )
        pred_landmarks_t = pred_landmarks_t_batch[0]  # type: ignore
        true_landmarks_t = []
        true_landmarks_original = []
        pred_landmarks = []
        for c in range(landmark_t.shape[0]):
            true_landmarks_t_class = []
            true_landmarks_original_class = []
            pred_landmarks_class = []
            for i in range(landmark_t.shape[1]):
                if not landmark_t[c, i].isnan().any():
                    true_landmarks_t_class.append(landmark_t[c, i])
                    true_landmarks_original_class.append(landmarks_original[c, i])
            for i in range(len(pred_landmarks_t[c])):
                pred_landmarks_class.append(
                    pixel_to_unit(
                        pred_landmarks_t[c][i].unsqueeze(0),
                        pixel_spacing=None,
                        dim=img_t.shape[-2:],
                        dim_orig=dim_orig,
                        padding=padding,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
            true_landmarks_t.append(true_landmarks_t_class)
            true_landmarks_original.append(true_landmarks_original_class)
            pred_landmarks.append(pred_landmarks_class)
        if img_t.shape[0] > 3:  # If more than 3 channels, remove other channels
            img_t = img_t[:3]
            img = img[:3]
        if img_t.shape[0] == 1:
            img_t = img_t[0]
            img = img[0]
            img = (img - img.min()) / (img.max() - img.min()) * 255
        else:
            img_t = img_t.permute(1, 2, 0)
            img = img.permute(1, 2, 0)
        img = img.detach().numpy().astype(np.uint8)
        heatmap = heatmap.squeeze(0)
        axs = subfig.subplots(nrows=1, ncols=3)
        axs[0].imshow(img_t)
        axs[1].imshow(img_t)
        axs[1].imshow(heatmap.detach().numpy().sum(axis=0), cmap="jet", alpha=0.5)
        if len(img.shape) == 2:
            axs[2].imshow(img, cmap="gray")
        else:
            axs[2].imshow(img)
        for c in range(len(true_landmarks_t)):
            for i in range(max(len(true_landmarks_t[c]), len(pred_landmarks_t[c]))):
                if i < len(true_landmarks_t[c]):
                    axs[0].scatter(true_landmarks_t[c][i][1], true_landmarks_t[c][i][0], c="b", s=5)
                    axs[2].scatter(
                        true_landmarks_original[c][i][1],
                        true_landmarks_original[c][i][0],
                        c="b",
                        s=5,
                    )
                if i < len(pred_landmarks_t[c]):
                    axs[0].scatter(pred_landmarks_t[c][i][1], pred_landmarks_t[c][i][0], c="r", s=5)
                    axs[2].scatter(pred_landmarks[c][i][1], pred_landmarks[c][i][0], c="r", s=5)

        axs[0].set_title("Transformed w/ landmarks")
        axs[1].set_title("Transformed w/ heatmap")
        axs[2].set_title("Original w/ landmarks")
        # subfig.suptitle(f"Image {ds_idx}")
    fig.legend(["True", "Predicted"])
    if save_path:
        plt.savefig(save_path)
    plt.show()
