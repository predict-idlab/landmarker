"""
From plants to landmarks dataset.

This dataset is a subset of the Plant Centroids dataset, which is a dataset of images of plants
with their corresponding masks indicating the plant stem. This is a multi-instance single-class
landmark dataset, where the landmarks are the centroids of the masks. The dataset is split into a
training and three testing sets (A_W5, B_W5, C_W8). The "images" are stacks of the RGB and NIR
images.

The dataset can be downloaded from http://plantcentroids.cs.uni-freiburg.de/
"""

import os
import pathlib
import zipfile

import cv2
import numpy as np
import opendatasets as od  # type: ignore
import torch
from tqdm import tqdm  # type: ignore

from landmarker.data import HeatmapDataset, LandmarkDataset, MaskDataset
from landmarker.utils import get_paths


def get_plant_centroids_dataset(path_dir: str):
    if not os.path.exists(path_dir + "/plantcentroids"):
        od.download("http://plantcentroids.cs.uni-freiburg.de/dataset/plantcentroids.zip", path_dir)
        with zipfile.ZipFile(path_dir + "/plantcentroids.zip", "r") as zip_ref:
            zip_ref.extractall(path_dir)
        os.rename(path_dir + "/plantcentroids_dataset", path_dir + "/plantcentroids")
        os.remove(path_dir + "/plantcentroids.zip")

    train_mask_paths = []
    for train_folder in ["G_W5", "H_W5", "I_W6", "J_W8", "K_W8"]:
        train_mask_paths += sorted(
            get_paths(path_dir + "/plantcentroids/" + train_folder + "/label_sparse", "png")
        )

    train_img_rgb_paths = []
    for train_folder in ["G_W5", "H_W5", "I_W6", "J_W8", "K_W8"]:
        train_img_rgb_paths += sorted(
            get_paths(path_dir + "/plantcentroids/" + train_folder + "/rgb_undistorted", "png")
        )

    train_img_nir_paths = []
    for train_folder in ["G_W5", "H_W5", "I_W6", "J_W8", "K_W8"]:
        train_img_nir_paths += sorted(
            get_paths(path_dir + "/plantcentroids/" + train_folder + "/nir", "png")
        )

    A_W4_mask_paths = sorted(get_paths(path_dir + "/plantcentroids/A_W4/label_sparse", "png"))
    A_W4_img_rgb_paths = sorted(get_paths(path_dir + "/plantcentroids/A_W4/rgb_undistorted", "png"))
    A_W4_img_nir_paths = sorted(get_paths(path_dir + "/plantcentroids/A_W4/nir", "png"))

    B_W5_mask_paths = sorted(get_paths(path_dir + "/plantcentroids/B_W5/label_sparse", "png"))
    B_W5_img_rgb_paths = sorted(get_paths(path_dir + "/plantcentroids/B_W5/rgb_undistorted", "png"))
    B_W5_img_nir_paths = sorted(get_paths(path_dir + "/plantcentroids/B_W5/nir", "png"))

    C_W8_mask_paths = sorted(get_paths(path_dir + "/plantcentroids/C_W8/label_sparse", "png"))
    C_W8_img_rgb_paths = sorted(get_paths(path_dir + "/plantcentroids/C_W8/rgb_undistorted", "png"))
    C_W8_img_nir_paths = sorted(get_paths(path_dir + "/plantcentroids/C_W8/nir", "png"))

    train_out_imgs = stack_norm(train_img_rgb_paths, train_img_nir_paths, store=True)
    A_out_imgs = stack_norm(A_W4_img_rgb_paths, A_W4_img_nir_paths, store=True)
    B_out_imgs = stack_norm(B_W5_img_rgb_paths, B_W5_img_nir_paths, store=True)
    C_out_imgs = stack_norm(C_W8_img_rgb_paths, C_W8_img_nir_paths, store=True)

    return (
        train_mask_paths,
        train_img_rgb_paths,
        train_out_imgs,
        A_W4_mask_paths,
        A_W4_img_rgb_paths,
        A_out_imgs,
        B_W5_mask_paths,
        B_W5_img_rgb_paths,
        B_out_imgs,
        C_W8_mask_paths,
        C_W8_img_rgb_paths,
        C_out_imgs,
    )


def get_plant_centroids_mask_datasets(
    path_dir, train_transform=None, inference_transform=None, **kwargs
):
    (
        train_mask_paths,
        train_img_rgb_paths,
        train_out_imgs,
        A_W4_mask_paths,
        A_W4_img_rgb_paths,
        A_out_imgs,
        B_W5_mask_paths,
        B_W5_img_rgb_paths,
        B_out_imgs,
        C_W8_mask_paths,
        C_W8_img_rgb_paths,
        C_out_imgs,
    ) = get_plant_centroids_dataset(path_dir)
    test_kwargs = kwargs.copy()
    return (
        MaskDataset(
            train_img_rgb_paths, mask_paths=train_mask_paths, transform=train_transform, **kwargs
        ),
        MaskDataset(
            A_W4_img_rgb_paths,
            mask_paths=A_W4_mask_paths,
            transform=inference_transform,
            **test_kwargs,
        ),
        MaskDataset(
            B_W5_img_rgb_paths,
            mask_paths=B_W5_mask_paths,
            transform=inference_transform,
            **test_kwargs,
        ),
        MaskDataset(
            C_W8_img_rgb_paths,
            mask_paths=C_W8_mask_paths,
            transform=inference_transform,
            **test_kwargs,
        ),
    )


def get_plant_centroids_landmark_datasets(
    path_dir, train_transform=None, inference_transform=None, **kwargs
):
    mask_kwargs = kwargs.copy()
    mask_kwargs["dim_img"] = None
    try:
        mask_kwargs.pop("store_imgs")
    except KeyError:
        pass
    test_kwargs = kwargs.copy()
    mask_ds_trian, mask_ds_A, mask_ds_B, mask_ds_C = get_plant_centroids_mask_datasets(
        path_dir, store_imgs=False, **mask_kwargs
    )
    return (
        LandmarkDataset(
            imgs=mask_ds_trian.img_paths,
            landmarks=mask_ds_trian.landmarks_original,
            transform=train_transform,
            **kwargs,
        ),
        LandmarkDataset(
            imgs=mask_ds_A.img_paths,
            landmarks=mask_ds_A.landmarks_original,
            transform=inference_transform,
            **test_kwargs,
        ),
        LandmarkDataset(
            imgs=mask_ds_B.img_paths,
            landmarks=mask_ds_B.landmarks_original,
            transform=inference_transform,
            **test_kwargs,
        ),
        LandmarkDataset(
            imgs=mask_ds_C.img_paths,
            landmarks=mask_ds_C.landmarks_original,
            transform=inference_transform,
            **test_kwargs,
        ),
    )


def get_plant_centroids_heatmap_datasets(
    path_dir, train_transform=None, inference_transform=None, **kwargs
):
    mask_kwargs = kwargs.copy()
    mask_kwargs["dim_img"] = None
    mask_kwargs.pop("sigma")
    try:
        mask_kwargs.pop("store_imgs")
    except KeyError:
        pass
    test_kwargs = kwargs.copy()
    mask_ds_trian, mask_ds_A, mask_ds_B, mask_ds_C = get_plant_centroids_mask_datasets(
        path_dir, store_imgs=False, **mask_kwargs
    )
    return (
        HeatmapDataset(
            imgs=mask_ds_trian.img_paths,
            landmarks=mask_ds_trian.landmarks_original,
            transform=train_transform,
            **kwargs,
        ),
        HeatmapDataset(
            imgs=mask_ds_A.img_paths,
            landmarks=mask_ds_A.landmarks_original,
            transform=inference_transform,
            **test_kwargs,
        ),
        HeatmapDataset(
            imgs=mask_ds_B.img_paths,
            landmarks=mask_ds_B.landmarks_original,
            transform=inference_transform,
            **test_kwargs,
        ),
        HeatmapDataset(
            imgs=mask_ds_C.img_paths,
            landmarks=mask_ds_C.landmarks_original,
            transform=inference_transform,
            **test_kwargs,
        ),
    )


def stack_norm(img_paths, img_nir_paths, store=False):
    img_stack = []
    for i, img_path in enumerate(tqdm(img_paths)):
        if store:
            img_name = pathlib.Path(img_path).name.replace(".png", ".npy")
            img_stack_path = pathlib.Path(img_path).parents[1].joinpath(f"rgb_nir/{img_name}")
            img_stack.append(img_stack_path.__str__())
            if not os.path.exists(img_stack_path.parents[0]):
                os.makedirs(img_stack_path.parents[0])
            if os.path.exists(img_stack[i]):
                continue
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = img.transpose(2, 0, 1)
        if img.dtype == np.uint16:
            img = torch.tensor(img / 65535.0, dtype=float).view(-1, img.shape[-2], img.shape[-1])
        elif img.dtype == np.uint8:
            img = torch.tensor(img / 255.0, dtype=float).view(-1, img.shape[-2], img.shape[-1])
        else:
            raise ValueError(f"Image type {img.dtype} not supported")
        nir = cv2.imread(img_nir_paths[i], cv2.IMREAD_UNCHANGED)
        if len(nir.shape) == 3:
            nir = nir[:, :, 0]
        if nir.dtype == np.uint16:
            nir = torch.tensor(nir / 65535.0, dtype=float).view(-1, nir.shape[-2], nir.shape[-1])
        elif nir.dtype == np.uint8:
            nir = torch.tensor(nir / 255.0, dtype=float).view(-1, nir.shape[-2], nir.shape[-1])
        else:
            raise ValueError(f"Image type {nir.dtype} not supported")
        if store:
            np.save(
                img_stack[i],
                torch.cat((img, nir), dim=0).numpy(),
            )
        else:
            img_stack.append(torch.cat((img, nir), dim=0))
    return img_stack
