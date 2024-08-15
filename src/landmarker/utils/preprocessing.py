"""Preprocessing utils functions for images"""

import glob
import os
from os.path import join

import cv2
import numpy as np
import torch
from tqdm import tqdm  # type: ignore


def normalize(img: np.ndarray) -> np.ndarray:
    """
    Normalize image to 0-1 range

    Args:
        img (np.ndarray): Image to normalize

    Returns:
        np.ndarray: Normalized image
    """
    if img.dtype == np.uint16:
        return cv2.normalize(img, None, 0, 65535, cv2.NORM_MINMAX)  # type: ignore
    if img.dtype == np.uint8:
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)  # type: ignore
    raise TypeError("Image type not supported")


def preprocess_all(folder: str, output_folder: str, fun_name: str) -> None:
    """
    Apply a preprocessing function to all images in a folder.

    Args:
        folder (str): Folder containing images to preprocess
        output_folder (str): Folder to store the preprocessed images
        fun_name (str): Name of the function to apply. (Only normalize supported for now)
    """
    fun = globals()[fun_name]
    for file in tqdm(glob.glob(folder + r"/**/*.png", recursive=True)):
        img = cv2.imread(join(folder, file), -1)
        adj_img = fun(img)
        target_path = join(folder, file)
        target_path = target_path.replace(folder, output_folder)
        if os.path.exists(os.path.dirname(target_path)) is False:
            os.makedirs(os.path.dirname(target_path))
        cv2.imwrite(target_path, adj_img)


def extract_roi(
    imgs: torch.Tensor,
    roi_middle: torch.Tensor,
    landmarks: torch.Tensor,
    size: int | tuple[int, ...] = 256,
    spatial_dims: int = 2,
    ensure_dim=True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract a region of interest around the landmarks in the image.

    Args:
        img (torch.Tensor): Image to extract the region of interest from.
        roi_middle (torch.Tensor): Middle of the region of interest.
        landmark (torch.Tensor): Landmark to transform to the region of interest.
        size (int | tuple[int, int]): Size of the region of interest.
        spatial_dims (int): Number of spatial dimensions of the image.
        ensure_dim (bool): Ensure the output has the same number of dimensions as the input.

    Returns:
        torch.Tensor: Region of interest around the landmarks.
        torch.Tensor: Landmarks in the region of interest.
        torch.Tensor: Upper left corner of the region of interest.
    """
    if spatial_dims == 2:
        return extract_roi_2d(imgs, roi_middle, landmarks, size, ensure_dim)
    elif spatial_dims == 3:
        return extract_roi_3d(imgs, roi_middle, landmarks, size, ensure_dim)
    raise ValueError("spatial_dims must be 2 or 3.")


def extract_roi_2d(
    imgs: torch.Tensor,
    roi_middle: torch.Tensor,
    landmarks: torch.Tensor,
    size: int | tuple[int, ...],
    ensure_dim=True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract a region of interest around the landmarks in the image.

    Args:
        img (torch.Tensor): Image to extract the region of interest from. (C, H, W)
        roi_middle (torch.Tensor): Middle of the region of interest. (2)
        landmarks (torch.Tensor): Landmarks to transform to the region of interest. (2)
        size (int): Size of the region of interest.
        ensure_dim (bool): Ensure the output has the same number of dimensions as the input.

    Returns:
        torch.Tensor: Region of interest around the landmarks.
        torch.Tensor: Landmarks in the region of interest.
        torch.Tensor: Upper left corner of the region of interest.
    """
    # perform checks on the input
    if imgs.dim() != 3:
        raise ValueError("Input image must have 3 dimensions.")
    if landmarks.shape[0] != 2:
        raise ValueError("Landmarks must have 2 dimensions.")
    if roi_middle.shape[0] != 2:
        raise ValueError("ROI middle must have 2 dimensions.")
    # check if tuple is passed
    if isinstance(size, int):
        size = (size, size)
    if imgs.shape[-2] < size[0] or imgs.shape[-1] < size[1]:
        raise ValueError("Size must be smaller than the image.")
    H, W = imgs.shape[-2:]
    roi_height_lower = (roi_middle[0] - size[0] // 2).int()
    roi_height_upper = (roi_middle[0] + size[0] // 2).int()
    roi_width_lower = (roi_middle[1] - size[1] // 2).int()
    roi_width_upper = (roi_middle[1] + size[1] // 2).int()
    if ensure_dim:
        if roi_height_lower < 0:
            roi_height_upper -= roi_height_lower  # type: ignore
            roi_height_lower = 0  # type: ignore
        if roi_height_upper > H:
            roi_height_lower -= roi_height_upper - H  # type: ignore
            roi_height_upper = H  # type: ignore
        if roi_width_lower < 0:
            roi_width_upper -= roi_width_lower  # type: ignore
            roi_width_lower = 0  # type: ignore
        if roi_width_upper > W:
            roi_width_lower -= roi_width_upper - W  # type: ignore
            roi_width_upper = W  # type: ignore
    else:
        torch.clamp(roi_height_lower, 0, H)
        torch.clamp(roi_height_upper, 0, H)
        torch.clamp(roi_width_lower, 0, W)
        torch.clamp(roi_width_upper, 0, W)
    roi_upper_left_point = torch.tensor([roi_height_lower, roi_width_lower])
    landmarks_patch = landmarks - torch.tensor([roi_height_upper - size[0], roi_width_lower])
    return (
        imgs[..., roi_height_lower:roi_height_upper, roi_width_lower:roi_width_upper],
        landmarks_patch,
        roi_upper_left_point,
    )


def extract_roi_3d(
    imgs: torch.Tensor,
    roi_middle: torch.Tensor,
    landmarks: torch.Tensor,
    size: int | tuple[int, ...],
    ensure_dim=True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if imgs.dim() != 5:
        raise ValueError("Input image must have 5 dimensions.")
    if landmarks.dim() != 3:
        raise ValueError("Landmarks must have 3 dimensions.")
    if landmarks.size(1) != 3:
        raise ValueError("Landmarks must have 3 columns.")
    if landmarks.size(0) != imgs.size(0):
        raise ValueError("Number of landmarks must match the batch size.")
    if roi_middle.dim() != 2:
        raise ValueError("ROI middle must have 2 dimensions.")
    if roi_middle.size(1) != 3:
        raise ValueError("ROI middle must have 3 columns.")
    if roi_middle.size(0) != imgs.size(0):
        raise ValueError("Number of ROI middle must match the batch size.")
    if isinstance(size, int):
        size = (size, size, size)
    if size[0] % 2 == 0 or size[1] % 2 == 0 or size[2] % 2 == 0:
        raise ValueError("Size must be an odd number.")
    roi_front_upper_left_point = roi_middle[:, 0] - torch.tensor(size).unsqueeze(0) // 2
    roi_height_lower = roi_middle[:, 0] - size[0] // 2
    roi_height_upper = roi_middle[:, 0] + size[0] // 2
    roi_width_lower = roi_middle[:, 1] - size[1] // 2
    roi_width_upper = roi_middle[:, 1] + size[1] // 2
    roi_depth_lower = roi_middle[:, 2] - size[2] // 2
    roi_depth_upper = roi_middle[:, 2] + size[2] // 2
    landmarks = landmarks - roi_front_upper_left_point
    return (
        imgs[
            :,
            :,
            roi_height_lower:roi_height_upper,
            roi_width_lower:roi_width_upper,
            roi_depth_lower:roi_depth_upper,
        ],
        landmarks,
        roi_front_upper_left_point,
    )
