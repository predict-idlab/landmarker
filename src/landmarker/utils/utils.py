"""Utility functions for the package."""

import glob
import json
import os
from typing import Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm  # type: ignore


def get_paths(folder_path: str, extension: str) -> list[str]:
    """Retrieve all paths with a defined extension files recursively from the folder path.

    Args:
        folder_path (str): path to the folder
        extension (str): file extension

    Returns:
        list[str]: list of paths
    """
    # return all paths with a defined extension files recursively from the folder path
    return glob.glob(folder_path + "/**/*." + extension, recursive=True)


def annotation_to_landmark(json_obj: dict, class_names: list[str]) -> torch.Tensor:
    """
    Convert the annotation of LabelMe to landmarks.

    Args:
        json_obj (dict): annotation in json format
        class_names (list[str]): class names

    Returns:
        torch.Tensor: landmarks
    """
    shapes_list = json_obj["shapes"]
    landmarks = torch.ones((len(class_names), 2)) * torch.nan

    for i, name in enumerate(class_names):
        for shape in shapes_list:
            if shape["label"] == name:
                landmarks[i] = torch.Tensor(shape["points"][0][::-1])
                break
    return landmarks


def annotation_to_landmark_numpy(json_obj: dict, class_names: list[str]) -> np.ndarray:
    """
    Convert the annotation of LabelMe to landmarks (Numpy version).

    Args:
        json_obj (dict): annotation in json format
        class_names (list[str]): class names

    Returns:
        np.ndarray: landmarks
    """
    return annotation_to_landmark(json_obj, class_names).numpy()


def all_annotations_to_landmarks(paths: list[str], class_names: list[str]) -> torch.Tensor:
    """
    Convert all annotations of LabelMe to landmarks.

    Args:
        paths (list[str]): list of paths to the annotations
        class_names (list[str]): class names

    Returns:
        torch.Tensor: landmarks
    """
    landmarks = torch.zeros((len(paths), len(class_names), 2))
    for i, path in enumerate(paths):
        with open(path, encoding="utf8") as f:
            json_obj = json.load(f)
        landmarks[i] = annotation_to_landmark(json_obj, class_names)
    return landmarks


def all_annotations_to_landmarks_numpy(paths: list[str], class_names: list[str]) -> np.ndarray:
    """
    Convert all annotations of LabelMe to landmarks (Numpy version).

    Args:
        paths (list[str]): list of paths to the annotations
        class_names (list[str]): class names

    Returns:
        np.ndarray: landmarks
    """
    return all_annotations_to_landmarks(paths, class_names).numpy()


def get_angle(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, radial=True) -> torch.Tensor:
    """
    Calculate the angle between three points. The angle is calculated in degrees or radians. The
    angle is calculated in the direction of p2 -> p3. If the angle is radial, the angle is between
    0 and 2pi. If the angle is not radial, the angle is between 0 and 360.

    Args:
        p1 (torch.Tensor): first point
        p2 (torch.Tensor): second point
        p3 (torch.Tensor): third point
        radial (bool, optional): whether the angle is radial. Defaults to True.

    Returns:
        torch.Tensor: angle
    """
    p1 = p1.reshape(-1, 2)
    p2 = p2.reshape(-1, 2)
    p3 = p3.reshape(-1, 2)
    angle = torch.acos(
        torch.sum((p2 - p1) * (p3 - p1), 1)
        / torch.sqrt(torch.sum((p2 - p1) ** 2, 1) * torch.sum((p3 - p1) ** 2, 1))
    )
    if radial:
        return angle % (2 * np.pi)
    return angle * 180 / np.pi % 360


def get_angle_numpy(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, radial=True) -> np.ndarray:
    """
    Calculate the angle between three points (Numpy version). See documentation of get_angle.

    Args:
        p1 (np.ndarray): first point
        p2 (np.ndarray): second point
        p3 (np.ndarray): third point
        radial (bool, optional): whether the angle is radial. Defaults to True.

    Returns:
        np.ndarray: angle
    """
    p1 = p1.reshape(-1, 2)
    p2 = p2.reshape(-1, 2)
    p3 = p3.reshape(-1, 2)
    angle = np.arccos(
        np.sum((p2 - p1) * (p3 - p1), 1)
        / np.sqrt(np.sum((p2 - p1) ** 2, 1) * np.sum((p3 - p1) ** 2, 1))
    )
    if radial:
        return angle % (2 * np.pi)
    return angle * 180 / np.pi % 360


def pixel_to_unit(
    landmarks: torch.Tensor,
    pixel_spacing: Optional[torch.Tensor] = None,
    dim: Optional[tuple[int, ...] | torch.Tensor] = None,
    dim_orig: Optional[torch.Tensor] = None,
    padding: Optional[torch.Tensor] = None,
):
    """Convert the landmarks from pixel to unit.

    Args:
        landmarks (torch.Tensor): landmarks
        pixel_spacing (Optional[torch.Tensor], optional): pixel spacing. Defaults to None.
        dim (Optional[tuple[int, ...] | torch.Tensor], optional): image size. Defaults to None.
        dim_orig (Optional[torch.Tensor], optional): original image size. Defaults to None.
        padding (Optional[torch.Tensor], optional): padding. Defaults to None.

    Returns:
        torch.Tensor: landmarks in units
    """
    spatial_dim = landmarks.shape[-1]
    if dim is not None:
        assert len(dim) == spatial_dim, f"dim must have {spatial_dim} elements."
    if dim_orig is not None:
        assert dim_orig.shape[-1] == spatial_dim, f"dim_orig must have {spatial_dim} elements."
    if len(landmarks.shape) == 4:
        added_dim: tuple[int, ...] = (1, 1)  # for multi-instance
    else:
        added_dim = (1,)
    if dim is not None and dim_orig is not None:
        if padding is None:
            padding = torch.zeros_like(dim_orig, device=landmarks.device)
        else:
            assert padding.shape == dim_orig.shape
        if pixel_spacing is None:
            pixel_spacing = torch.ones((1, spatial_dim), device=landmarks.device)
        dim = torch.as_tensor(dim, device=landmarks.device)
        landmarks_unresize = (
            landmarks
            * (dim_orig + 2.0 * padding).reshape((-1, *added_dim, spatial_dim))
            / dim.reshape((-1, *added_dim, spatial_dim))
        )
        landmarks_unpadded = landmarks_unresize - padding.reshape((-1, *added_dim, spatial_dim))
        return landmarks_unpadded * pixel_spacing.reshape((-1, *added_dim, spatial_dim))
    if dim is not None or dim_orig is not None:
        raise ValueError("dim and dim_orig must be both None or both not None.")
    if pixel_spacing is None:
        return landmarks
    return landmarks * pixel_spacing.reshape((-1, *added_dim, spatial_dim))


def pixel_to_unit_numpy(
    landmarks: np.ndarray,
    pixel_spacing: Optional[np.ndarray] = None,
    dim: Optional[tuple[int, ...] | np.ndarray] = None,
    dim_orig: Optional[np.ndarray] = None,
    padding: Optional[np.ndarray] = None,
) -> np.ndarray:
    # TODO: adjust to account for four dimensions (multi-instance)
    """
    Convert the landmarks from pixel to unit (Numpy version).

    Args:
        landmarks (np.ndarray): landmarks
        pixel_spacing (Optional[np.ndarray], optional): pixel spacing. Defaults to None.
        dim (Optional[tuple[int, ...] | np.ndarray], optional): image size. Defaults to None.
        dim_orig (Optional[np.ndarray], optional): original image size. Defaults to None.
        padding (Optional[np.ndarray], optional): padding. Defaults to None.

    Returns:
        np.ndarray: landmarks in units
    """
    spatial_dim = landmarks.shape[-1]
    if dim is not None:
        assert len(dim) == spatial_dim, f"dim must have {spatial_dim} elements."
    if dim_orig is not None:
        assert dim_orig.shape[-1] == spatial_dim, f"dim_orig must have {spatial_dim} elements."
    if len(landmarks.shape) == 4:
        added_dim: tuple[int, ...] = (1, 1)
    else:
        added_dim = (1,)
    if dim is not None and dim_orig is not None:
        if padding is None:
            padding = np.zeros_like(dim_orig)
        else:
            assert padding.shape == dim_orig.shape
        if pixel_spacing is None:
            pixel_spacing = np.ones((1, spatial_dim))
        dim = np.array(dim)
        landmarks_unresize = (
            landmarks
            * (dim_orig + 2.0 * padding).reshape((-1, *added_dim, spatial_dim))
            / dim.reshape((-1, *added_dim, spatial_dim))
        )
        landmarks_unpadded = landmarks_unresize - padding.reshape((-1, *added_dim, spatial_dim))
        return landmarks_unpadded * pixel_spacing.reshape((-1, *added_dim, spatial_dim))
    if dim is not None or dim_orig is not None:
        raise ValueError("dim and dim_orig must be both None or both not None.")
    if pixel_spacing is None:
        return landmarks
    return landmarks * pixel_spacing.reshape((-1, *added_dim, spatial_dim))


def covert_video_to_frames(video_path: str, frames_path: str, zero_fill: int = 6) -> None:
    """Convert a video to frames.

    Args:
        video_path (str): path to the video
        frames_path (str): path to the folder where the frames are saved
        zero_fill (int, optional): number of digits in the frame name. Defaults to 6.

    Returns:
        None
    """
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)
    vidcap = cv2.VideoCapture(video_path)
    count = 0
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    tqdm.write(f"Transform video to frames: {frame_count} frames")
    for _ in tqdm(range(frame_count)):
        count += 1
        success, image = vidcap.read()
        if not success:
            print(f"Error in frame {count}")
            continue
        # save frame as PNG file
        cv2.imwrite(frames_path + f"/img_{str(count).zfill(zero_fill)}.png", image)
