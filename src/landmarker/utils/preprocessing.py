"""Preprocessing utils functions for images"""

import glob
import os
from os.path import join

import cv2
import numpy as np
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
