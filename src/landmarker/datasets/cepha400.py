"""
This module contains the functions to load the ISBI 2015 cephalometric landmark detection challenge
dataset.
"""

import os
import zipfile
from typing import Callable, Optional

import numpy as np
import opendatasets as od  # type: ignore
import pandas as pd  # type: ignore
import rarfile  # type: ignore

from landmarker.data.landmark_dataset import HeatmapDataset, LandmarkDataset


def get_cepha_dataset_kaggle(path_dir: str):
    """
    Returns the paths to the images and the landmarks of the cephalogram dataset.

    Args:
        path_dir (str): The path to the directory where the dataset should be stored.
    """
    if not os.path.exists(path_dir + "/ISBI2015-kaggle/train_senior.csv"):
        od.download("https://www.kaggle.com/datasets/jiahongqian/cephalometric-landmarks", path_dir)
        # Change the name of the folder to ISBI2015-kaggle
        os.rename(path_dir + "/cephalometric-landmarks", path_dir + "/ISBI2015-kaggle")
    df_train_senior = pd.read_csv(path_dir + "/ISBI2015-kaggle/train_senior.csv")
    df_train_senior["image_path"] = (
        path_dir + "/ISBI2015-kaggle/cepha400/cepha400/" + df_train_senior["image_path"]
    )
    landmarks_train = np.flip(
        df_train_senior.drop(columns=["image_path"]).to_numpy().reshape(-1, 19, 2), axis=-1
    )
    df_test1_senior = pd.read_csv(path_dir + "/ISBI2015-kaggle/test1_senior.csv")
    df_test1_senior["image_path"] = (
        path_dir + "/ISBI2015-kaggle/cepha400/cepha400/" + df_test1_senior["image_path"]
    )
    landmarks_test1 = np.flip(
        df_test1_senior.drop(columns=["image_path"]).to_numpy().reshape(-1, 19, 2), axis=-1
    )
    df_test2_senior = pd.read_csv(path_dir + "/ISBI2015-kaggle/test2_senior.csv")
    df_test2_senior["image_path"] = (
        path_dir + "/ISBI2015-kaggle/cepha400/cepha400/" + df_test2_senior["image_path"]
    )
    landmarks_test2 = np.flip(
        df_test2_senior.drop(columns=["image_path"]).to_numpy().reshape(-1, 19, 2), axis=-1
    )

    pixel_spacings_train = np.array([[0.1, 0.1]]).repeat(len(landmarks_train), axis=0)
    pixel_spacings_test1 = np.array([[0.1, 0.1]]).repeat(len(landmarks_test1), axis=0)
    pixel_spacings_test2 = np.array([[0.1, 0.1]]).repeat(len(landmarks_test2), axis=0)

    return (
        df_train_senior["image_path"].to_list(),
        df_test1_senior["image_path"].to_list(),
        df_test2_senior["image_path"].to_list(),
        landmarks_train,
        landmarks_test1,
        landmarks_test2,
        pixel_spacings_train,
        pixel_spacings_test1,
        pixel_spacings_test2,
    )


def get_cepha_dataset(path_dir: str, junior: bool = False, cv: bool = True):
    """Returns the paths to the images and the landmarks of the CEPH dataset from the ISBI 2014 &
    2015 challenges. But not from the kaggle dataset but from this repository:
        https://figshare.com/s/37ec464af8e81ae6ebbf?file=5466581

    Args:
        path_dir (str): The path to the directory where the dataset should be stored.
        junior (bool, optional): Whether to use the junior or senior annotator. Defaults to False.
        cv (bool, optional): Whether to use the cross validation splits from the paper. Defaults to
            True.
    """
    if not os.path.exists(path_dir + "/ISBI2015"):
        od.download(
            "https://figshare.com/ndownloader/articles/3471833?private_link=37ec464af8e81ae6ebbf",
            path_dir,
        )
        # Change the name of the folder to ISBI2015
        with zipfile.ZipFile(path_dir + "/3471833.zip", "r") as zip_ref:
            zip_ref.extractall(path=path_dir + "/ISBI2015")
        os.remove(path_dir + "/3471833.zip")
        try:
            with rarfile.RarFile(path_dir + "/ISBI2015/AnnotationsByMD.rar", "r") as zip_ref:
                zip_ref.extractall(path=path_dir + "/ISBI2015")
            with rarfile.RarFile(path_dir + "/ISBI2015/RawImage.rar", "r") as zip_ref:
                zip_ref.extractall(path=path_dir + "/ISBI2015")
        except rarfile.BadRarFile:
            raise rarfile.BadRarFile(
                "Possibly you need to install unrar. On linux: sudo apt install unrar"
            )
        os.remove(path_dir + "/ISBI2015/AnnotationsByMD.rar")
        os.remove(path_dir + "/ISBI2015/RawImage.rar")
        os.remove(path_dir + "/ISBI2015/EvaluationCode.rar")
        if cv:
            os.mkdir(path_dir + "/ISBI2015/cv_payer")
            for i in range(1, 5):
                od.download(
                    "https://raw.githubusercontent.com/christianpayer/MedicalDataAugmentationTool"
                    + f"-HeatmapUncertainty/main/setup_ann/all_landmarks/cv/{i}.txt",
                    path_dir + "/ISBI2015/cv_payer",
                )

    if junior:
        annotator = "junior"
    else:
        annotator = "senior"
    landmarks_list = []
    for i in range(400):
        landmarks_list.append(
            pd.read_csv(
                path_dir + f"/ISBI2015/400_{annotator}/{str(i+1).zfill(3)}.txt",
                sep=",",
                header=None,
            )[:19].to_numpy()
        )
    landmarks = np.concatenate(landmarks_list, axis=0).reshape((-1, 19, 2))
    landmarks = np.flip(landmarks, axis=-1)
    image_paths_train = [
        path_dir + f"/ISBI2015/RawImage/TrainingData/{str(i).zfill(3)}.bmp" for i in range(1, 151)
    ]
    landmarks_train = landmarks[:150]
    pixel_spacings_train = np.array([[0.1, 0.1]]).repeat(len(landmarks_train), axis=0)
    image_paths_test1 = [
        path_dir + f"/ISBI2015/RawImage/Test1Data/{str(i).zfill(3)}.bmp" for i in range(151, 301)
    ]
    landmarks_test1 = landmarks[150:300]
    pixel_spacings_test1 = np.array([[0.1, 0.1]]).repeat(len(landmarks_test1), axis=0)
    image_paths_test2 = [
        path_dir + f"/ISBI2015/RawImage/Test2Data/{str(i).zfill(3)}.bmp" for i in range(301, 401)
    ]
    landmarks_test2 = landmarks[300:]
    pixel_spacings_test2 = np.array([[0.1, 0.1]]).repeat(len(landmarks_test2), axis=0)
    return (
        image_paths_train,
        image_paths_test1,
        image_paths_test2,
        landmarks_train,
        landmarks_test1,
        landmarks_test2,
        pixel_spacings_train,
        pixel_spacings_test1,
        pixel_spacings_test2,
    )


def get_cepha_landmark_datasets(
    path_dir: str,
    transform: Optional[Callable] = None,
    store_imgs=True,
    dim_img=None,
    kaggle=False,
    junior=False,
    single_dataset=False,
) -> LandmarkDataset | tuple[LandmarkDataset, LandmarkDataset, LandmarkDataset]:
    """Returns a LandmarkDataset objects with the CEPH dataset, a combination of the ISBI 2014 &
    2015 challenges. The dataset is split into train, test1 and test2. The same approach as in
    "CephaNN: A Multi-Head Attention Network for Cephalometric Landmark Detection" - JIAHOONG QIAN
        et al. is used.
    """
    if junior and kaggle:
        raise ValueError("Junior annotator is not available for the kaggle dataset.")
    if kaggle:
        retrieve_data = get_cepha_dataset_kaggle
    else:

        def retrieve_data(path):
            return get_cepha_dataset(path, junior=junior)

    (
        image_paths_train,
        image_paths_test1,
        image_paths_test2,
        landmarks_train,
        landmarks_test1,
        landmarks_test2,
        pixel_spacings_train,
        pixel_spacings_test1,
        pixel_spacings_test2,
    ) = retrieve_data(path_dir)
    if single_dataset:
        return LandmarkDataset(
            image_paths_train + image_paths_test1 + image_paths_test2,
            np.concatenate([landmarks_train, landmarks_test1, landmarks_test2], axis=0),
            pixel_spacing=np.concatenate(
                [pixel_spacings_train, pixel_spacings_test1, pixel_spacings_test2], axis=0
            ),
            transform=transform,
            store_imgs=store_imgs,
            dim_img=dim_img,
        )
    return (
        LandmarkDataset(
            image_paths_train,
            landmarks_train,
            pixel_spacing=pixel_spacings_train,
            transform=transform,
            store_imgs=store_imgs,
            dim_img=dim_img,
        ),
        LandmarkDataset(
            image_paths_test1,
            landmarks_test1,
            pixel_spacing=pixel_spacings_test1,
            transform=None,
            store_imgs=store_imgs,
            dim_img=dim_img,
        ),
        LandmarkDataset(
            image_paths_test2,
            landmarks_test2,
            pixel_spacing=pixel_spacings_test2,
            transform=None,
            store_imgs=store_imgs,
            dim_img=dim_img,
        ),
    )


def get_cepha_heatmap_datasets(
    path_dir: str,
    transform: Optional[Callable] = None,
    sigma: float = 1,
    kaggle: bool = True,
    junior: bool = False,
    single_dataset: bool = False,
    **kwargs,
) -> HeatmapDataset | tuple[HeatmapDataset, HeatmapDataset, HeatmapDataset]:
    """Returns a HeatmapDataset with the ISBI 2015 cephalogram challenge dataset. The dataset is
    split into train, test1 and test2. The same approach as in "CephaNN: A Multi-Head Attention
    Network for Cephalometric Landmark Detection" - JIAHOONG QIAN et al. is used.

    Args:
        path_dir (str): The path to the directory where the dataset should be stored.
        transform (Optional[Callable], optional): A transformation to apply to the images and
            heatmaps. Defaults to None.
        sigma (int, optional): The sigma value for the gaussian kernel. Defaults to 1.
        kaggle (bool, optional): Whether to use the kaggle dataset. Defaults to True.
        junior (bool, optional): Whether to use the junior or senior annotator. Defaults to False.
        single_dataset (bool, optional): Whether to return a single dataset with all images and
            landmarks. Defaults to False.
        **kwargs: Additional keyword arguments for the HeatmapDataset.
    """
    if junior and kaggle:
        raise ValueError("Junior annotator is not available for the kaggle dataset.")
    if kaggle:
        retrieve_data = get_cepha_dataset_kaggle
    else:

        def retrieve_data(path):
            return get_cepha_dataset(path, junior=junior)

    (
        image_paths_train,
        image_paths_test1,
        image_paths_test2,
        landmarks_train,
        landmarks_test1,
        landmarks_test2,
        pixel_spacings_train,
        pixel_spacings_test1,
        pixel_spacings_test2,
    ) = retrieve_data(path_dir)
    if single_dataset:
        return HeatmapDataset(
            image_paths_train + image_paths_test1 + image_paths_test2,
            np.concatenate([landmarks_train, landmarks_test1, landmarks_test2], axis=0),
            pixel_spacing=np.concatenate(
                [pixel_spacings_train, pixel_spacings_test1, pixel_spacings_test2], axis=0
            ),
            transform=transform,
            sigma=sigma,
            **kwargs,
        )
    return (
        HeatmapDataset(
            image_paths_train,
            landmarks_train,
            pixel_spacing=pixel_spacings_train,
            transform=transform,
            sigma=sigma,
            **kwargs,
        ),
        HeatmapDataset(
            image_paths_test1,
            landmarks_test1,
            pixel_spacing=pixel_spacings_test1,
            transform=None,
            sigma=sigma,
            **kwargs,
        ),
        HeatmapDataset(
            image_paths_test2,
            landmarks_test2,
            pixel_spacing=pixel_spacings_test2,
            transform=None,
            sigma=sigma,
            **kwargs,
        ),
    )
