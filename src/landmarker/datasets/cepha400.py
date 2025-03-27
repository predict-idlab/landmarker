"""
This module contains the functions to load the ISBI 2015 cephalometric landmark detection challenge
dataset.
"""

import glob
import os
import zipfile

import numpy as np
import opendatasets as od  # type: ignore
import pandas as pd  # type: ignore
import rarfile  # type: ignore

from landmarker.data.landmark_dataset import (
    HeatmapDataset,
    LandmarkDataset,
    MaskDataset,
    PatchDataset,
    PatchMaskDataset,
)


def get_cepha_dataset(path_dir: str, junior: bool = False, cv: bool = True):
    """Returns the paths to the images and the landmarks of the CEPH dataset from the ISBI 2014 &
    2015 challenges. But not from the kaggle dataset but from this repository:
        https://figshare.com/s/37ec464af8e81ae6ebbf?file=5466581

    Args:
        path_dir (str): The path to the directory where the dataset should be stored.
        junior (bool, optional): Whether to use the junior or the average of senior and junior annotator. Defaults to False.
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
        os.mkdir(path_dir + "/ISBI2015/cv_payer")
        for i in range(1, 5):
            od.download(
                "https://raw.githubusercontent.com/christianpayer/MedicalDataAugmentationTool"
                + f"-HeatmapUncertainty/main/setup_ann/all_landmarks/cv/{i}.txt",
                path_dir + "/ISBI2015/cv_payer",
            )

    landmarks_list = []
    for i in range(400):
        landmarks_list.append(
            pd.read_csv(
                path_dir + f"/ISBI2015/400_junior/{str(i+1).zfill(3)}.txt",
                sep=",",
                header=None,
            )[:19].to_numpy()
        )
    junior_landmarks = np.concatenate(landmarks_list, axis=0).reshape((-1, 19, 2))
    junior_landmarks = np.flip(junior_landmarks, axis=-1)
    if junior:
        landmarks = junior_landmarks
    else:
        landmarks_list = []
        for i in range(400):
            landmarks_list.append(
                pd.read_csv(
                    path_dir + f"/ISBI2015/400_senior/{str(i+1).zfill(3)}.txt",
                    sep=",",
                    header=None,
                )[:19].to_numpy()
            )
        senior_landmarks = np.concatenate(landmarks_list, axis=0).reshape((-1, 19, 2))
        senior_landmarks = np.flip(senior_landmarks, axis=-1)
        landmarks = (junior_landmarks + senior_landmarks) / 2

    if cv:
        indices_fold_1 = (
            pd.read_table(path_dir + f"/ISBI2015/cv_payer/{1}.txt", header=None)
            .to_numpy()
            .flatten()
            .tolist()
        )
        indices_fold_2 = (
            pd.read_table(path_dir + f"/ISBI2015/cv_payer/{2}.txt", header=None)
            .to_numpy()
            .flatten()
            .tolist()
        )
        indices_fold_3 = (
            pd.read_table(path_dir + f"/ISBI2015/cv_payer/{3}.txt", header=None)
            .to_numpy()
            .flatten()
            .tolist()
        )
        indices_fold_4 = (
            pd.read_table(path_dir + f"/ISBI2015/cv_payer/{4}.txt", header=None)
            .to_numpy()
            .flatten()
            .tolist()
        )
        image_paths_fold1 = [
            glob.glob(path_dir + f"/ISBI2015/RawImage/*/{str(i).zfill(3)}.bmp")[0]
            for i in indices_fold_1
        ]
        image_paths_fold2 = [
            glob.glob(path_dir + f"/ISBI2015/RawImage/*/{str(i).zfill(3)}.bmp")[0]
            for i in indices_fold_2
        ]
        image_paths_fold3 = [
            glob.glob(path_dir + f"/ISBI2015/RawImage/*/{str(i).zfill(3)}.bmp")[0]
            for i in indices_fold_3
        ]
        image_paths_fold4 = [
            glob.glob(path_dir + f"/ISBI2015/RawImage/*/{str(i).zfill(3)}.bmp")[0]
            for i in indices_fold_4
        ]
        return (
            image_paths_fold1,
            image_paths_fold2,
            image_paths_fold3,
            image_paths_fold4,
            landmarks[[i - 1 for i in indices_fold_1]],
            landmarks[[i - 1 for i in indices_fold_2]],
            landmarks[[i - 1 for i in indices_fold_3]],
            landmarks[[i - 1 for i in indices_fold_4]],
            np.array([[0.1, 0.1]]).repeat(len(image_paths_fold1), axis=0),
            np.array([[0.1, 0.1]]).repeat(len(image_paths_fold2), axis=0),
            np.array([[0.1, 0.1]]).repeat(len(image_paths_fold3), axis=0),
            np.array([[0.1, 0.1]]).repeat(len(image_paths_fold4), axis=0),
        )

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
    train_transform=None,
    inference_transform=None,
    store_imgs=True,
    dim_img=None,
    junior=False,
    single_dataset=False,
    cv=False,
) -> (
    LandmarkDataset
    | tuple[LandmarkDataset, LandmarkDataset, LandmarkDataset]
    | tuple[LandmarkDataset, LandmarkDataset, LandmarkDataset, LandmarkDataset]
):
    """Returns a LandmarkDataset objects with the CEPH dataset, a combination of the ISBI 2014 &
    2015 challenges. The dataset is split into train, test1 and test2. The same approach as in
    "CephaNN: A Multi-Head Attention Network for Cephalometric Landmark Detection" - JIAHOONG QIAN
        et al. is used.
    """
    if single_dataset and cv:
        raise ValueError("Cannot have single dataset and cross validation at the same time.")
    if cv:
        (
            image_paths_fold1,
            image_paths_fold2,
            image_paths_fold3,
            image_paths_fold4,
            landmarks_fold1,
            landmarks_fold2,
            landmarks_fold3,
            landmarks_fold4,
            pixel_spacings_fold1,
            pixel_spacings_fold2,
            pixel_spacings_fold3,
            pixel_spacings_fold4,
        ) = get_cepha_dataset(path_dir, junior=junior, cv=True)
        return (
            LandmarkDataset(
                image_paths_fold1,
                landmarks_fold1,
                pixel_spacing=pixel_spacings_fold1,
                transform=train_transform,
                store_imgs=store_imgs,
                dim_img=dim_img,
            ),
            LandmarkDataset(
                image_paths_fold2,
                landmarks_fold2,
                pixel_spacing=pixel_spacings_fold2,
                transform=train_transform,
                store_imgs=store_imgs,
                dim_img=dim_img,
            ),
            LandmarkDataset(
                image_paths_fold3,
                landmarks_fold3,
                pixel_spacing=pixel_spacings_fold3,
                transform=train_transform,
                store_imgs=store_imgs,
                dim_img=dim_img,
            ),
            LandmarkDataset(
                image_paths_fold4,
                landmarks_fold4,
                pixel_spacing=pixel_spacings_fold4,
                transform=train_transform,
                store_imgs=store_imgs,
                dim_img=dim_img,
            ),
        )
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
    ) = get_cepha_dataset(path_dir, junior=junior, cv=False)
    if single_dataset:
        return LandmarkDataset(
            image_paths_train + image_paths_test1 + image_paths_test2,
            np.concatenate([landmarks_train, landmarks_test1, landmarks_test2], axis=0),
            pixel_spacing=np.concatenate(
                [pixel_spacings_train, pixel_spacings_test1, pixel_spacings_test2], axis=0
            ),
            transform=train_transform,
            store_imgs=store_imgs,
            dim_img=dim_img,
        )
    return (
        LandmarkDataset(
            image_paths_train,
            landmarks_train,
            pixel_spacing=pixel_spacings_train,
            transform=train_transform,
            store_imgs=store_imgs,
            dim_img=dim_img,
        ),
        LandmarkDataset(
            image_paths_test1,
            landmarks_test1,
            pixel_spacing=pixel_spacings_test1,
            transform=inference_transform,
            store_imgs=store_imgs,
            dim_img=dim_img,
        ),
        LandmarkDataset(
            image_paths_test2,
            landmarks_test2,
            pixel_spacing=pixel_spacings_test2,
            transform=inference_transform,
            store_imgs=store_imgs,
            dim_img=dim_img,
        ),
    )


def get_cepha_heatmap_datasets(
    path_dir: str,
    train_transform=None,
    inference_transform=None,
    sigma: float = 1,
    junior: bool = False,
    single_dataset: bool = False,
    cv: bool = False,
    **kwargs,
) -> (
    HeatmapDataset
    | tuple[HeatmapDataset, HeatmapDataset, HeatmapDataset]
    | tuple[HeatmapDataset, HeatmapDataset, HeatmapDataset, HeatmapDataset]
):
    """Returns a HeatmapDataset with the ISBI 2015 cephalogram challenge dataset. The dataset is
    split into train, test1 and test2. The same approach as in "CephaNN: A Multi-Head Attention
    Network for Cephalometric Landmark Detection" - JIAHOONG QIAN et al. is used.

    Args:
        path_dir (str): The path to the directory where the dataset should be stored.
        train_transform (Optional[Callable], optional): A transformation to apply to the images and
            landmarks during training. Defaults to None.
        inference_transform (Optional[Callable], optional): A transformation to apply to the images
            and landmarks during inference. Defaults to None.
        sigma (int, optional): The sigma value for the gaussian kernel. Defaults to 1.
        junior (bool, optional): Whether to use the junior or senior annotator. Defaults to False.
        single_dataset (bool, optional): Whether to return a single dataset with all images and
            landmarks. Defaults to False.
        **kwargs: Additional keyword arguments for the HeatmapDataset.
    """
    if single_dataset and cv:
        raise ValueError("Cannot have single dataset and cross validation at the same time.")
    if cv:
        (
            image_paths_fold1,
            image_paths_fold2,
            image_paths_fold3,
            image_paths_fold4,
            landmarks_fold1,
            landmarks_fold2,
            landmarks_fold3,
            landmarks_fold4,
            pixel_spacings_fold1,
            pixel_spacings_fold2,
            pixel_spacings_fold3,
            pixel_spacings_fold4,
        ) = get_cepha_dataset(path_dir, junior=junior, cv=True)
        return (
            HeatmapDataset(
                image_paths_fold1,
                landmarks_fold1,
                pixel_spacing=pixel_spacings_fold1,
                transform=train_transform,
                sigma=sigma,
                **kwargs,
            ),
            HeatmapDataset(
                image_paths_fold2,
                landmarks_fold2,
                pixel_spacing=pixel_spacings_fold2,
                transform=train_transform,
                sigma=sigma,
                **kwargs,
            ),
            HeatmapDataset(
                image_paths_fold3,
                landmarks_fold3,
                pixel_spacing=pixel_spacings_fold3,
                transform=train_transform,
                sigma=sigma,
                **kwargs,
            ),
            HeatmapDataset(
                image_paths_fold4,
                landmarks_fold4,
                pixel_spacing=pixel_spacings_fold4,
                transform=train_transform,
                sigma=sigma,
                **kwargs,
            ),
        )
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
    ) = get_cepha_dataset(path_dir, junior=junior, cv=False)
    if single_dataset:
        return HeatmapDataset(
            image_paths_train + image_paths_test1 + image_paths_test2,
            np.concatenate([landmarks_train, landmarks_test1, landmarks_test2], axis=0),
            pixel_spacing=np.concatenate(
                [pixel_spacings_train, pixel_spacings_test1, pixel_spacings_test2], axis=0
            ),
            transform=train_transform,
            sigma=sigma,
            **kwargs,
        )
    return (
        HeatmapDataset(
            image_paths_train,
            landmarks_train,
            pixel_spacing=pixel_spacings_train,
            transform=train_transform,
            sigma=sigma,
            **kwargs,
        ),
        HeatmapDataset(
            image_paths_test1,
            landmarks_test1,
            pixel_spacing=pixel_spacings_test1,
            transform=inference_transform,
            sigma=sigma,
            **kwargs,
        ),
        HeatmapDataset(
            image_paths_test2,
            landmarks_test2,
            pixel_spacing=pixel_spacings_test2,
            transform=inference_transform,
            sigma=sigma,
            **kwargs,
        ),
    )


def get_cepha_mask_datasets(
    path_dir: str,
    train_transform=None,
    inference_transform=None,
    junior: bool = False,
    single_dataset: bool = False,
    cv: bool = False,
    **kwargs,
) -> (
    MaskDataset
    | tuple[MaskDataset, MaskDataset, MaskDataset]
    | tuple[MaskDataset, MaskDataset, MaskDataset, MaskDataset]
):
    """Returns a MaskDataset with the ISBI 2015 cephalogram challenge dataset. The dataset is
    split into train, test1 and test2. The same approach as in "CephaNN: A Multi-Head Attention
    Network for Cephalometric Landmark Detection" - JIAHOONG QIAN et al. is used.

    Args:
        path_dir (str): The path to the directory where the dataset should be stored.
        train_transform (Optional[Callable], optional): A transformation to apply to the images and
            masks during training. Defaults to None.
        inference_transform (Optional[Callable], optional): A transformation to apply to the images
            and masks during inference. Defaults to None.
        junior (bool, optional): Whether to use the junior or senior annotator. Defaults to False.
        single_dataset (bool, optional): Whether to return a single dataset with all images and
            landmarks. Defaults to False.
        **kwargs: Additional keyword arguments for the MaskDataset.
    """
    if single_dataset and cv:
        raise ValueError("Cannot have single dataset and cross validation at the same time.")
    if cv:
        (
            image_paths_fold1,
            image_paths_fold2,
            image_paths_fold3,
            image_paths_fold4,
            landmarks_fold1,
            landmarks_fold2,
            landmarks_fold3,
            landmarks_fold4,
            pixel_spacings_fold1,
            pixel_spacings_fold2,
            pixel_spacings_fold3,
            pixel_spacings_fold4,
        ) = get_cepha_dataset(path_dir, junior=junior, cv=True)
        return (
            MaskDataset(
                image_paths_fold1,
                landmarks_fold1,
                pixel_spacing=pixel_spacings_fold1,
                transform=train_transform,
                **kwargs,
            ),
            MaskDataset(
                image_paths_fold2,
                landmarks_fold2,
                pixel_spacing=pixel_spacings_fold2,
                transform=train_transform,
                **kwargs,
            ),
            MaskDataset(
                image_paths_fold3,
                landmarks_fold3,
                pixel_spacing=pixel_spacings_fold3,
                transform=train_transform,
                **kwargs,
            ),
            MaskDataset(
                image_paths_fold4,
                landmarks_fold4,
                pixel_spacing=pixel_spacings_fold4,
                transform=train_transform,
                **kwargs,
            ),
        )
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
    ) = get_cepha_dataset(path_dir, junior=junior, cv=False)
    if single_dataset:
        return MaskDataset(
            image_paths_train + image_paths_test1 + image_paths_test2,
            np.concatenate([landmarks_train, landmarks_test1, landmarks_test2], axis=0),
            pixel_spacing=np.concatenate(
                [pixel_spacings_train, pixel_spacings_test1, pixel_spacings_test2], axis=0
            ),
            transform=train_transform,
            **kwargs,
        )
    return (
        MaskDataset(
            image_paths_train,
            landmarks_train,
            pixel_spacing=pixel_spacings_train,
            transform=train_transform,
            **kwargs,
        ),
        MaskDataset(
            image_paths_test1,
            landmarks_test1,
            pixel_spacing=pixel_spacings_test1,
            transform=inference_transform,
            **kwargs,
        ),
        MaskDataset(
            image_paths_test2,
            landmarks_test2,
            pixel_spacing=pixel_spacings_test2,
            transform=inference_transform,
            **kwargs,
        ),
    )


def get_cepha_patch_datasets(
    path_dir: str,
    index_landmark: int = 0,
    train_transform=None,
    inference_transform=None,
    store_imgs=True,
    junior=False,
    single_dataset=False,
    cv=False,
    **kwargs,
):
    """Returns a PatchDataset objects with the CEPH dataset, a combination of the ISBI 2014 &
    2015 challenges. The dataset is split into train, test1 and test2. The same approach as in
    "CephaNN: A Multi-Head Attention Network for Cephalometric Landmark Detection" - JIAHOONG QIAN
        et al. is used.
    """
    if single_dataset and cv:
        raise ValueError("Cannot have single dataset and cross validation at the same time.")
    if cv:
        (
            image_paths_fold1,
            image_paths_fold2,
            image_paths_fold3,
            image_paths_fold4,
            landmarks_fold1,
            landmarks_fold2,
            landmarks_fold3,
            landmarks_fold4,
            pixel_spacings_fold1,
            pixel_spacings_fold2,
            pixel_spacings_fold3,
            pixel_spacings_fold4,
        ) = get_cepha_dataset(path_dir, junior=junior, cv=True)
        return (
            PatchDataset(
                image_paths_fold1,
                landmarks_fold1,
                index_landmark=index_landmark,
                pixel_spacing=pixel_spacings_fold1,
                transform=train_transform,
                store_imgs=store_imgs,
                **kwargs,
            ),
            PatchDataset(
                image_paths_fold2,
                landmarks_fold2,
                index_landmark=index_landmark,
                pixel_spacing=pixel_spacings_fold2,
                transform=train_transform,
                store_imgs=store_imgs,
                **kwargs,
            ),
            PatchDataset(
                image_paths_fold3,
                landmarks_fold3,
                index_landmark=index_landmark,
                pixel_spacing=pixel_spacings_fold3,
                transform=train_transform,
                store_imgs=store_imgs,
                **kwargs,
            ),
            PatchDataset(
                image_paths_fold4,
                landmarks_fold4,
                index_landmark=index_landmark,
                pixel_spacing=pixel_spacings_fold4,
                transform=train_transform,
                store_imgs=store_imgs,
                **kwargs,
            ),
        )
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
    ) = get_cepha_dataset(path_dir, junior=junior, cv=False)
    if single_dataset:
        return PatchDataset(
            image_paths_train + image_paths_test1 + image_paths_test2,
            np.concatenate([landmarks_train, landmarks_test1, landmarks_test2], axis=0),
            index_landmark=index_landmark,
            pixel_spacing=np.concatenate(
                [pixel_spacings_train, pixel_spacings_test1, pixel_spacings_test2], axis=0
            ),
            transform=train_transform,
            store_imgs=store_imgs,
            **kwargs,
        )
    return (
        PatchDataset(
            image_paths_train,
            landmarks_train,
            index_landmark=index_landmark,
            pixel_spacing=pixel_spacings_train,
            transform=train_transform,
            store_imgs=store_imgs,
            **kwargs,
        ),
        PatchDataset(
            image_paths_test1,
            landmarks_test1,
            index_landmark=index_landmark,
            pixel_spacing=pixel_spacings_test1,
            transform=inference_transform,
            store_imgs=store_imgs,
            **kwargs,
        ),
        PatchDataset(
            image_paths_test2,
            landmarks_test2,
            index_landmark=index_landmark,
            pixel_spacing=pixel_spacings_test2,
            transform=inference_transform,
            store_imgs=store_imgs,
            **kwargs,
        ),
    )


def get_cepha_patch_mask_datasets(
    path_dir: str,
    index_landmark: int = 0,
    train_transform=None,
    inference_transform=None,
    store_imgs=True,
    junior=False,
    single_dataset=False,
    cv=False,
    **kwargs,
):
    """Returns a PatchMaskDataset objects with the CEPH dataset, a combination of the ISBI 2014 &
    2015 challenges. The dataset is split into train, test1 and test2. The same approach as in
    "CephaNN: A Multi-Head Attention Network for Cephalometric Landmark Detection" - JIAHOONG QIAN
        et al. is used.
    """
    if single_dataset and cv:
        raise ValueError("Cannot have single dataset and cross validation at the same time.")
    if cv:
        (
            image_paths_fold1,
            image_paths_fold2,
            image_paths_fold3,
            image_paths_fold4,
            landmarks_fold1,
            landmarks_fold2,
            landmarks_fold3,
            landmarks_fold4,
            pixel_spacings_fold1,
            pixel_spacings_fold2,
            pixel_spacings_fold3,
            pixel_spacings_fold4,
        ) = get_cepha_dataset(path_dir, junior=junior, cv=True)
        return (
            PatchMaskDataset(
                image_paths_fold1,
                landmarks_fold1,
                index_landmark=index_landmark,
                pixel_spacing=pixel_spacings_fold1,
                transform=train_transform,
                store_imgs=store_imgs,
                **kwargs,
            ),
            PatchMaskDataset(
                image_paths_fold2,
                landmarks_fold2,
                index_landmark=index_landmark,
                pixel_spacing=pixel_spacings_fold2,
                transform=train_transform,
                store_imgs=store_imgs,
                **kwargs,
            ),
            PatchMaskDataset(
                image_paths_fold3,
                landmarks_fold3,
                index_landmark=index_landmark,
                pixel_spacing=pixel_spacings_fold3,
                transform=train_transform,
                store_imgs=store_imgs,
                **kwargs,
            ),
            PatchMaskDataset(
                image_paths_fold4,
                landmarks_fold4,
                index_landmark=index_landmark,
                pixel_spacing=pixel_spacings_fold4,
                transform=train_transform,
                store_imgs=store_imgs,
                **kwargs,
            ),
        )
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
    ) = get_cepha_dataset(path_dir, junior=junior, cv=False)
    if single_dataset:
        return PatchMaskDataset(
            image_paths_train + image_paths_test1 + image_paths_test2,
            np.concatenate([landmarks_train, landmarks_test1, landmarks_test2], axis=0),
            index_landmark=index_landmark,
            pixel_spacing=np.concatenate(
                [pixel_spacings_train, pixel_spacings_test1, pixel_spacings_test2], axis=0
            ),
            transform=train_transform,
            store_imgs=store_imgs,
            **kwargs,
        )
    return (
        PatchMaskDataset(
            image_paths_train,
            landmarks_train,
            index_landmark=index_landmark,
            pixel_spacing=pixel_spacings_train,
            transform=train_transform,
            store_imgs=store_imgs,
            **kwargs,
        ),
        PatchMaskDataset(
            image_paths_test1,
            landmarks_test1,
            index_landmark=index_landmark,
            pixel_spacing=pixel_spacings_test1,
            transform=inference_transform,
            store_imgs=store_imgs,
            **kwargs,
        ),
        PatchMaskDataset(
            image_paths_test2,
            landmarks_test2,
            index_landmark=index_landmark,
            pixel_spacing=pixel_spacings_test2,
            transform=inference_transform,
            store_imgs=store_imgs,
            **kwargs,
        ),
    )
