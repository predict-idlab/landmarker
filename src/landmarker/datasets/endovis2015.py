"""
Multi-instance and multi-class landmark dataset using data from the EndoVis 2015 challenge. The
dataset contains 4 training and 6 testing videos of robotic surgery. The goal is to predict the
location of instruments in the video, more specifically the tip of the clasper. We only consider the
clasper points and ignore the other points, since they are way more ambiguous. The videos are
transformed into images and the annotations are given as 2D points. The dataset is split into a
training and testing set. The training set contains 4 videos and the testing set contains 6 videos,
such as specified in the challenge. However, it is possible to use the other points by specifying
the class_names argument.

The dataset is available at https://opencas.webarchiv.kit.edu/data/endovis15_ins and the annotations
are available at https://github.com/surgical-vision/EndoVisPoseAnnotation.
"""

import json
import os
import zipfile

import opendatasets as od  # type: ignore
import torch

from landmarker.data import HeatmapDataset, LandmarkDataset
from landmarker.utils import covert_video_to_frames

# class_names = ["LeftClasperPoint", "RightClasperPoint", "HeadPoint",
#                "ShaftPoint", "TrackedPoint", "EndPoint"]


def get_endovis2015_dataset(
    path_dir: str, class_names: list[str] = ["LeftClasperPoint", "RightClasperPoint"]
):
    if not os.path.exists(path_dir + "/EndoVis2015"):
        od.download_url(
            "https://opencas.webarchiv.kit.edu/data/endovis15_ins/Tracking_Robotic_Training.zip",
            path_dir,
        )
        od.download_url(
            "https://opencas.webarchiv.kit.edu/data/endovis15_ins/Tracking_Robotic_Testing.zip",
            path_dir,
        )
        with zipfile.ZipFile(path_dir + "/Tracking_Robotic_Training.zip", "r") as zip_ref:
            zip_ref.extractall(path=path_dir + "/EndoVis2015")
        with zipfile.ZipFile(path_dir + "/Tracking_Robotic_Testing.zip", "r") as zip_ref:
            zip_ref.extractall(path=path_dir + "/EndoVis2015")
        os.rename(
            path_dir + "/EndoVis2015/Tracking_Robotic_Training/Training",
            path_dir + "/EndoVis2015/Training",
        )
        os.rmdir(path_dir + "/EndoVis2015/Tracking_Robotic_Training")
        os.rename(
            path_dir + "/EndoVis2015/Training", path_dir + "/EndoVis2015/Tracking_Robotic_Training"
        )
        os.rename(
            path_dir + "/EndoVis2015/Tracking", path_dir + "/EndoVis2015/Tracking_Robotic_Testing"
        )
        os.remove(path_dir + "/Tracking_Robotic_Training.zip")
        os.remove(path_dir + "/Tracking_Robotic_Testing.zip")
        label_path = (
            "https://raw.githubusercontent.com/" + "surgical-vision/EndoVisPoseAnnotation/master"
        )
        for i in range(4):
            od.download_url(
                f"{label_path}/train_labels/train{i+1}_labels.json",
                path_dir + "/EndoVis2015/Tracking_Robotic_Training/labels",
            )
        for i in range(6):
            od.download_url(
                f"{label_path}/test_labels/test{i+1}_labels.json",
                path_dir + "/EndoVis2015/Tracking_Robotic_Testing/labels",
            )
        for i in range(4):
            covert_video_to_frames(
                path_dir + f"/EndoVis2015/Tracking_Robotic_Training/Dataset{i+1}/Video.avi",
                path_dir + f"/EndoVis2015/Tracking_Robotic_Training/Dataset{i+1}/raw",
                zero_fill=6,
            )
        for i in range(6):
            covert_video_to_frames(
                path_dir + f"/EndoVis2015/Tracking_Robotic_Testing/Dataset{i+1}/Video.avi",
                path_dir + f"/EndoVis2015/Tracking_Robotic_Testing/Dataset{i+1}/raw",
                zero_fill=4,
            )

    train_landmarks = {}
    for train_id in range(1, 5):
        json_labels = json.load(
            open(
                path_dir
                + f"/EndoVis2015/Tracking_Robotic_Training/labels/train{train_id}_labels.json"
            )
        )
        for annotation in json_labels:
            if annotation["annotations"] == []:
                continue
            landmarks = {}
            for name in class_names:
                landmarks[name] = torch.zeros((2, 2)).fill_(torch.nan)
                for labels in annotation["annotations"]:
                    if labels["class"] == name:
                        if landmarks[name][0, 0].isnan():
                            landmarks[name][0] = torch.tensor([labels["y"], labels["x"]])
                        else:
                            landmarks[name][1] = torch.tensor([labels["y"], labels["x"]])
            train_landmarks[
                path_dir
                + f"/EndoVis2015/Tracking_Robotic_Training/Dataset{train_id}/raw/"
                + os.path.basename(annotation["filename"].replace("_raw", ""))
            ] = landmarks

    train_landmarks_torch = torch.zeros((len(train_landmarks), len(class_names), 2, 2))
    for i, (key, value) in enumerate(train_landmarks.items()):
        for j, class_name in enumerate(class_names):
            train_landmarks_torch[i, j] = value[class_name]

    test_landmarks = {}
    for test_id in range(1, 7):
        json_labels = json.load(
            open(
                path_dir + f"/EndoVis2015/Tracking_Robotic_Testing/labels/test{test_id}_labels.json"
            )
        )
        for annotation in json_labels:
            if annotation["annotations"] == []:
                continue
            landmarks = {}
            for name in class_names:
                landmarks[name] = torch.zeros((2, 2)).fill_(torch.nan)
                for labels in annotation["annotations"]:
                    if labels["class"] == name:
                        if landmarks[name][0, 0].isnan():
                            landmarks[name][0] = torch.tensor([labels["y"], labels["x"]])
                        else:
                            landmarks[name][1] = torch.tensor([labels["y"], labels["x"]])
            test_landmarks[
                path_dir
                + f"/EndoVis2015/Tracking_Robotic_Testing/Dataset{test_id}/raw/"
                + os.path.basename(annotation["filename"])
            ] = landmarks

    test_landmarks_torch = torch.zeros((len(test_landmarks), len(class_names), 2, 2))
    for i, (key, value) in enumerate(test_landmarks.items()):
        for j, class_name in enumerate(class_names):
            test_landmarks_torch[i, j] = value[class_name]

    train_image_paths = list(train_landmarks.keys())
    test_image_paths = list(test_landmarks.keys())

    return train_image_paths, test_image_paths, train_landmarks_torch, test_landmarks_torch


def get_endovis2015_landmark_datasets(
    path_dir: str,
    class_names: list[str] = ["LeftClasperPoint", "RightClasperPoint"],
    train_transform=None,
    inference_transform=None,
    **kwargs,
) -> tuple[LandmarkDataset, LandmarkDataset]:
    (
        train_image_paths,
        test_image_paths,
        train_landmarks_torch,
        test_landmarks_torch,
    ) = get_endovis2015_dataset(path_dir, class_names)
    return LandmarkDataset(
        train_image_paths,
        train_landmarks_torch,
        class_names=class_names,
        transform=train_transform,
        **kwargs,
    ), LandmarkDataset(
        test_image_paths,
        test_landmarks_torch,
        class_names=class_names,
        transform=inference_transform,
        **kwargs,
    )


def get_endovis2015_heatmap_datasets(
    path_dir: str,
    class_names: list[str] = ["LeftClasperPoint", "RightClasperPoint"],
    sigma: float = 1.0,
    train_transform=None,
    inference_transform=None,
    **kwargs,
):
    (
        train_image_paths,
        test_image_paths,
        train_landmarks_torch,
        test_landmarks_torch,
    ) = get_endovis2015_dataset(path_dir, class_names)
    return (
        HeatmapDataset(
            train_image_paths,
            train_landmarks_torch,
            class_names=class_names,
            sigma=sigma,
            transform=train_transform,
            **kwargs,
        ),
        HeatmapDataset(
            test_image_paths,
            test_landmarks_torch,
            class_names=class_names,
            sigma=sigma,
            transform=inference_transform,
            **kwargs,
        ),
    )
