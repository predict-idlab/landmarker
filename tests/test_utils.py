"""Tests for utils module."""

import json
import os

import cv2
import numpy as np
import pytest
import torch

from landmarker.utils.utils import (
    all_annotations_to_landmarks,
    all_annotations_to_landmarks_numpy,
    annotation_to_landmark,
    annotation_to_landmark_numpy,
    covert_video_to_frames,
    get_angle,
    get_angle_numpy,
    get_paths,
    pixel_to_unit,
    pixel_to_unit_numpy,
)


@pytest.fixture(scope="session", autouse=True)
def setup_data():
    """Create data for tests."""
    # Create a temporary directory and files for testing
    tmpdir = os.path.join(os.path.dirname(__file__), "tmp")
    os.mkdir(tmpdir)
    test_files = ["file1.txt", "file2.txt", "file3.jpg"]
    for file in test_files:
        with open(os.path.join(tmpdir, file), "a", encoding="utf8") as f:
            f.close()

    # Define test cases
    json_objs = [
        {
            "shapes": [
                {"label": "class1", "points": [[1, 2]]},
                {"label": "class2", "points": [[3, 4]]},
                {"label": "class3", "points": [[5, 6]]},
            ]
        },
        {
            "shapes": [
                {"label": "class1", "points": [[7, 8]]},
                {"label": "class2", "points": [[9, 10]]},
                {"label": "class3", "points": [[11, 12]]},
            ]
        },
    ]

    # Write the test cases to temporary files
    paths = []
    for i, json_obj in enumerate(json_objs):
        path = os.path.join(tmpdir, f"test{i}.json")
        with open(path, "w", encoding="utf8") as f:
            json.dump(json_obj, f)
        paths.append(path)

    yield

    # Remove the temporary directory and files
    for file in test_files:
        os.remove(os.path.join(tmpdir, file))
    for path in paths:
        os.remove(path)

    os.rmdir(tmpdir)


def test_get_paths():
    """Test the get_paths function."""

    # Test if get_paths correctly finds .txt files
    txt_paths = get_paths(os.path.dirname(__file__), "txt")
    assert len(txt_paths) == 2
    assert all(path.endswith(".txt") for path in txt_paths)

    # Test if get_paths correctly finds .jpg files
    jpg_paths = get_paths(os.path.dirname(__file__), "jpg")
    assert len(jpg_paths) == 1
    assert all(path.endswith(".jpg") for path in jpg_paths)


def test_annotation_to_landmark():
    """Test the annotation_to_landmark function."""
    # Define a test case
    json_obj = {
        "shapes": [
            {"label": "class1", "points": [[1, 2]]},
            {"label": "class2", "points": [[3, 4]]},
            {"label": "class3", "points": [[5, 6]]},
        ]
    }
    class_names = ["class1", "class2", "class3"]

    # Call the function with the test case
    landmarks = annotation_to_landmark(json_obj, class_names)

    # Check the result
    expected_landmarks = torch.Tensor([[2, 1], [4, 3], [6, 5]])
    assert torch.allclose(landmarks, expected_landmarks)

    # Test with a class name that is not in the json_obj
    class_names = ["class1", "class4"]
    landmarks = annotation_to_landmark(json_obj, class_names)
    expected_landmarks = torch.Tensor([[2, 1], [float("nan"), float("nan")]])
    assert torch.allclose(landmarks, expected_landmarks, equal_nan=True)


def test_annotation_to_landmark_numpy():
    """Test the annotation_to_landmark function. (Numpy version)"""
    # Define a test case
    json_obj = {
        "shapes": [
            {"label": "class1", "points": [[1, 2]]},
            {"label": "class2", "points": [[3, 4]]},
            {"label": "class3", "points": [[5, 6]]},
        ]
    }
    class_names = ["class1", "class2", "class3"]

    # Call the function with the test case
    landmarks = annotation_to_landmark_numpy(json_obj, class_names)

    # Check the result
    expected_landmarks = np.array([[2, 1], [4, 3], [6, 5]])
    assert np.allclose(landmarks, expected_landmarks)

    # Test with a class name that is not in the json_obj
    class_names = ["class1", "class4"]
    landmarks = annotation_to_landmark_numpy(json_obj, class_names)
    expected_landmarks = np.array([[2, 1], [np.nan, np.nan]])
    assert np.allclose(landmarks, expected_landmarks, equal_nan=True)


def test_all_annotations_to_landmarks():
    """Test the all_annotations_to_landmarks function."""
    tmpdir = os.path.join(os.path.dirname(__file__), "tmp")

    class_names = ["class1", "class2", "class3"]

    # Write the test cases to temporary files
    paths = sorted(list(get_paths(tmpdir, "json")))

    # Call the function with the test cases
    landmarks = all_annotations_to_landmarks(paths, class_names)

    # Check the result
    expected_landmarks = torch.Tensor(
        [
            [[2, 1], [4, 3], [6, 5]],
            [[8, 7], [10, 9], [12, 11]],
        ]
    )
    assert torch.allclose(landmarks, expected_landmarks)


def test_all_annotations_to_landmarks_numpy():
    """Test the all_annotations_to_landmarks function. (Numpy version)"""
    tmpdir = os.path.join(os.path.dirname(__file__), "tmp")

    class_names = ["class1", "class2", "class3"]

    # Write the test cases to temporary files
    paths = sorted(list(get_paths(tmpdir, "json")))

    # Call the function with the test cases
    landmarks = all_annotations_to_landmarks_numpy(paths, class_names)

    # Check the result
    expected_landmarks = np.array(
        [
            [[2, 1], [4, 3], [6, 5]],
            [[8, 7], [10, 9], [12, 11]],
        ]
    )
    assert np.allclose(landmarks, expected_landmarks)


def test_get_angle():
    """Test the get_angle function."""
    # Define test cases
    p1 = torch.Tensor([0, 0])
    p2 = torch.Tensor([1, 0])
    p3 = torch.Tensor([0, 1])

    # Call the function with the test cases
    angle_radial = get_angle(p1, p2, p3, radial=True)
    angle_degrees = get_angle(p1, p2, p3, radial=False)

    # Check the results
    assert torch.allclose(angle_radial, torch.Tensor([np.pi / 2]))
    assert torch.allclose(angle_degrees, torch.Tensor([90]))

    # Test with different points
    p3 = torch.Tensor([1, 1])
    angle_radial = get_angle(p1, p2, p3, radial=True)
    angle_degrees = get_angle(p1, p2, p3, radial=False)

    # Check the results
    assert torch.allclose(angle_radial, torch.Tensor([np.pi / 4]))
    assert torch.allclose(angle_degrees, torch.Tensor([45]))


def test_get_angle_numpy():
    """Test the get_angle function. (Numpy version)"""
    # Define test cases
    p1 = np.array([0, 0])
    p2 = np.array([1, 0])
    p3 = np.array([0, 1])

    # Call the function with the test cases
    angle_radial = get_angle_numpy(p1, p2, p3, radial=True)
    angle_degrees = get_angle_numpy(p1, p2, p3, radial=False)

    # Check the results
    assert np.allclose(angle_radial, np.array([np.pi / 2]))
    assert np.allclose(angle_degrees, np.array([90]))

    # Test with different points
    p3 = np.array([1, 1])
    angle_radial = get_angle_numpy(p1, p2, p3, radial=True)
    angle_degrees = get_angle_numpy(p1, p2, p3, radial=False)

    # Check the results
    assert np.allclose(angle_radial, np.array([np.pi / 4]))
    assert np.allclose(angle_degrees, np.array([45]))


def test_pixel_to_unit_spacing_torch():
    """
    Test the pixel_to_unit function with pixel spacing and torch tensors.
    """
    landmarks = torch.Tensor([[[30, 20], [100, 70]], [[105, 60], [70, 80]]])
    pixel_spacing = torch.Tensor([[0.3, 0.1], [0.1, 0.3]])
    dim = (50, 100)
    dim_orig = torch.Tensor([[100, 150], [200, 200]])

    landmarks_resized = (
        landmarks * torch.Tensor(dim).reshape((-1, 1, 2)) / dim_orig.reshape((-1, 1, 2))
    )

    expected_result = landmarks * pixel_spacing.reshape((-1, 1, 2))

    result = pixel_to_unit(landmarks_resized, pixel_spacing, dim, dim_orig)

    assert torch.allclose(result, expected_result)


def test_pixel_to_unit_no_resize_torch():
    """
    Test the pixel_to_unit function without resizing.
    """
    landmarks = torch.Tensor([[[30, 20], [100, 70]], [[105, 60], [70, 80]]])
    pixel_spacing = torch.Tensor([[0.3, 0.1], [0.1, 0.3]])

    expected_result = torch.Tensor(
        [
            [[30 * 0.3, 20 * 0.1], [100 * 0.3, 70 * 0.1]],
            [[105 * 0.1, 60 * 0.3], [70 * 0.1, 80 * 0.3]],
        ]
    )

    result = pixel_to_unit(landmarks, pixel_spacing)

    assert torch.allclose(result, expected_result)

    with pytest.raises(ValueError):
        result = pixel_to_unit(landmarks, pixel_spacing, dim=(100, 100))


def test_pixel_to_unit_spacing_numpy():
    """
    Test the pixel_to_unit_numpy function with pixel spacing and numpy arrays.
    """
    landmarks = np.array([[[30, 20], [100, 70]], [[105, 60], [70, 80]]])
    pixel_spacing = np.array([[0.3, 0.1], [0.1, 0.3]])
    dim = (50, 100)
    dim_orig = np.array([[100, 150], [200, 200]])

    landmarks_resized = landmarks * np.array(dim).reshape((-1, 1, 2)) / dim_orig.reshape((-1, 1, 2))

    expected_result = landmarks * pixel_spacing.reshape((-1, 1, 2))

    result = pixel_to_unit_numpy(landmarks_resized, pixel_spacing, dim, dim_orig)

    assert np.allclose(result, expected_result)


def test_pixel_to_unit_padding_numpy():
    """
    Test the pixel_to_unit_numpy function with padding and numpy arrays.
    """
    landmarks_orig = np.array([[[30, 20], [100, 70]], [[105, 60], [70, 80]]])
    pixel_spacing = None
    dim = (300, 200)
    dim_orig = np.array([[100, 200], [200, 200]])
    padding = np.array([[100, 0], [50, 0]])

    landmarks_padded = landmarks_orig + padding.reshape((-1, 1, 2))
    landmarks = landmarks_padded

    result = pixel_to_unit_numpy(landmarks, pixel_spacing, dim, dim_orig, padding)

    assert np.allclose(result, landmarks_orig)

    landmarks_orig = np.array([[[30, 20], [100, 70]], [[105, 60], [70, 80]]])
    pixel_spacing = None
    dim = (150, 100)
    dim_orig = np.array([[100, 200], [200, 200]])
    padding = np.array([[100, 0], [50, 0]])

    landmarks_padded = landmarks_orig + padding.reshape((-1, 1, 2))
    landmarks = landmarks_padded * (
        np.array(dim).reshape((-1, 1, 2)) / (dim_orig + 2 * padding).reshape((-1, 1, 2))
    )

    result = pixel_to_unit_numpy(landmarks, pixel_spacing, dim, dim_orig, padding)

    assert np.allclose(result, landmarks_orig)


def test_pixel_to_unit_no_resize_numpy():
    """
    Test the pixel_to_unit_numpy function without resizing.
    """
    landmarks = np.array([[[30, 20], [100, 70]], [[105, 60], [70, 80]]])
    pixel_spacing = np.array([[0.3, 0.1], [0.1, 0.3]])

    expected_result = np.array(
        [
            [[30 * 0.3, 20 * 0.1], [100 * 0.3, 70 * 0.1]],
            [[105 * 0.1, 60 * 0.3], [70 * 0.1, 80 * 0.3]],
        ]
    )

    result = pixel_to_unit_numpy(landmarks, pixel_spacing)

    assert np.allclose(result, expected_result)

    try:
        result = pixel_to_unit_numpy(landmarks, pixel_spacing, dim=(100, 100))
        assert False
    except ValueError:
        assert True


def test_pixel_to_unit_array_dim_numpy():
    """
    Test the pixel_to_unit_numpy function with array dim.
    """
    landmarks = np.array([[[30, 20], [100, 70]], [[105, 60], [70, 80]]])
    pixel_spacing = np.array([[0.3, 0.1], [0.1, 0.3]])
    dim = np.array([(50, 100), (100, 200)])
    dim_orig = np.array([[100, 150], [200, 200]])

    landmarks_resized = landmarks * np.array(dim).reshape((-1, 1, 2)) / dim_orig.reshape((-1, 1, 2))

    expected_result = landmarks * pixel_spacing.reshape((-1, 1, 2))

    result = pixel_to_unit_numpy(landmarks_resized, pixel_spacing, dim, dim_orig)

    assert np.allclose(result, expected_result)


def test_covert_video_to_frames(tmpdir):
    """Test the covert_video_to_frames function."""
    # Create a temporary video file for testing
    video_path = os.path.join(tmpdir, "test.avi")
    height, width, frames = 240, 320, 10
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (width, height))
    for _ in range(frames):
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        video.write(frame)
    video.release()

    # Call the function with the test video
    frames_path = os.path.join(tmpdir, "frames")
    covert_video_to_frames(video_path, frames_path)

    # Check if the correct number of frames were saved
    saved_frames = os.listdir(frames_path)
    assert len(saved_frames) == frames

    # Check if the frames were saved with the correct names
    for i in range(1, frames + 1):
        assert f"img_{str(i).zfill(6)}.png" in saved_frames
