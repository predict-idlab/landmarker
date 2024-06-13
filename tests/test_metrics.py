"""Tests for the metrics module."""

import numpy as np
import torch

from src.landmarker.metrics.metrics import multi_instance_point_error, point_error, sdr


def test_point_error():
    """Test the point_error function."""
    true_landmarks = torch.Tensor([[[10, 20], [30, 40]], [[50, 60], [70, 80]]])
    pred_landmarks = torch.Tensor([[[10, 20], [30, 42]], [[52, 62], [75, 82]]])
    pixel_spacing = torch.Tensor([[0.1, 0.2], [0.3, 0.4]])
    dim = (100, 200)
    dim_orig = torch.Tensor([[50, 100], [150, 200]])

    expected_result = torch.Tensor(
        [
            [0, 2 * 0.2 * 100 / 200],
            [
                np.sqrt((2 * 0.3 * 150 / 100) ** 2 + (2 * 0.4 * 200 / 200) ** 2),
                np.sqrt((5 * 0.3 * 150 / 100) ** 2 + (2 * 0.4 * 200 / 200) ** 2),
            ],
        ]
    )

    result = point_error(
        true_landmarks,
        pred_landmarks,
        dim=dim,
        dim_orig=dim_orig,
        pixel_spacing=pixel_spacing,
        reduction="none",
        padding=None,
    )

    assert torch.allclose(result, expected_result)

    expected_result = torch.mean(expected_result)
    result = point_error(
        true_landmarks,
        pred_landmarks,
        dim=dim,
        dim_orig=dim_orig,
        pixel_spacing=pixel_spacing,
        reduction="mean",
    )

    assert torch.allclose(result, expected_result)

    expected_result = torch.Tensor(
        [
            [0, 2 * 100 / 200],
            [
                np.sqrt((2 * 150 / 100) ** 2 + (2 * 200 / 200) ** 2),
                np.sqrt((5 * 150 / 100) ** 2 + (2 * 200 / 200) ** 2),
            ],
        ]
    )

    result = point_error(
        true_landmarks,
        pred_landmarks,
        dim=dim,
        dim_orig=dim_orig,
        pixel_spacing=None,
        reduction="none",
    )

    assert torch.allclose(result, expected_result)

    expected_result = torch.Tensor(
        [[0, 2], [np.sqrt((2) ** 2 + (2) ** 2), np.sqrt((5) ** 2 + (2) ** 2)]]
    )

    try:
        result = point_error(
            true_landmarks,
            pred_landmarks,
            dim=dim,
            dim_orig=dim_orig,
            pixel_spacing=None,
            reduction="not supported",
        )
        assert False
    except ValueError:
        assert True


def test_point_error_3d():
    """Test the point_error function. 3D case."""
    true_landmarks = torch.Tensor([[[10, 20, 34], [30, 40, 20]], [[50, 60, 14], [70, 80, 45]]])
    pred_landmarks = torch.Tensor([[[10, 20, 34], [30, 42, 20]], [[52, 62, 14], [75, 82, 38]]])
    pixel_spacing = torch.Tensor([[0.1, 0.2, 0.1], [0.3, 0.4, 0.2]])
    dim = (100, 200, 120)
    dim_orig = torch.Tensor([[50, 100, 200], [150, 200, 200]])

    expected_result = torch.Tensor(
        [
            [0, 2 * 0.2 * 100 / 200],
            [
                np.sqrt((2 * 0.3 * 150 / 100) ** 2 + (2 * 0.4 * 200 / 200) ** 2),
                np.sqrt(
                    (5 * 0.3 * 150 / 100) ** 2
                    + (2 * 0.4 * 200 / 200) ** 2
                    + (7 * 0.2 * 200 / 120) ** 2
                ),
            ],
        ]
    )

    result = point_error(
        true_landmarks,
        pred_landmarks,
        dim=dim,
        dim_orig=dim_orig,
        pixel_spacing=pixel_spacing,
        reduction="none",
        padding=None,
    )

    assert torch.allclose(result, expected_result)

    expected_result = torch.mean(expected_result)
    result = point_error(
        true_landmarks,
        pred_landmarks,
        dim=dim,
        dim_orig=dim_orig,
        pixel_spacing=pixel_spacing,
        reduction="mean",
    )

    assert torch.allclose(result, expected_result)

    expected_result = torch.Tensor(
        [
            [0, 2 * 100 / 200],
            [
                np.sqrt((2 * 150 / 100) ** 2 + (2 * 200 / 200) ** 2),
                np.sqrt((5 * 150 / 100) ** 2 + (2 * 200 / 200) ** 2 + (7 * 200 / 120) ** 2),
            ],
        ]
    )

    result = point_error(
        true_landmarks,
        pred_landmarks,
        dim=dim,
        dim_orig=dim_orig,
        pixel_spacing=None,
        reduction="none",
    )

    assert torch.allclose(result, expected_result)

    try:
        result = point_error(
            true_landmarks,
            pred_landmarks,
            dim=dim,
            dim_orig=dim_orig,
            pixel_spacing=None,
            reduction="not supported",
        )
        assert False
    except ValueError:
        assert True


def test_sdr_single_radius():
    """
    Test the sdr function for a single radius.
    """
    # create some dummy itorchut tensors
    radius = 5
    true_landmarks = torch.Tensor(np.random.rand(4, 10, 2))
    pred_landmarks = torch.Tensor(np.random.rand(4, 10, 2))

    # calculate the SDR
    sdr_ = sdr(radius, true_landmarks, pred_landmarks)

    # check that the output is a scalar value
    assert isinstance(sdr_, float)

    # check that the output is between 0 and 100
    assert 0 <= sdr_ <= 100


def test_sdr_single_radius_3d():
    """
    Test the sdr function for a single radius. 3D case.
    """
    # create some dummy itorchut tensors
    radius = 5
    true_landmarks = torch.Tensor(np.random.rand(4, 10, 3))
    pred_landmarks = torch.Tensor(np.random.rand(4, 10, 3))

    # calculate the SDR
    sdr_ = sdr(radius, true_landmarks, pred_landmarks)

    # check that the output is a scalar value
    assert isinstance(sdr_, float)

    # check that the output is between 0 and 100
    assert 0 <= sdr_ <= 100


def test_sdr_multiple_radii():
    """
    Test the sdr function for multiple radii.
    """
    # create some dummy itorchut tensors
    radii = [1, 5, 10]
    true_landmarks = torch.Tensor(np.random.rand(4, 10, 2))
    pred_landmarks = torch.Tensor(np.random.rand(4, 10, 2))

    # calculate the SDR for multiple radii
    sdr_dict = sdr(radii, true_landmarks, pred_landmarks)

    # check that the output is a dictionary
    assert isinstance(sdr_dict, dict)

    # check that the dictionary has the correct keys
    assert set(sdr_dict.keys()) == set(radii)

    # check that the dictionary values are between 0 and 100
    for sdr_ in sdr_dict.values():
        assert 0 <= sdr_ <= 100


def test_sdr_multiple_radii_3d():
    """
    Test the sdr function for multiple radii. 3D case.
    """
    # create some dummy itorchut tensors
    radii = [1, 5, 10]
    true_landmarks = torch.Tensor(np.random.rand(4, 10, 3))
    pred_landmarks = torch.Tensor(np.random.rand(4, 10, 3))

    # calculate the SDR for multiple radii
    sdr_dict = sdr(radii, true_landmarks, pred_landmarks)

    # check that the output is a dictionary
    assert isinstance(sdr_dict, dict)

    # check that the dictionary has the correct keys
    assert set(sdr_dict.keys()) == set(radii)

    # check that the dictionary values are between 0 and 100
    for sdr_ in sdr_dict.values():
        assert 0 <= sdr_ <= 100


def test_multi_instance_point_error():
    # Create lists of true and predicted landmarks
    true_landmarks = torch.Tensor([[[[1, 2], [3, 4], [5, 6]]]])
    pred_landmarks = [torch.Tensor([1, 2]), torch.Tensor([3, 4]), torch.Tensor([5, 6])]

    assert true_landmarks.shape == (1, 1, 3, 2)  # (batch, class, instance, dim)

    # Call the function
    pe, tp, fp, fn, _ = multi_instance_point_error(true_landmarks, pred_landmarks)

    # Check that the output has the expected values
    assert pe == 0
    assert tp == 3
    assert fp == 0


def test_multi_instance_point_error_3d():
    # Create lists of true and predicted landmarks
    true_landmarks = torch.Tensor([[[[1, 2, 2], [3, 4, 2], [5, 6, 5]]]])
    pred_landmarks = [torch.Tensor([1, 2, 2]), torch.Tensor([3, 4, 2]), torch.Tensor([5, 6, 5])]

    assert true_landmarks.shape == (1, 1, 3, 3)  # (batch, class, instance, dim)

    # Call the function
    pe, tp, fp, fn, _ = multi_instance_point_error(true_landmarks, pred_landmarks)

    # Check that the output has the expected values
    assert pe == 0
    assert tp == 3
    assert fp == 0


def test_multi_instance_point_error_different_landmarks():
    # Create lists of true and predicted landmarks
    true_landmarks = torch.Tensor([[[[1, 2], [3, 4], [5, 6]]]])
    assert true_landmarks.shape == (1, 1, 3, 2)
    pred_landmarks = [torch.tensor([1, 2]), torch.tensor([3, 4]), torch.tensor([7, 8])]

    # Call the function
    pe, tp, fp, fn, _ = multi_instance_point_error(true_landmarks, pred_landmarks)

    # Check that the output has the expected values
    assert pe > 0
    assert tp == 3
    assert fp == 0


def test_multi_instance_point_error_different_landmarks_3d():
    # Create lists of true and predicted landmarks
    true_landmarks = torch.Tensor([[[[1, 2, 4], [3, 4, 5], [5, 6, 3]]]])
    assert true_landmarks.shape == (1, 1, 3, 3)
    pred_landmarks = [torch.tensor([1, 2, 3]), torch.tensor([3, 4, 2]), torch.tensor([7, 8, 5])]

    # Call the function
    pe, tp, fp, fn, _ = multi_instance_point_error(true_landmarks, pred_landmarks)

    # Check that the output has the expected values
    assert pe > 0
    assert tp == 3
    assert fp == 0


def test_multi_instance_point_error_extra_predicted_landmark():
    # Create lists of true and predicted landmarks
    true_landmarks = torch.Tensor([[[[1, 2], [3, 4], [5, 6]]]])
    pred_landmarks = [
        torch.tensor([1, 2]),
        torch.tensor([3, 4]),
        torch.tensor([5, 6]),
        torch.tensor([7, 8]),
    ]

    # Call the function
    pe, tp, fp, fn, _ = multi_instance_point_error(true_landmarks, pred_landmarks)

    # Check that the output has the expected values
    assert pe == 0
    assert tp == 3
    assert fp == 1


def test_multi_instance_point_error_extra_predicted_landmark_3d():
    # Create lists of true and predicted landmarks
    true_landmarks = torch.Tensor([[[[1, 2, 4], [3, 4, 5], [5, 6, 3]]]])
    pred_landmarks = [
        torch.tensor([1, 2, 4]),
        torch.tensor([3, 4, 5]),
        torch.tensor([5, 6, 3]),
        torch.tensor([7, 8, 2]),
    ]

    # Call the function
    pe, tp, fp, fn, _ = multi_instance_point_error(true_landmarks, pred_landmarks)

    # Check that the output has the expected values
    assert pe == 0
    assert tp == 3
    assert fp == 1
