import numpy as np
import torch
import pytest

from src.landmarker.heatmap.generator import (
    GaussianHeatmapGenerator,
    LaplacianHeatmapGenerator,
    from_2by2_to_4by4,
    from_3by3_to_4by4,
)
from src.landmarker.heatmap.decoder import coord_argmax

landmarks = torch.tensor([[[10, 20], [30, 40], [50, 60]]], dtype=torch.int)
landmarks_batch = torch.tensor(
    [
        [[10, 20], [30, 40], [50, 60]],
        [[11, 22], [24, 35], [52, 58]],
        [[12, 24], [32, 42], [54, 60]],
        [[13, 26], [36, 46], [56, 61]],
    ],
    dtype=torch.int,
)


def test_from_2by2_to_4by4():
    affine_matrix = torch.tensor([[2.3, 0.0], [0.0, 0.5]])
    expected_result = torch.tensor(
        [[2.3, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )

    result = from_2by2_to_4by4(affine_matrix)

    assert expected_result.shape == result.shape
    assert torch.allclose(result, expected_result)

    affine_matrix = affine_matrix.unsqueeze(0)
    expected_result = expected_result.unsqueeze(0)

    result = from_2by2_to_4by4(affine_matrix)

    assert expected_result.shape == result.shape
    assert torch.allclose(result, expected_result)

    channel_affine_matrix = torch.tensor(
        [[[2.3, 0.0], [0.0, 0.5]], [[1, 0], [0, 1]], [[0, -1], [1, 0]]]
    )
    expected_result = torch.tensor(
        [
            [
                [2.3, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ]
    )

    result = from_2by2_to_4by4(channel_affine_matrix)

    assert expected_result.shape == result.shape
    assert torch.allclose(result, expected_result)

    batch_channel_affine_matrix = channel_affine_matrix.unsqueeze(0).repeat(4, 1, 1, 1)
    expected_result = expected_result.unsqueeze(0).repeat(4, 1, 1, 1)

    result = from_2by2_to_4by4(batch_channel_affine_matrix)

    assert expected_result.shape == result.shape
    assert torch.allclose(result, expected_result)


def test_from_3by3_to_4by4():
    affine_matrix = torch.tensor([[2.3, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]])
    expected_result = torch.tensor(
        [[2.3, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )

    result = from_3by3_to_4by4(affine_matrix)

    assert expected_result.shape == result.shape
    assert torch.allclose(result, expected_result)

    affine_matrix = affine_matrix.unsqueeze(0)
    expected_result = expected_result.unsqueeze(0)

    result = from_3by3_to_4by4(affine_matrix)

    assert expected_result.shape == result.shape
    assert torch.allclose(result, expected_result)

    channel_affine_matrix = torch.tensor(
        [
            [[2.3, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0], [0, 1, 0], [0, 0, 1]],
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        ],
        dtype=float,
    )
    expected_result = torch.tensor(
        [
            [
                [2.3, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ],
        dtype=float,
    )

    result = from_3by3_to_4by4(channel_affine_matrix)
    assert expected_result.shape == result.shape
    assert torch.allclose(result, expected_result)

    batch_channel_affine_matrix = channel_affine_matrix.unsqueeze(0).repeat(4, 1, 1, 1)
    expected_result = expected_result.unsqueeze(0).repeat(4, 1, 1, 1)
    result = from_3by3_to_4by4(batch_channel_affine_matrix)
    assert expected_result.shape == result.shape
    assert torch.allclose(result, expected_result)


def test_gaussian_heatmap_generator():
    # Create a generator
    generator = GaussianHeatmapGenerator(
        nb_landmarks=landmarks.shape[1],
        sigmas=1.0,
        heatmap_size=(64, 64),
        learnable=False,
        gamma=None,
        device="cpu",
    )

    # Call generator method
    heatmaps = generator(landmarks)

    # Check that the output has the correct shape
    assert heatmaps.shape == (1, 3, 64, 64)

    # Check that the output values are within the expected range
    assert torch.all(heatmaps >= 0.0)  # No negative values
    assert torch.all(heatmaps <= 1.0)  # Because gamma is None

    # Check that the output values are non-zero where expected
    # Because gamma = None and we don't deal with subpixel accuracy, we expect the heatmap to be 1 at the center of the
    # landmark.
    for i in range(3):
        assert heatmaps[0, i, landmarks[0, i, 0], landmarks[0, i, 1]] == 1

    # Call the generator method
    heatmaps = generator(landmarks_batch)

    # Check that the output has the correct shape
    assert heatmaps.shape == (4, 3, 64, 64)

    # Check that the output values are within the expected range
    assert torch.all(heatmaps >= 0.0)  # No negative values
    assert torch.all(heatmaps <= 1.0)  # Because gamma is None

    # Check that the output values are non-zero where expected
    for i in range(4):
        for j in range(landmarks_batch.shape[1]):
            assert heatmaps[i, j, landmarks_batch[i, j, 0], landmarks_batch[i, j, 1]] == 1


def test_gaussian_heatmap_generator_multiple_instances():
    landmarks_multiple_instances = torch.tensor(
        [
            [[[10, 20], [30, 40], [50, 60]], [[11, 22], [24, 35], [52, 58]]],
            [[[12, 24], [32, 42], [54, 60]], [[13, 26], [36, 46], [56, 61]]],
        ],
        dtype=torch.int,
    )

    assert landmarks_multiple_instances.shape == (2, 2, 3, 2)
    # Create a generator
    generator = GaussianHeatmapGenerator(
        nb_landmarks=landmarks_multiple_instances.shape[1],
        sigmas=1.0,
        heatmap_size=(64, 64),
        learnable=False,
        gamma=None,
        device="cpu",
    )

    # Call generator method
    heatmaps = generator(landmarks_multiple_instances)

    # Check that the output has the correct shape
    assert heatmaps.shape == (2, 2, 64, 64)

    # Check that the output values are within the expected range
    assert torch.all(heatmaps >= 0.0)  # No negative values

    # Check that the output values are non-zero where expected
    # Because gamma = None and we don't deal with subpixel accuracy, we expect the heatmap to be 1 at the center of the
    # landmark.
    for b in range(2):
        for c in range(2):
            for i in range(3):
                assert (
                    heatmaps[
                        b,
                        c,
                        landmarks_multiple_instances[b, c, i, 0],
                        landmarks_multiple_instances[b, c, i, 1],
                    ]
                    == 1
                )


def test_gaussian_heatmap_generator_multiple_instances_with_nan():
    landmarks_multiple_instances = torch.tensor(
        [
            [
                [[10, 20], [30, 40], [torch.nan, torch.nan]],
                [[11, 22], [24, 35], [torch.nan, torch.nan]],
            ],
            [
                [[12, 24], [32, 42], [torch.nan, torch.nan]],
                [[13, 26], [36, 46], [torch.nan, torch.nan]],
            ],
        ],
        dtype=torch.float,
    )

    assert landmarks_multiple_instances.shape == (2, 2, 3, 2)
    # Create a generator
    generator = GaussianHeatmapGenerator(
        nb_landmarks=landmarks_multiple_instances.shape[1],
        sigmas=1.0,
        heatmap_size=(64, 64),
        learnable=False,
        gamma=None,
        device="cpu",
    )

    # Call generator method
    heatmaps = generator(landmarks_multiple_instances)

    # Check that the output has the correct shape
    assert heatmaps.shape == (2, 2, 64, 64)

    # Check that the output values are within the expected range
    assert torch.all(heatmaps >= 0.0)  # No negative values

    # Check that the output values are non-zero where expected
    # Because gamma = None and we don't deal with subpixel accuracy, we expect the heatmap to be 1 at the center of the
    # landmark.
    for b in range(2):
        for c in range(2):
            for i in range(2):
                assert (
                    heatmaps[
                        b,
                        c,
                        int(landmarks_multiple_instances[b, c, i, 0]),
                        int(landmarks_multiple_instances[b, c, i, 1]),
                    ]
                    == 1
                )


# def test_gaussian_heatmap_generator_no_full_map():
#     generator = GaussianHeatmapGenerator(nb_landmarks=landmarks.shape[1], sigmas=1.0, heatmap_size=(64, 64), learnable=False,
#                                             gamma=None, device="cpu", full_map=False)

#     # Call the generator method
#     heatmaps = generator(landmarks)

#     # Check that the output has the correct shape
#     assert heatmaps.shape == (1, 3, 64, 64)

#     # Check that the output values are within the expected range
#     assert torch.all(heatmaps >= 0.0) # No negative values
#     assert torch.all(heatmaps <= 1.0) # Because gamma is None

#     # Check that the output values are non-zero where expected
#     # Because gamma = None and we don't deal with subpixel accuracy, we expect the heatmap to be 1 at the center of the
#     # landmark.
#     for i in range(3):
#         assert heatmaps[0, i, landmarks[0, i, 0], landmarks[0, i, 1]] == 1

#     # Call the generator method
#     heatmaps = generator(landmarks_batch)

#     # Check that the output has the correct shape
#     assert heatmaps.shape == (4, 3, 64, 64)

#     # Check that the output values are within the expected range
#     assert torch.all(heatmaps >= 0.0) # No negative values
#     assert torch.all(heatmaps <= 1.0) # Because gamma is None


#     # Check that the output values are non-zero where expected
#     for i in range(4):
#         for j in range(landmarks_batch.shape[1]):
#             assert heatmaps[i, j, landmarks_batch[i, j, 0], landmarks_batch[i, j, 1]] == 1


def test_gaussian_heatmap_generator_subpixel():
    landmarks_batch_ = landmarks_batch + torch.rand(landmarks_batch.shape)

    # Create a generator
    generator = GaussianHeatmapGenerator(
        nb_landmarks=landmarks_batch_.shape[1],
        sigmas=1.0,
        heatmap_size=(64, 64),
        learnable=False,
        gamma=None,
        device="cpu",
    )

    # Call the generator method
    heatmaps = generator(landmarks_batch_)

    # Check that the output has the correct shape
    assert heatmaps.shape == (4, 3, 64, 64)

    # Check that the output values are within the expected range
    assert torch.all(heatmaps >= 0.0)  # No negative values
    assert torch.all(heatmaps <= 1.0)  # Because gamma is None

    # Check that the output values are non-zero where expected
    pred_landmarks_discrete = coord_argmax(heatmaps)
    assert torch.equal(landmarks_batch_.round(), pred_landmarks_discrete)


def test_gaussian_heatmap_generator_learnable():
    # Create a generator
    generator = GaussianHeatmapGenerator(
        nb_landmarks=landmarks.shape[1],
        sigmas=1.0,
        heatmap_size=(64, 64),
        learnable=True,
        gamma=None,
        device="cpu",
    )

    # Call the generator method
    heatmaps = generator(landmarks_batch)

    # Check that the output has the correct shape
    assert heatmaps.shape == (4, 3, 64, 64)

    # Check that the output values are within the expected range
    assert torch.all(heatmaps >= 0.0)  # No negative values
    assert torch.all(heatmaps <= 1.0)  # Because gamma is None

    # Check that the output values are non-zero where expected
    for i in range(4):
        for j in range(landmarks_batch.shape[1]):
            assert heatmaps[i, j, landmarks_batch[i, j, 0], landmarks_batch[i, j, 1]] == 1


def test_gaussian_heatmap_generator_background():
    # Create a generator
    generator = GaussianHeatmapGenerator(
        nb_landmarks=landmarks.shape[1],
        sigmas=1.0,
        heatmap_size=(64, 64),
        learnable=True,
        gamma=None,
        background=True,
        device="cpu",
    )

    # Call the generator method
    heatmaps = generator(landmarks_batch)

    # Check that the output has the correct shape
    assert heatmaps.shape == (4, landmarks.shape[1] + 1, 64, 64)

    # Check that the output values are within the expected range
    assert torch.all(heatmaps >= 0.0)  # No negative values
    assert torch.all(heatmaps <= 1.0)  # Because gamma is None

    # check if background is first channel
    assert torch.equal(heatmaps[:, 0], 1 - heatmaps[:, 1:].sum(dim=1))

    # Check that the output values are non-zero where expected
    for i in range(4):
        for j in range(landmarks_batch.shape[1]):
            assert heatmaps[i, j + 1, landmarks_batch[i, j, 0], landmarks_batch[i, j, 1]] == 1


def test_gaussian_heatmap_generator_all_points():
    # Create a generator
    generator = GaussianHeatmapGenerator(
        nb_landmarks=landmarks.shape[1],
        sigmas=1.0,
        heatmap_size=(64, 64),
        learnable=True,
        gamma=None,
        background=False,
        all_points=True,
        device="cpu",
    )

    # Call the generator method
    heatmaps = generator(landmarks_batch)

    # Check that the output has the correct shape
    assert heatmaps.shape == (4, landmarks.shape[1] + 1, 64, 64)

    # Check that the output values are within the expected range
    assert torch.all(heatmaps >= 0.0)  # No negative values
    assert torch.all(heatmaps <= 1.0)  # Because gamma is None

    # Check if all_points is first channel
    assert torch.equal(heatmaps[:, 0], heatmaps[:, 1:].sum(dim=1))

    # Check that the output values are non-zero where expected
    for i in range(4):
        for j in range(landmarks_batch.shape[1]):
            assert heatmaps[i, j + 1, landmarks_batch[i, j, 0], landmarks_batch[i, j, 1]] == 1


def test_gaussian_heatmap_generator_not_continuous():
    landmarks_batch_ = landmarks_batch + torch.rand(landmarks_batch.shape)

    # Create a generator
    generator = GaussianHeatmapGenerator(
        nb_landmarks=landmarks_batch_.shape[1],
        sigmas=1.0,
        heatmap_size=(64, 64),
        learnable=False,
        gamma=None,
        continuous=False,
        device="cpu",
    )

    # Call the generator method
    heatmaps = generator(landmarks_batch_)

    # Check that the output has the correct shape
    assert heatmaps.shape == (4, 3, 64, 64)

    # Check that the output values are within the expected range
    assert torch.all(heatmaps >= 0.0)  # No negative values
    assert torch.all(heatmaps <= 1.0)  # Because gamma is None

    # Check that the output values are non-zero where expected
    for i in range(4):
        for j in range(landmarks_batch.shape[1]):
            assert (
                heatmaps[
                    i,
                    j,
                    landmarks_batch_[i, j, 0].round().int(),
                    landmarks_batch_[i, j, 1].round().int(),
                ]
                == 1
            )


def test_gaussian_heatmap_generator_assymetric_sigmas():
    # Create a generator
    generator = GaussianHeatmapGenerator(
        nb_landmarks=landmarks.shape[1],
        sigmas=np.array([[1, 0.5], [0.5, 1], [1.5, 2.1]]),
        heatmap_size=(64, 64),
        learnable=False,
        gamma=None,
        device="cpu",
    )

    # Call generator method
    heatmaps = generator(landmarks)

    # Check that the output has the correct shape
    assert heatmaps.shape == (1, 3, 64, 64)

    # Check that the output values are within the expected range
    assert torch.all(heatmaps >= 0.0)  # No negative values
    assert torch.all(heatmaps <= 1.0)  # Because gamma is None

    # Check that the output values are non-zero where expected
    # Because gamma = None and we don't deal with subpixel accuracy, we expect the heatmap to be 1 at the center of the
    # landmark.
    for i in range(3):
        assert heatmaps[0, i, landmarks[0, i, 0], landmarks[0, i, 1]] == 1

    # Call the generator method
    heatmaps = generator(landmarks_batch)

    # Check that the output has the correct shape
    assert heatmaps.shape == (4, 3, 64, 64)

    # Check that the output values are within the expected range
    assert torch.all(heatmaps >= 0.0)  # No negative values
    assert torch.all(heatmaps <= 1.0)  # Because gamma is None

    # Check that the output values are non-zero where expected
    for i in range(4):
        for j in range(landmarks_batch.shape[1]):
            assert heatmaps[i, j, landmarks_batch[i, j, 0], landmarks_batch[i, j, 1]] == 1

    # Check the assymetry
    for i in range(4):
        assert (
            heatmaps[i, 0, landmarks_batch[i, 0, 0] + 1, landmarks_batch[i, 0, 1]]
            > heatmaps[i, 0, landmarks_batch[i, 0, 0], landmarks_batch[i, 0, 1] + 1]
        )
        assert (
            heatmaps[i, 1, landmarks_batch[i, 1, 0] + 1, landmarks_batch[i, 1, 1]]
            < heatmaps[i, 1, landmarks_batch[i, 1, 0], landmarks_batch[i, 1, 1] + 1]
        )
        assert (
            heatmaps[i, 2, landmarks_batch[i, 2, 0] + 1, landmarks_batch[i, 2, 1]]
            < heatmaps[i, 2, landmarks_batch[i, 2, 0], landmarks_batch[i, 2, 1] + 1]
        )


def test_gaussian_heatmap_generator_rotation():
    # Create a generator
    generator = GaussianHeatmapGenerator(
        nb_landmarks=landmarks.shape[1],
        sigmas=np.array([[0.5, 1], [0.5, 1], [1.5, 2.1]]),
        rotation=np.array([30, 45, 90]),
        heatmap_size=(64, 64),
        learnable=False,
        gamma=None,
        device="cpu",
    )

    # Call generator method
    heatmaps = generator(landmarks)

    # Check that the output has the correct shape
    assert heatmaps.shape == (1, 3, 64, 64)

    # Check that the output values are within the expected range
    assert torch.all(heatmaps >= 0.0)  # No negative values
    assert torch.all(heatmaps <= 1.0)  # Because gamma is None

    # Check that the output values are non-zero where expected
    # Because gamma = None and we don't deal with subpixel accuracy, we expect the heatmap to be 1 at the center of the
    # landmark.
    for i in range(3):
        assert heatmaps[0, i, landmarks[0, i, 0], landmarks[0, i, 1]] == 1

    # Call the generator method
    heatmaps = generator(landmarks_batch)

    # Check that the output has the correct shape
    assert heatmaps.shape == (4, 3, 64, 64)

    # Check that the output values are within the expected range
    assert torch.all(heatmaps >= 0.0)  # No negative values
    assert torch.all(heatmaps <= 1.0)  # Because gamma is None

    # Check that the output values are non-zero where expected
    for i in range(4):
        for j in range(landmarks_batch.shape[1]):
            assert heatmaps[i, j, landmarks_batch[i, j, 0], landmarks_batch[i, j, 1]] == 1

    # Check the rotateed assymetry
    for i in range(4):
        # point 1: rotatation of 30°, point 2: 45°
        assert (
            heatmaps[i, 1, landmarks_batch[i, 1, 0] + 1, landmarks_batch[i, 1, 1]]
            < heatmaps[i, 0, landmarks_batch[i, 0, 0] + 1, landmarks_batch[i, 0, 1]]
        )
        assert (
            heatmaps[i, 1, landmarks_batch[i, 1, 0], landmarks_batch[i, 1, 1] + 1]
            > heatmaps[i, 0, landmarks_batch[i, 0, 0], landmarks_batch[i, 0, 1] + 1]
        )
        # point 3: rotatation of 90°
        assert (
            heatmaps[i, 2, landmarks_batch[i, 2, 0] + 1, landmarks_batch[i, 2, 1]]
            > heatmaps[i, 2, landmarks_batch[i, 2, 0], landmarks_batch[i, 2, 1] + 1]
        )


def test_gaussian_heatmap_generator_adaptive():
    # Create a generator
    generator = GaussianHeatmapGenerator(
        nb_landmarks=landmarks.shape[1],
        sigmas=np.array([[1, 0.5], [0.5, 1], [1.5, 2.1]]),
        heatmap_size=(64, 64),
        learnable=False,
        gamma=None,
        device="cpu",
    )

    # Change sigmas and rotation
    generator.set_sigmas(np.array([[0.5, 1], [0.5, 1], [1.5, 2.1]]))
    generator.set_rotation(np.array([30, 45, 90]))

    # Call generator method
    heatmaps = generator(landmarks_batch)

    # Check that the output has the correct shape
    assert heatmaps.shape == (4, 3, 64, 64)

    # Check that the output values are within the expected range
    assert torch.all(heatmaps >= 0.0)  # No negative values
    assert torch.all(heatmaps <= 1.0)  # Because gamma is None

    # Check that the output values are non-zero where expected
    for i in range(4):
        for j in range(landmarks_batch.shape[1]):
            assert heatmaps[i, j, landmarks_batch[i, j, 0], landmarks_batch[i, j, 1]] == 1

    # Check the rotateed assymetry
    for i in range(4):
        # point 1: rotatation of 30°, point 2: 45°
        assert (
            heatmaps[i, 1, landmarks_batch[i, 1, 0] + 1, landmarks_batch[i, 1, 1]]
            < heatmaps[i, 0, landmarks_batch[i, 0, 0] + 1, landmarks_batch[i, 0, 1]]
        )
        assert (
            heatmaps[i, 1, landmarks_batch[i, 1, 0], landmarks_batch[i, 1, 1] + 1]
            > heatmaps[i, 0, landmarks_batch[i, 0, 0], landmarks_batch[i, 0, 1] + 1]
        )
        # point 3: rotatation of 90°
        assert (
            heatmaps[i, 2, landmarks_batch[i, 2, 0] + 1, landmarks_batch[i, 2, 1]]
            > heatmaps[i, 2, landmarks_batch[i, 2, 0], landmarks_batch[i, 2, 1] + 1]
        )


def test_gaussian_heatmap_generator_affine():
    # Create a generator
    generator = GaussianHeatmapGenerator(
        nb_landmarks=landmarks.shape[1],
        sigmas=np.array([[1, 0.5], [0.5, 1], [1.5, 2.1]]),
        heatmap_size=(64, 64),
        learnable=False,
        gamma=None,
        device="cpu",
    )

    # Set affine matrix
    affine_matrix = torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float
    )

    # Call generator method
    heatmaps = generator(landmarks, affine_matrix=affine_matrix)

    # Check that the output has the correct shape
    assert heatmaps.shape == (1, 3, 64, 64)

    # Check that the output values are within the expected range
    assert torch.all(heatmaps >= 0.0)  # No negative values
    assert torch.all(heatmaps <= 1.0)  # Because gamma is None

    # Check that the output values are non-zero where expected
    # Because gamma = None and we don't deal with subpixel accuracy, we expect the heatmap to be 1 at the center of the
    # landmark.
    for i in range(3):
        assert heatmaps[0, i, landmarks[0, i, 0], landmarks[0, i, 1]] == 1

    # Call the generator method
    heatmaps = generator(landmarks_batch)

    # Check that the output has the correct shape
    assert heatmaps.shape == (4, 3, 64, 64)

    # Check that the output values are within the expected range
    assert torch.all(heatmaps >= 0.0)  # No negative values
    assert torch.all(heatmaps <= 1.0)  # Because gamma is None

    # Check that the output values are non-zero where expected
    for i in range(4):
        for j in range(landmarks_batch.shape[1]):
            assert heatmaps[i, j, landmarks_batch[i, j, 0], landmarks_batch[i, j, 1]] == 1

    # Check the assymetry
    for i in range(4):
        assert (
            heatmaps[i, 0, landmarks_batch[i, 0, 0] + 1, landmarks_batch[i, 0, 1]]
            > heatmaps[i, 0, landmarks_batch[i, 0, 0], landmarks_batch[i, 0, 1] + 1]
        )
        assert (
            heatmaps[i, 1, landmarks_batch[i, 1, 0] + 1, landmarks_batch[i, 1, 1]]
            < heatmaps[i, 1, landmarks_batch[i, 1, 0], landmarks_batch[i, 1, 1] + 1]
        )
        assert (
            heatmaps[i, 2, landmarks_batch[i, 2, 0] + 1, landmarks_batch[i, 2, 1]]
            < heatmaps[i, 2, landmarks_batch[i, 2, 0], landmarks_batch[i, 2, 1] + 1]
        )

    # Set affine matrix (rotation of 90°)
    affine_matrix = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float
    )
    # Call generator method
    heatmaps = generator(landmarks_batch, affine_matrix=affine_matrix)

    # Check that the output has the correct shape
    assert heatmaps.shape == (4, 3, 64, 64)

    # Check that the output values are within the expected range
    assert torch.all(heatmaps >= 0.0)  # No negative values
    assert torch.all(heatmaps <= 1.0)  # Because gamma is None

    # Check that the output values are non-zero where expected
    for i in range(4):
        for j in range(landmarks_batch.shape[1]):
            assert heatmaps[i, j, landmarks_batch[i, j, 0], landmarks_batch[i, j, 1]] == 1

    # Check the assymetry
    for i in range(4):
        assert (
            heatmaps[i, 0, landmarks_batch[i, 0, 0] + 1, landmarks_batch[i, 0, 1]]
            < heatmaps[i, 0, landmarks_batch[i, 0, 0], landmarks_batch[i, 0, 1] + 1]
        )
        assert (
            heatmaps[i, 1, landmarks_batch[i, 1, 0] + 1, landmarks_batch[i, 1, 1]]
            > heatmaps[i, 1, landmarks_batch[i, 1, 0], landmarks_batch[i, 1, 1] + 1]
        )
        assert (
            heatmaps[i, 2, landmarks_batch[i, 2, 0] + 1, landmarks_batch[i, 2, 1]]
            > heatmaps[i, 2, landmarks_batch[i, 2, 0], landmarks_batch[i, 2, 1] + 1]
        )

    # Set affine matrix (rotation of 90°)
    affine_matrix = torch.tensor(
        [
            [[np.cos(np.pi / 6), -np.sin(np.pi / 6)], [np.sin(np.pi / 6), np.cos(np.pi / 6)]],
            [[np.cos(np.pi / 4), -np.sin(np.pi / 4)], [np.sin(np.pi / 4), np.cos(np.pi / 4)]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, -1.0], [1.0, 0.0]],
        ],
        dtype=torch.float,
    )

    # Call generator method
    heatmaps = generator(landmarks_batch, affine_matrix=affine_matrix)

    # Check that the output has the correct shape
    assert heatmaps.shape == (4, 3, 64, 64)

    # Check that the output values are within the expected range
    assert torch.all(heatmaps >= 0.0)  # No negative values
    assert torch.all(heatmaps <= 1.0)  # Because gamma is None

    # Check that the output values are non-zero where expected
    for i in range(4):
        for j in range(landmarks_batch.shape[1]):
            assert heatmaps[i, j, landmarks_batch[i, j, 0], landmarks_batch[i, j, 1]] == 1


def test_laplacian_heatmap_generator():
    # Create a generator
    generator = LaplacianHeatmapGenerator(
        nb_landmarks=landmarks.shape[1],
        sigmas=1.0,
        heatmap_size=(64, 64),
        learnable=False,
        gamma=None,
        device="cpu",
    )

    # Call generator method
    heatmaps = generator(landmarks)

    # Check that the output has the correct shape
    assert heatmaps.shape == (1, 3, 64, 64)

    # Check that the output values are within the expected range
    assert torch.all(heatmaps >= 0.0)  # No negative values
    assert torch.all(heatmaps <= 1.0)  # Because gamma is None

    # Check that the output values are non-zero where expected
    # Because gamma = None and we don't deal with subpixel accuracy, we expect the heatmap to be 1 at the center of the
    # landmark.
    for i in range(3):
        assert heatmaps[0, i, landmarks[0, i, 0], landmarks[0, i, 1]] == 1

    # Call the generator method
    heatmaps = generator(landmarks_batch)

    # Check that the output has the correct shape
    assert heatmaps.shape == (4, 3, 64, 64)

    # Check that the output values are within the expected range
    assert torch.all(heatmaps >= 0.0)  # No negative values
    assert torch.all(heatmaps <= 1.0)  # Because gamma is None

    # Check that the output values are non-zero where expected
    for i in range(4):
        for j in range(landmarks_batch.shape[1]):
            assert heatmaps[i, j, landmarks_batch[i, j, 0], landmarks_batch[i, j, 1]] == 1


def test_laplacian_heatmap_generator_with_gamma():
    for gamma in range(1, 4):
        generator = LaplacianHeatmapGenerator(
            nb_landmarks=landmarks.shape[1],
            sigmas=1.0,
            heatmap_size=(64, 64),
            learnable=False,
            gamma=gamma,
            device="cpu",
        )

        # Call the generator method
        heatmaps = generator(landmarks)

        # Check that the output has the correct shape
        assert heatmaps.shape == (1, 3, 64, 64)

        # Check that the output values are within the expected range
        assert torch.all(heatmaps >= 0.0)  # No negative values
        assert torch.all(heatmaps <= (gamma / ((2 / 3) * torch.pi)))

        # Check that the output values are non-zero where expected
        for i in range(3):
            assert pytest.approx(heatmaps[0, i, landmarks[0, i, 0], landmarks[0, i, 1]], 0.01) == (
                gamma / ((2 / 3) * torch.pi)
            )

        # Repeat same tests for batch
        # Call the generator method
        heatmaps = generator(landmarks_batch)

        # Check that the output has the correct shape
        assert heatmaps.shape == (4, 3, 64, 64)

        # Check that the output values are within the expected range
        assert torch.all(heatmaps >= 0.0)
        assert torch.all(heatmaps <= (gamma / ((2 / 3) * torch.pi)))

        # Check that the output values are non-zero where expected
        for i in range(4):
            for j in range(landmarks_batch.shape[1]):
                assert pytest.approx(
                    heatmaps[i, j, landmarks_batch[i, j, 0], landmarks_batch[i, j, 1]], 0.01
                ) == (gamma / ((2 / 3) * torch.pi))
                assert pytest.approx(
                    heatmaps[i, j, landmarks_batch[i, j, 0] + 1, landmarks_batch[i, j, 1]], 0.01
                ) == (gamma / ((2 / 3) * torch.pi) * np.exp(-np.sqrt(3)))


# def test_laplacian_heatmap_generator_no_full_map():
#     generator = LaplacianHeatmapGenerator(nb_landmarks=landmarks.shape[1], sigmas=1.0, heatmap_size=(64, 64), learnable=False,
#                                             gamma=None, device="cpu", full_map=False)

#     # Call the generator method
#     heatmaps = generator(landmarks)

#     # Check that the output has the correct shape
#     assert heatmaps.shape == (1, 3, 64, 64)

#     # Check that the output values are within the expected range
#     assert torch.all(heatmaps >= 0.0) # No negative values
#     assert torch.all(heatmaps <= 1.0) # Because gamma is None

#     # Check that the output values are non-zero where expected
#     # Because gamma = None and we don't deal with subpixel accuracy, we expect the heatmap to be 1 at the center of the
#     # landmark.
#     for i in range(3):
#         assert heatmaps[0, i, landmarks[0, i, 0], landmarks[0, i, 1]] == 1

#     # Call the generator method
#     heatmaps = generator(landmarks_batch)

#     # Check that the output has the correct shape
#     assert heatmaps.shape == (4, 3, 64, 64)

#     # Check that the output values are within the expected range
#     assert torch.all(heatmaps >= 0.0) # No negative values
#     assert torch.all(heatmaps <= 1.0) # Because gamma is None


#     # Check that the output values are non-zero where expected
#     for i in range(4):
#         for j in range(landmarks_batch.shape[1]):
#             assert heatmaps[i, j, landmarks_batch[i, j, 0], landmarks_batch[i, j, 1]] == 1


def test_laplacian_heatmap_generator_subpixel():
    landmarks_batch_ = landmarks_batch + torch.rand(landmarks_batch.shape)

    # Create a generator
    generator = LaplacianHeatmapGenerator(
        nb_landmarks=landmarks_batch_.shape[1],
        sigmas=1.0,
        heatmap_size=(64, 64),
        learnable=False,
        gamma=None,
        device="cpu",
    )

    # Call the generator method
    heatmaps = generator(landmarks_batch_)

    # Check that the output has the correct shape
    assert heatmaps.shape == (4, 3, 64, 64)

    # Check that the output values are within the expected range
    assert torch.all(heatmaps >= 0.0)  # No negative values
    assert torch.all(heatmaps <= 1.0)  # Because gamma is None

    # Check that the output values are non-zero where expected
    pred_landmarks_discrete = coord_argmax(heatmaps)
    assert torch.equal(landmarks_batch_.round(), pred_landmarks_discrete)


def test_laplacian_heatmap_generator_learnable():
    # Create a generator
    generator = LaplacianHeatmapGenerator(
        nb_landmarks=landmarks.shape[1],
        sigmas=1.0,
        heatmap_size=(64, 64),
        learnable=True,
        gamma=None,
        device="cpu",
    )

    # Call the generator method
    heatmaps = generator(landmarks_batch)

    # Check that the output has the correct shape
    assert heatmaps.shape == (4, 3, 64, 64)

    # Check that the output values are within the expected range
    assert torch.all(heatmaps >= 0.0)  # No negative values
    assert torch.all(heatmaps <= 1.0)  # Because gamma is None

    # Check that the output values are non-zero where expected
    for i in range(4):
        for j in range(landmarks_batch.shape[1]):
            assert heatmaps[i, j, landmarks_batch[i, j, 0], landmarks_batch[i, j, 1]] == 1


def test_laplacian_heatmap_generator_background():
    # Create a generator
    generator = LaplacianHeatmapGenerator(
        nb_landmarks=landmarks.shape[1],
        sigmas=1.0,
        heatmap_size=(64, 64),
        learnable=True,
        gamma=None,
        background=True,
        device="cpu",
    )

    # Call the generator method
    heatmaps = generator(landmarks_batch)

    # Check that the output has the correct shape
    assert heatmaps.shape == (4, landmarks.shape[1] + 1, 64, 64)

    # Check that the output values are within the expected range
    assert torch.all(heatmaps >= 0.0)  # No negative values
    assert torch.all(heatmaps <= 1.0)  # Because gamma is None

    # check if background is first channel
    assert torch.equal(heatmaps[:, 0], 1 - heatmaps[:, 1:].sum(dim=1))

    # Check that the output values are non-zero where expected
    for i in range(4):
        for j in range(landmarks_batch.shape[1]):
            assert heatmaps[i, j + 1, landmarks_batch[i, j, 0], landmarks_batch[i, j, 1]] == 1


def test_laplacian_heatmap_generator_all_points():
    # Create a generator
    generator = LaplacianHeatmapGenerator(
        nb_landmarks=landmarks.shape[1],
        sigmas=1.0,
        heatmap_size=(64, 64),
        learnable=True,
        gamma=None,
        background=False,
        all_points=True,
        device="cpu",
    )

    # Call the generator method
    heatmaps = generator(landmarks_batch)

    # Check that the output has the correct shape
    assert heatmaps.shape == (4, landmarks.shape[1] + 1, 64, 64)

    # Check that the output values are within the expected range
    assert torch.all(heatmaps >= 0.0)  # No negative values
    assert torch.all(heatmaps <= 1.0)  # Because gamma is None

    # Check if all_points is first channel
    assert torch.equal(heatmaps[:, 0], heatmaps[:, 1:].sum(dim=1))

    # Check that the output values are non-zero where expected
    for i in range(4):
        for j in range(landmarks_batch.shape[1]):
            assert heatmaps[i, j + 1, landmarks_batch[i, j, 0], landmarks_batch[i, j, 1]] == 1


def test_laplacian_heatmap_generator_not_continuous():
    landmarks_batch_ = landmarks_batch + torch.rand(landmarks_batch.shape)

    # Create a generator
    generator = LaplacianHeatmapGenerator(
        nb_landmarks=landmarks_batch_.shape[1],
        sigmas=1.0,
        heatmap_size=(64, 64),
        learnable=False,
        gamma=None,
        continuous=False,
        device="cpu",
    )

    # Call the generator method
    heatmaps = generator(landmarks_batch_)

    # Check that the output has the correct shape
    assert heatmaps.shape == (4, 3, 64, 64)

    # Check that the output values are within the expected range
    assert torch.all(heatmaps >= 0.0)  # No negative values
    assert torch.all(heatmaps <= 1.0)  # Because gamma is None

    # Check that the output values are non-zero where expected
    for i in range(4):
        for j in range(landmarks_batch.shape[1]):
            assert (
                heatmaps[
                    i,
                    j,
                    landmarks_batch_[i, j, 0].round().int(),
                    landmarks_batch_[i, j, 1].round().int(),
                ]
                == 1
            )


def test_laplacian_heatmap_generator_assymetric_sigmas():
    # Create a generator
    generator = LaplacianHeatmapGenerator(
        nb_landmarks=landmarks.shape[1],
        sigmas=np.array([[1, 0.5], [0.5, 1], [1.5, 2.1]]),
        heatmap_size=(64, 64),
        learnable=False,
        gamma=None,
        device="cpu",
    )

    # Call generator method
    heatmaps = generator(landmarks)

    # Check that the output has the correct shape
    assert heatmaps.shape == (1, 3, 64, 64)

    # Check that the output values are within the expected range
    assert torch.all(heatmaps >= 0.0)  # No negative values
    assert torch.all(heatmaps <= 1.0)  # Because gamma is None

    # Check that the output values are non-zero where expected
    # Because gamma = None and we don't deal with subpixel accuracy, we expect the heatmap to be 1 at the center of the
    # landmark.
    for i in range(3):
        assert heatmaps[0, i, landmarks[0, i, 0], landmarks[0, i, 1]] == 1

    # Call the generator method
    heatmaps = generator(landmarks_batch)

    # Check that the output has the correct shape
    assert heatmaps.shape == (4, 3, 64, 64)

    # Check that the output values are within the expected range
    assert torch.all(heatmaps >= 0.0)  # No negative values
    assert torch.all(heatmaps <= 1.0)  # Because gamma is None

    # Check that the output values are non-zero where expected
    for i in range(4):
        for j in range(landmarks_batch.shape[1]):
            assert heatmaps[i, j, landmarks_batch[i, j, 0], landmarks_batch[i, j, 1]] == 1

    # Check the assymetry
    for i in range(4):
        assert (
            heatmaps[i, 0, landmarks_batch[i, 0, 0] + 1, landmarks_batch[i, 0, 1]]
            > heatmaps[i, 0, landmarks_batch[i, 0, 0], landmarks_batch[i, 0, 1] + 1]
        )
        assert (
            heatmaps[i, 1, landmarks_batch[i, 1, 0] + 1, landmarks_batch[i, 1, 1]]
            < heatmaps[i, 1, landmarks_batch[i, 1, 0], landmarks_batch[i, 1, 1] + 1]
        )
        assert (
            heatmaps[i, 2, landmarks_batch[i, 2, 0] + 1, landmarks_batch[i, 2, 1]]
            < heatmaps[i, 2, landmarks_batch[i, 2, 0], landmarks_batch[i, 2, 1] + 1]
        )


def test_laplacian_heatmap_generator_rotation():
    # Create a generator
    generator = LaplacianHeatmapGenerator(
        nb_landmarks=landmarks.shape[1],
        sigmas=np.array([[0.5, 1], [0.5, 1], [1.5, 2.1]]),
        rotation=np.array([30, 45, 90]),
        heatmap_size=(64, 64),
        learnable=False,
        gamma=None,
        device="cpu",
    )

    # Call generator method
    heatmaps = generator(landmarks)

    # Check that the output has the correct shape
    assert heatmaps.shape == (1, 3, 64, 64)

    # Check that the output values are within the expected range
    assert torch.all(heatmaps >= 0.0)  # No negative values
    assert torch.all(heatmaps <= 1.0)  # Because gamma is None

    # Check that the output values are non-zero where expected
    # Because gamma = None and we don't deal with subpixel accuracy, we expect the heatmap to be 1 at the center of the
    # landmark.
    for i in range(3):
        assert heatmaps[0, i, landmarks[0, i, 0], landmarks[0, i, 1]] == 1

    # Call the generator method
    heatmaps = generator(landmarks_batch)

    # Check that the output has the correct shape
    assert heatmaps.shape == (4, 3, 64, 64)

    # Check that the output values are within the expected range
    assert torch.all(heatmaps >= 0.0)  # No negative values
    assert torch.all(heatmaps <= 1.0)  # Because gamma is None

    # Check that the output values are non-zero where expected
    for i in range(4):
        for j in range(landmarks_batch.shape[1]):
            assert heatmaps[i, j, landmarks_batch[i, j, 0], landmarks_batch[i, j, 1]] == 1

    # Check the rotateed assymetry
    for i in range(4):
        # point 1: rotatation of 30°, point 2: 45°
        assert (
            heatmaps[i, 1, landmarks_batch[i, 1, 0] + 1, landmarks_batch[i, 1, 1]]
            < heatmaps[i, 0, landmarks_batch[i, 0, 0] + 1, landmarks_batch[i, 0, 1]]
        )
        assert (
            heatmaps[i, 1, landmarks_batch[i, 1, 0], landmarks_batch[i, 1, 1] + 1]
            > heatmaps[i, 0, landmarks_batch[i, 0, 0], landmarks_batch[i, 0, 1] + 1]
        )
        # point 3: rotatation of 90°
        assert (
            heatmaps[i, 2, landmarks_batch[i, 2, 0] + 1, landmarks_batch[i, 2, 1]]
            > heatmaps[i, 2, landmarks_batch[i, 2, 0], landmarks_batch[i, 2, 1] + 1]
        )


def test_laplacian_heatmap_generator_adaptive():
    # Create a generator
    generator = LaplacianHeatmapGenerator(
        nb_landmarks=landmarks.shape[1],
        sigmas=np.array([[1, 0.5], [0.5, 1], [1.5, 2.1]]),
        heatmap_size=(64, 64),
        learnable=False,
        gamma=None,
        device="cpu",
    )

    # Change sigmas and rotation
    generator.set_sigmas(np.array([[0.5, 1], [0.5, 1], [1.5, 2.1]]))
    generator.set_rotation(np.array([30, 45, 90]))

    # Call generator method
    heatmaps = generator(landmarks_batch)

    # Check that the output has the correct shape
    assert heatmaps.shape == (4, 3, 64, 64)

    # Check that the output values are within the expected range
    assert torch.all(heatmaps >= 0.0)  # No negative values
    assert torch.all(heatmaps <= 1.0)  # Because gamma is None

    # Check that the output values are non-zero where expected
    for i in range(4):
        for j in range(landmarks_batch.shape[1]):
            assert heatmaps[i, j, landmarks_batch[i, j, 0], landmarks_batch[i, j, 1]] == 1

    # Check the rotateed assymetry
    for i in range(4):
        # point 1: rotatation of 30°, point 2: 45°
        assert (
            heatmaps[i, 1, landmarks_batch[i, 1, 0] + 1, landmarks_batch[i, 1, 1]]
            < heatmaps[i, 0, landmarks_batch[i, 0, 0] + 1, landmarks_batch[i, 0, 1]]
        )
        assert (
            heatmaps[i, 1, landmarks_batch[i, 1, 0], landmarks_batch[i, 1, 1] + 1]
            > heatmaps[i, 0, landmarks_batch[i, 0, 0], landmarks_batch[i, 0, 1] + 1]
        )
        # point 3: rotatation of 90°
        assert (
            heatmaps[i, 2, landmarks_batch[i, 2, 0] + 1, landmarks_batch[i, 2, 1]]
            > heatmaps[i, 2, landmarks_batch[i, 2, 0], landmarks_batch[i, 2, 1] + 1]
        )


def test_laplacian_heatmap_generator_affine():
    # Create a generator
    generator = LaplacianHeatmapGenerator(
        nb_landmarks=landmarks.shape[1],
        sigmas=np.array([[1, 0.5], [0.5, 1], [1.5, 2.1]]),
        heatmap_size=(64, 64),
        learnable=False,
        gamma=None,
        device="cpu",
    )

    # Set affine matrix
    affine_matrix = torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float
    )

    # Call generator method
    heatmaps = generator(landmarks, affine_matrix=affine_matrix)

    # Check that the output has the correct shape
    assert heatmaps.shape == (1, 3, 64, 64)

    # Check that the output values are within the expected range
    assert torch.all(heatmaps >= 0.0)  # No negative values
    assert torch.all(heatmaps <= 1.0)  # Because gamma is None

    # Check that the output values are non-zero where expected
    # Because gamma = None and we don't deal with subpixel accuracy, we expect the heatmap to be 1 at the center of the
    # landmark.
    for i in range(3):
        assert heatmaps[0, i, landmarks[0, i, 0], landmarks[0, i, 1]] == 1

    # Call the generator method
    heatmaps = generator(landmarks_batch)

    # Check that the output has the correct shape
    assert heatmaps.shape == (4, 3, 64, 64)

    # Check that the output values are within the expected range
    assert torch.all(heatmaps >= 0.0)  # No negative values
    assert torch.all(heatmaps <= 1.0)  # Because gamma is None

    # Check that the output values are non-zero where expected
    for i in range(4):
        for j in range(landmarks_batch.shape[1]):
            assert heatmaps[i, j, landmarks_batch[i, j, 0], landmarks_batch[i, j, 1]] == 1

    # Check the assymetry
    for i in range(4):
        assert (
            heatmaps[i, 0, landmarks_batch[i, 0, 0] + 1, landmarks_batch[i, 0, 1]]
            > heatmaps[i, 0, landmarks_batch[i, 0, 0], landmarks_batch[i, 0, 1] + 1]
        )
        assert (
            heatmaps[i, 1, landmarks_batch[i, 1, 0] + 1, landmarks_batch[i, 1, 1]]
            < heatmaps[i, 1, landmarks_batch[i, 1, 0], landmarks_batch[i, 1, 1] + 1]
        )
        assert (
            heatmaps[i, 2, landmarks_batch[i, 2, 0] + 1, landmarks_batch[i, 2, 1]]
            < heatmaps[i, 2, landmarks_batch[i, 2, 0], landmarks_batch[i, 2, 1] + 1]
        )

    # Set affine matrix (rotation of 90°)
    affine_matrix = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float
    )
    # Call generator method
    heatmaps = generator(landmarks_batch, affine_matrix=affine_matrix)

    # Check that the output has the correct shape
    assert heatmaps.shape == (4, 3, 64, 64)

    # Check that the output values are within the expected range
    assert torch.all(heatmaps >= 0.0)  # No negative values
    assert torch.all(heatmaps <= 1.0)  # Because gamma is None

    # Check that the output values are non-zero where expected
    for i in range(4):
        for j in range(landmarks_batch.shape[1]):
            assert heatmaps[i, j, landmarks_batch[i, j, 0], landmarks_batch[i, j, 1]] == 1

    # Check the assymetry
    for i in range(4):
        assert (
            heatmaps[i, 0, landmarks_batch[i, 0, 0] + 1, landmarks_batch[i, 0, 1]]
            < heatmaps[i, 0, landmarks_batch[i, 0, 0], landmarks_batch[i, 0, 1] + 1]
        )
        assert (
            heatmaps[i, 1, landmarks_batch[i, 1, 0] + 1, landmarks_batch[i, 1, 1]]
            > heatmaps[i, 1, landmarks_batch[i, 1, 0], landmarks_batch[i, 1, 1] + 1]
        )
        assert (
            heatmaps[i, 2, landmarks_batch[i, 2, 0] + 1, landmarks_batch[i, 2, 1]]
            > heatmaps[i, 2, landmarks_batch[i, 2, 0], landmarks_batch[i, 2, 1] + 1]
        )

    # Set affine matrix (rotation of 90°)
    affine_matrix = torch.tensor(
        [
            [[np.cos(np.pi / 6), -np.sin(np.pi / 6)], [np.sin(np.pi / 6), np.cos(np.pi / 6)]],
            [[np.cos(np.pi / 4), -np.sin(np.pi / 4)], [np.sin(np.pi / 4), np.cos(np.pi / 4)]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, -1.0], [1.0, 0.0]],
        ],
        dtype=torch.float,
    )

    # Call generator method
    heatmaps = generator(landmarks_batch, affine_matrix=affine_matrix)

    # Check that the output has the correct shape
    assert heatmaps.shape == (4, 3, 64, 64)

    # Check that the output values are within the expected range
    assert torch.all(heatmaps >= 0.0)  # No negative values
    assert torch.all(heatmaps <= 1.0)  # Because gamma is None

    # Check that the output values are non-zero where expected
    for i in range(4):
        for j in range(landmarks_batch.shape[1]):
            assert heatmaps[i, j, landmarks_batch[i, j, 0], landmarks_batch[i, j, 1]] == 1

    # TODO add more elaborate tests for affine matrix
