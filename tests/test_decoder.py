"""Tests for the decoder module."""

import numpy as np
import torch

from src.landmarker.heatmap.decoder import (
    coord_argmax,
    coord_cov_from_gaussian_ls,
    coord_cov_windowed_weigthed_sample_cov,
    coord_local_soft_argmax,
    coord_soft_argmax,
    coord_soft_argmax_cov,
    coord_weighted_spatial_mean,
    cov_from_gaussian_ls,
    heatmap_coord_to_weighted_sample_cov,
    heatmap_to_coord,
    heatmap_to_coord_cov,
    heatmap_to_coord_enlarge,
    heatmap_to_multiple_coord,
    non_maximum_surpression,
    non_maximum_surpression_local_soft_argmax,
)
from src.landmarker.heatmap.generator import GaussianHeatmapGenerator


def create_heatmap(subpixel=False, sigmas=1.0, rotations=0.0, return_covs=False, gamma=None):
    """Create a heatmap and corresponding landmarks for testing."""
    # Create a generator with some example parameters
    generator = GaussianHeatmapGenerator(
        nb_landmarks=3,
        sigmas=sigmas,
        rotation=rotations,
        heatmap_size=(64, 64),
        learnable=False,
        gamma=gamma,
        device="cpu",
    )

    # Create some example input data
    landmarks = torch.tensor([[[20, 20], [30, 40], [45, 25]]], dtype=torch.float)

    if subpixel:
        landmarks += torch.rand(landmarks.shape, dtype=torch.float32)

    # Call the method being tested
    heatmaps = generator(landmarks)

    if return_covs:
        return heatmaps, landmarks, generator.get_covariance_matrix()
    return heatmaps, landmarks


def create_batch_of_heatmaps(
    subpixel=False, sigmas=1.0, rotations=0.0, return_covs=False, gamma=None
):
    """Create a batch of heatmaps and corresponding landmarks for testing."""
    # Create a generator with some example parameters
    generator = GaussianHeatmapGenerator(
        nb_landmarks=3,
        sigmas=sigmas,
        rotation=rotations,
        heatmap_size=(64, 64),
        learnable=False,
        gamma=gamma,
        device="cpu",
    )

    landmarks_batch = torch.tensor(
        [
            [[10, 10], [30, 40], [50, 60]],
            [[16, 17], [24, 35], [30, 58]],
            [[24, 24], [14, 14], [54, 30]],
            [[30, 30], [36, 46], [20, 10]],
        ],
        dtype=torch.float,
    )

    if subpixel:
        landmarks_batch += torch.rand(landmarks_batch.shape, dtype=torch.float32)

    # Call the method being tested
    heatmaps = generator(landmarks_batch)

    if return_covs:
        return heatmaps, landmarks_batch, generator.get_covariance_matrix()
    return heatmaps, landmarks_batch


def check_retriever_output(retriever_fun, atol=1e-2, rtol=0, **kwargs):
    """Check the output of a retriever function."""
    # Create some example input data
    heatmaps, landmarks = create_heatmap(  # pylint: disable=unbalanced-tuple-unpacking
        subpixel=False, return_covs=False
    )

    # Call the method being tested
    coords = retriever_fun(heatmaps, **kwargs).float()

    # Check the output
    assert coords.shape == (1, 3, 2)
    assert torch.allclose(coords, landmarks, atol=atol, rtol=rtol)

    # Create some batched example input data
    heatmaps, landmarks = create_batch_of_heatmaps(  # pylint: disable=unbalanced-tuple-unpacking
        subpixel=False, return_covs=False
    )

    # Call the method being tested
    coords = retriever_fun(heatmaps, **kwargs).float()

    # Check the output
    assert coords.shape == (4, 3, 2)
    assert torch.allclose(coords, landmarks, atol=atol, rtol=rtol)

    # Create some batched example input data with subpixel landmarks
    heatmaps, landmarks = create_batch_of_heatmaps(  # pylint: disable=unbalanced-tuple-unpacking
        subpixel=True
    )

    # Call the method being tested
    coords = retriever_fun(heatmaps, **kwargs).float()

    # Check the output
    assert coords.shape == (4, 3, 2)
    assert torch.allclose(coords.round(), landmarks.round(), atol=atol)


def test_coord_argmax():
    """Test the coord_argmax function."""
    check_retriever_output(coord_argmax, atol=1e-2, rtol=0)


def test_coord_local_soft_argmax():
    """Test the coord_local_soft_argmax function."""
    check_retriever_output(coord_local_soft_argmax, atol=1e-2, window=3, t=10)

    check_retriever_output(coord_local_soft_argmax, atol=1e-2, window=5, t=5)

    check_retriever_output(coord_local_soft_argmax, atol=1e-2, window=7, t=3)


def test_coord_weighted_spatial_mean():
    """Test the coord_weighted_spatial_mean function."""
    # TODO: the method seems to be kind of unstable hence the high tolerance
    check_retriever_output(coord_weighted_spatial_mean, atol=3)


def test_coord_soft_argmax():
    """Test the coord_soft_argmax function."""
    check_retriever_output(coord_soft_argmax, atol=1, rtol=0.1)


def test_coord_cov_from_guassian_ls_scipy():
    """Test the coord_cov_from_guassian_ls_scipy function."""
    # Create some example input data
    rotations = np.array([0.0, 0, 1])
    sigmas = np.array([[5, 5], [5, 6], [4, 3]])

    heatmap_generator = GaussianHeatmapGenerator(
        nb_landmarks=3,
        sigmas=sigmas,
        rotation=rotations,
        heatmap_size=(64, 64),
        learnable=False,
        gamma=1,
    )
    landmarks = torch.tensor(
        [[[64 // 2, 64 // 2], [64 // 2, 64 // 2], [64 // 2, 64 // 2]]], dtype=torch.float
    )
    heatmaps = heatmap_generator(landmarks)
    covs_true = heatmap_generator.get_covariance_matrix().unsqueeze(0)

    # Call the method being tested
    coords, covs = coord_soft_argmax_cov(torch.logit(heatmaps))
    coords = coords.float()

    # Check the output
    assert coords.shape == (1, 3, 2)
    assert covs.shape == (1, 3, 2, 2)
    assert torch.allclose(coords, landmarks, atol=0.5)
    assert torch.allclose(covs, covs_true, atol=0.5)

    # Create some example input data
    rotations = np.array([0, 0, 0])
    sigmas = np.array([[5, 5], [1, 1], [2, 1]])
    heatmaps, landmarks, covs_true = create_heatmap(
        subpixel=False, sigmas=sigmas, rotations=rotations, return_covs=True
    )
    covs_true = covs_true.unsqueeze(0)
    # Call the method being tested
    coords, covs = coord_cov_from_gaussian_ls(heatmaps, gamma=None, ls_library="scipy")
    coords = coords.float()

    # Check the output
    assert coords.shape == (1, 3, 2)
    assert covs.shape == (1, 3, 2, 2)
    assert torch.allclose(covs, covs_true, atol=0.5)

    # Create some batched example input data
    heatmaps, landmarks = create_batch_of_heatmaps(  # pylint: disable=unbalanced-tuple-unpacking
        subpixel=False
    )

    # Call the method being tested
    coords, covs = coord_cov_from_gaussian_ls(heatmaps, gamma=None, ls_library="scipy")
    coords = coords.float()

    # Check the output
    assert coords.shape == (4, 3, 2)
    assert covs.shape == (4, 3, 2, 2)
    assert torch.allclose(coords, landmarks, atol=0.5)

    # Create some batched example input data with subpixel landmarks
    heatmaps, landmarks, covs_true = create_batch_of_heatmaps(
        subpixel=True, sigmas=sigmas, rotations=rotations, return_covs=True
    )
    coords = coords.float()

    # Call the method being tested
    coords, covs = coord_cov_from_gaussian_ls(heatmaps, gamma=None, ls_library="scipy")
    coords = coords.float()

    # Check the output
    assert coords.shape == (4, 3, 2)
    assert covs.shape == (4, 3, 2, 2)
    assert torch.allclose(coords.round(), landmarks.round(), atol=0.5)
    assert torch.allclose(covs, covs_true, atol=0.5)


def test_coord_cov_from_soft_argmax():
    """Test the coord_cov_from_soft_argmax function."""
    # TODO: issue wtih smaller sigmas, however this just probelly inherent to the method
    # Create some example input data
    rotations = np.array([0.0, 0, 1])
    sigmas = np.array([[5, 5], [5, 6], [4, 3]])

    heatmap_generator = GaussianHeatmapGenerator(
        nb_landmarks=3,
        sigmas=sigmas,
        rotation=rotations,
        heatmap_size=(64, 64),
        learnable=False,
        gamma=1,
    )
    landmarks = torch.tensor(
        [[[64 // 2, 64 // 2], [64 // 2, 64 // 2], [64 // 2, 64 // 2]]], dtype=torch.float
    )
    heatmaps = heatmap_generator(landmarks)
    covs_true = heatmap_generator.get_covariance_matrix().unsqueeze(0)

    # Call the method being tested
    coords, covs = coord_soft_argmax_cov(torch.logit(heatmaps))
    coords = coords.float()

    # Check the output
    assert coords.shape == (1, 3, 2)
    assert covs.shape == (1, 3, 2, 2)
    assert torch.allclose(coords, landmarks, atol=0.5)
    assert torch.allclose(covs, covs_true, atol=0.5)

    # Call the method being tested
    coords, covs = coord_soft_argmax_cov(heatmaps, logit_scale=False)
    coords = coords.float()

    # Check the output
    assert coords.shape == (1, 3, 2)
    assert covs.shape == (1, 3, 2, 2)
    assert torch.allclose(coords, landmarks, atol=0.5)
    assert torch.allclose(covs, covs_true, atol=0.5, rtol=0.1)


def test_coord_cov_from_weighted_spatial_mean():
    """Test the coord_cov_from_weighted_spatial_mean function."""
    # Create some example input data
    rotations = np.array([0, 0, 0])
    sigmas = np.array([[1, 1], [1, 1], [1, 1]])

    h, w = 512, 512

    heatmap_generator = GaussianHeatmapGenerator(
        nb_landmarks=3, sigmas=sigmas, rotation=rotations, heatmap_size=(h, w), learnable=False
    )
    landmarks = torch.tensor(
        [[[h // 2, w // 2], [h // 2, w // 2], [h // 2, w // 2]]], dtype=torch.float
    )
    heatmaps = heatmap_generator(landmarks)

    covs_true = heatmap_generator.get_covariance_matrix().unsqueeze(0)

    # Call the method being tested
    coords, covs = heatmap_to_coord_cov(heatmaps, method="weighted_spatial_mean")
    coords = coords.float()

    # Check the output
    assert coords.shape == (1, 3, 2)
    assert covs.shape == (1, 3, 2, 2)
    assert torch.allclose(coords, landmarks, atol=0.5)
    assert torch.allclose(covs, covs_true, atol=0.5)

    # Create some example input data
    rotations = np.array([0, 0, 0])
    sigmas = np.array([[5, 5], [1, 1], [2, 1]])
    heatmaps, landmarks, covs_true = create_heatmap(  # pylint: disable=unbalanced-tuple-unpacking
        subpixel=False, sigmas=sigmas, rotations=rotations, return_covs=True, gamma=None
    )
    covs_true = covs_true.unsqueeze(0)
    # Call the method being tested
    coords, covs = heatmap_to_coord_cov(heatmaps, method="weighted_spatial_mean")
    coords = coords.float()

    # Check the output
    assert coords.shape == (1, 3, 2)
    assert covs.shape == (1, 3, 2, 2)
    assert torch.allclose(covs, covs_true, atol=2)

    # Create some batched example input data
    heatmaps, landmarks = create_batch_of_heatmaps(  # pylint: disable=unbalanced-tuple-unpacking
        subpixel=False
    )

    # Call the method being tested
    coords, covs = heatmap_to_coord_cov(heatmaps, method="weighted_spatial_mean")
    coords = coords.float()

    # Check the output
    assert coords.shape == (4, 3, 2)
    assert covs.shape == (4, 3, 2, 2)
    assert torch.allclose(coords, landmarks, atol=0.5)

    # Create some batched example input data with subpixel landmarks
    heatmaps, landmarks, covs_true = create_batch_of_heatmaps(
        subpixel=True, sigmas=sigmas, rotations=rotations, return_covs=True
    )
    coords = coords.float()

    # Call the method being tested
    coords, covs = heatmap_to_coord_cov(heatmaps, method="weighted_spatial_mean")
    coords = coords.float()

    # Check the output
    assert coords.shape == (4, 3, 2)
    assert covs.shape == (4, 3, 2, 2)
    # TODO seems to be unstable
    assert torch.allclose(coords, landmarks, atol=0.5)
    assert torch.allclose(covs, covs_true, atol=0.5, rtol=0.1)


def test_coord_cov_windowed_weighted_sample():
    """Test the coord_cov_windowed_weighted_sample function."""
    # Create some example input data
    rotations = np.array([0, 0, 0])
    sigmas = np.array([[3, 3], [1, 1], [2, 1]])

    h, w = 512, 512
    heatmap_generator = GaussianHeatmapGenerator(
        nb_landmarks=3, sigmas=sigmas, rotation=rotations, heatmap_size=(h, w), learnable=False
    )
    landmarks = torch.tensor(
        [[[h // 2, w // 2], [h // 2, w // 2], [h // 2, w // 2]]], dtype=torch.float
    )
    heatmaps = heatmap_generator(landmarks)

    covs_true = heatmap_generator.get_covariance_matrix().unsqueeze(0)

    # Call the method being tested
    coords, covs = coord_cov_windowed_weigthed_sample_cov(heatmaps)
    coords = coords.float()

    # Check the output
    assert coords.shape == (1, 3, 2)
    assert covs.shape == (1, 3, 2, 2)
    assert torch.allclose(coords, landmarks, atol=0.5)
    assert torch.allclose(covs, covs_true, atol=0.5, rtol=0.1)

    heatmaps, landmarks, covs_true = create_heatmap(
        subpixel=False, sigmas=sigmas, rotations=rotations, return_covs=True, gamma=None
    )
    covs_true = covs_true.unsqueeze(0)
    # Call the method being tested
    coords, covs = coord_cov_windowed_weigthed_sample_cov(heatmaps)
    coords = coords.float()

    # Check the output
    assert coords.shape == (1, 3, 2)
    assert covs.shape == (1, 3, 2, 2)
    assert torch.allclose(coords, landmarks, atol=0.5)
    assert torch.allclose(covs, covs_true, atol=0.5, rtol=0.1)


def test_cov_ls_scipy():
    """Test cov_ls_scipy."""
    # Create some example input data
    rotations = np.array([0.0, 0, 1])
    sigmas = np.array([[5, 5], [1, 1], [2, 1]])

    gamma = None
    heatmap_generator = GaussianHeatmapGenerator(
        nb_landmarks=3,
        sigmas=sigmas,
        rotation=rotations,
        heatmap_size=(64, 64),
        learnable=False,
        gamma=gamma,
    )
    landmarks = torch.tensor(
        [[[64 // 2 - 5, 64 // 2], [64 // 2, 64 // 2 + 5], [64 // 2, 64 // 2]]], dtype=torch.float
    )
    heatmaps = heatmap_generator(landmarks)
    covs_true = heatmap_generator.get_covariance_matrix().unsqueeze(0)

    # Call the method being tested
    covs = cov_from_gaussian_ls(heatmaps, landmarks, gamma=gamma, ls_library="scipy")

    # Check the output
    assert covs.shape == (1, 3, 2, 2)
    assert torch.allclose(covs, covs_true, atol=0.5)

    gamma = 1
    heatmap_generator = GaussianHeatmapGenerator(
        nb_landmarks=3,
        sigmas=sigmas,
        rotation=rotations,
        heatmap_size=(64, 64),
        learnable=False,
        gamma=gamma,
    )
    landmarks = torch.tensor(
        [[[64 // 2 - 5, 64 // 2], [64 // 2, 64 // 2 + 5], [64 // 2, 64 // 2]]], dtype=torch.float
    )
    heatmaps = heatmap_generator(landmarks)
    covs_true = heatmap_generator.get_covariance_matrix().unsqueeze(0)

    # Call the method being tested
    covs = cov_from_gaussian_ls(heatmaps, landmarks, gamma=gamma)

    # Check the output
    assert covs.shape == (1, 3, 2, 2)
    assert torch.allclose(covs, covs_true, atol=0.5)


def test_cov_weigthed_sample():
    """Test cov_weigthed_sample."""
    # Create some example input data
    rotations = np.array([0.0, 0, 1])
    sigmas = np.array([[5, 5], [1, 1], [2, 1]])

    gamma = 1
    heatmap_generator = GaussianHeatmapGenerator(
        nb_landmarks=3,
        sigmas=sigmas,
        rotation=rotations,
        heatmap_size=(64, 64),
        learnable=False,
        gamma=gamma,
    )
    landmarks = torch.tensor(
        [[[64 // 2 - 5, 64 // 2], [64 // 2, 64 // 2 + 5], [64 // 2, 64 // 2]]], dtype=torch.float
    )
    heatmaps = heatmap_generator(landmarks)
    covs_true = heatmap_generator.get_covariance_matrix().unsqueeze(0)

    # Call the method being tested
    covs = heatmap_coord_to_weighted_sample_cov(heatmaps, landmarks, apply_softmax=False)

    # Check the output
    assert covs.shape == (1, 3, 2, 2)
    assert torch.allclose(covs, covs_true, atol=0.5)

    # Call the method being tested
    covs = heatmap_coord_to_weighted_sample_cov(
        torch.logit(heatmaps), landmarks, apply_softmax=True
    )
    # Check the output
    assert covs.shape == (1, 3, 2, 2)
    assert torch.allclose(covs, covs_true, atol=0.5)


def test_heatmap_to_coord():
    """Test the heatmap to coord function."""
    heatmap, _ = create_batch_of_heatmaps(  # pylint: disable=unbalanced-tuple-unpacking
        subpixel=False
    )

    coord_1 = coord_argmax(heatmap)
    coord_2 = heatmap_to_coord(heatmap, method="argmax")
    assert torch.allclose(coord_1, coord_2)

    coord_1 = coord_local_soft_argmax(heatmap)
    coord_2 = heatmap_to_coord(heatmap, method="local_soft_argmax")
    assert torch.allclose(coord_1, coord_2)

    coord_1 = coord_weighted_spatial_mean(heatmap)
    coord_2 = heatmap_to_coord(heatmap, method="weighted_spatial_mean")
    assert torch.allclose(coord_1, coord_2)

    coord_1 = coord_soft_argmax(heatmap)
    coord_2 = heatmap_to_coord(heatmap, method="soft_argmax")
    assert torch.allclose(coord_1, coord_2)

    coord_1 = coord_soft_argmax(heatmap, logit_scale=True)
    coord_2 = heatmap_to_coord(heatmap, method="soft_argmax_logit")

    try:
        _ = heatmap_to_coord(heatmap, method="wrong_method")
        assert False
    except ValueError:
        assert True


def test_non_maxima_surpression():
    """Test the non-maxima surpression function."""
    # create some dummy input tensors
    heatmap = torch.tensor(
        [[9, 10, 9, 8, 7], [8, 9, 8, 7, 6], [7, 8, 7, 6, 5], [6, 7, 6, 5, 4], [5, 6, 5, 4, 3]]
    )

    # call the non-maxima surpression function with the input tensor
    local_maximums = non_maximum_surpression(heatmap, window=3)

    # check that the output has the correct length
    assert len(local_maximums) == 1

    # check that the output contains the correct values
    assert any(
        [
            torch.all(torch.Tensor((0, 1)).eq(local_maximum)).item()
            for local_maximum in local_maximums
        ]
    )

    heatmap = torch.tensor(
        [[9, 10, 9, 8, 7], [8, 9, 8, 7, 6], [7, 8, 7, 6, 5], [6, 7, 6, 12, 4], [5, 6, 5, 4, 3]]
    )

    # call the non-maxima surpression function with the input tensor
    local_maximums = non_maximum_surpression(heatmap, window=2)

    # check that the output has the correct length
    assert len(local_maximums) == 2

    # check that the output contains the correct values
    assert any(
        [
            torch.all(torch.Tensor((0, 1)).eq(local_maximum)).item()
            for local_maximum in local_maximums
        ]
    )
    assert any(
        [
            torch.all(torch.Tensor((3, 3)).eq(local_maximum)).item()
            for local_maximum in local_maximums
        ]
    )

    # create some dummy input tensors
    heatmap = torch.tensor(
        [[9, 10, 9, 8, 7], [8, 9, 8, 7, 6], [7, 8, 7, 6, 5], [6, 7, 6, 12, 4], [5, 6, 5, 4, 3]]
    )

    # call the non-maxima surpression function with the input tensor
    local_maximums = non_maximum_surpression(heatmap, window=3)

    # check that the output has the correct length
    assert len(local_maximums) == 1

    # check that the output contains the correct values
    assert any(
        [
            torch.all(torch.Tensor((3, 3)).eq(local_maximum)).item()
            for local_maximum in local_maximums
        ]
    )


def test_non_maxima_surpression_local_soft_argmax():
    """Test the non-maxima surpression function."""
    # create some dummy input tensors
    heatmap = torch.tensor(
        [[9, 10, 9, 8, 7], [8, 9, 8, 7, 6], [7, 8, 7, 6, 5], [6, 7, 6, 5, 4], [5, 6, 5, 4, 3]]
    )

    # call the non-maxima surpression function with the input tensor
    local_maximums = non_maximum_surpression_local_soft_argmax(heatmap, window=3)

    # check that the output has the correct length
    assert len(local_maximums) == 1

    heatmap = torch.tensor(
        [[9, 10, 9, 8, 7], [8, 9, 8, 7, 6], [7, 8, 7, 6, 5], [6, 7, 6, 12, 4], [5, 6, 5, 4, 3]]
    )

    # call the non-maxima surpression function with the input tensor
    local_maximums = non_maximum_surpression_local_soft_argmax(heatmap, window=2)

    # check that the output has the correct length
    assert len(local_maximums) == 2

    # create some dummy input tensors
    heatmap = torch.tensor(
        [[9, 10, 9, 8, 7], [8, 9, 8, 7, 6], [7, 8, 7, 6, 5], [6, 7, 6, 12, 4], [5, 6, 5, 4, 3]]
    )

    # call the non-maxima surpression function with the input tensor
    local_maximums = non_maximum_surpression_local_soft_argmax(heatmap, window=3)

    # check that the output has the correct length
    assert len(local_maximums) == 1


def test_heatmap_to_multiple_coord():
    """Test the heatmap to coord function."""
    heatmap, landmark = create_batch_of_heatmaps(  # pylint: disable=unbalanced-tuple-unpacking
        subpixel=False
    )
    heatmap = heatmap.sum(dim=0).unsqueeze(0)
    assert heatmap.shape == (1, 3, 64, 64)

    pred_coords, _ = heatmap_to_multiple_coord(heatmap, window=5, threshold=0.5, method="argmax")
    assert len(pred_coords) == 1
    for i in range(1):
        assert len(pred_coords[i]) == 3
        for j in range(3):
            assert len(pred_coords[i][j]) == 4
            for k in range(4):
                assert any([torch.allclose(pred_coords[i][j][k], landmark[b, j]) for b in range(4)])


def test_heatmap_to_coord_enlarge():
    """Test the heatmap to coord function."""
    check_retriever_output(
        heatmap_to_coord_enlarge, atol=1, rtol=0.1, method="argmax", enlarge_factor=2
    )

    check_retriever_output(
        heatmap_to_coord_enlarge, atol=1, rtol=0.1, method="argmax", enlarge_dim=(128, 128)
    )
