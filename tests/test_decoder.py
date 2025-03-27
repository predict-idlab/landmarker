"""Tests for the decoder module."""

import numpy as np
import torch

from src.landmarker.heatmap.decoder import (
    coord_argmax,
    coord_cov_from_gaussian_ls,
    coord_local_soft_argmax,
    coord_weighted_spatial_mean,
    cov_from_gaussian_ls,
    heatmap_to_coord,
    heatmap_to_coord_cov,
    heatmap_to_coord_enlarge,
    heatmap_to_multiple_coord,
    non_maximum_surpression,
    weighted_sample_cov,
    windowed_weigthed_sample_cov,
)
from src.landmarker.heatmap.generator import GaussianHeatmapGenerator


def create_heatmap(
    subpixel=False, sigmas=1.0, rotations=0.0, return_covs=False, gamma=None, square=True
):
    """Create a heatmap and corresponding landmarks for testing."""
    # Create a generator with some example parameters
    if square:
        generator = GaussianHeatmapGenerator(
            nb_landmarks=3,
            sigmas=sigmas,
            rotation=rotations,
            heatmap_size=(64, 64),
            learnable=False,
            gamma=gamma,
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
    else:
        generator = GaussianHeatmapGenerator(
            nb_landmarks=3,
            sigmas=sigmas,
            rotation=rotations,
            heatmap_size=(64, 32),
            learnable=False,
            gamma=gamma,
        )

        # Create some example input data
        landmarks = torch.tensor([[[20, 20], [30, 15], [45, 10]]], dtype=torch.float)

        if subpixel:
            landmarks += torch.rand(landmarks.shape, dtype=torch.float32)

        # Call the method being tested
        heatmaps = generator(landmarks)

        if return_covs:
            return heatmaps, landmarks, generator.get_covariance_matrix()
        return heatmaps, landmarks


def create_3d_heatmap(
    subpixel=False, sigmas=1.0, rotations=0.0, return_covs=False, gamma=None, square=True
):
    """Create a heatmap and corresponding landmarks for testing."""
    # Create a generator with some example parameters
    if square:
        generator = GaussianHeatmapGenerator(
            nb_landmarks=3,
            sigmas=sigmas,
            rotation=rotations,
            heatmap_size=(64, 64, 64),
            learnable=False,
            gamma=gamma,
        )

        # Create some example input data
        landmarks = torch.tensor([[[20, 20, 20], [30, 40, 30], [45, 25, 40]]], dtype=torch.float)

        if subpixel:
            landmarks += torch.rand(landmarks.shape, dtype=torch.float32)

        # Call the method being tested
        heatmaps = generator(landmarks)

        if return_covs:
            return heatmaps, landmarks, generator.get_covariance_matrix()
        return heatmaps, landmarks
    else:
        generator = GaussianHeatmapGenerator(
            nb_landmarks=3,
            sigmas=sigmas,
            rotation=rotations,
            heatmap_size=(64, 32, 40),
            learnable=False,
            gamma=gamma,
        )

        # Create some example input data
        landmarks = torch.tensor([[[20, 20, 20], [30, 15, 10], [45, 10, 15]]], dtype=torch.float)

        if subpixel:
            landmarks += torch.rand(landmarks.shape, dtype=torch.float32)

        # Call the method being tested
        heatmaps = generator(landmarks)

        if return_covs:
            return heatmaps, landmarks, generator.get_covariance_matrix()
        return heatmaps, landmarks


def create_batch_of_heatmaps(
    subpixel=False, sigmas=1.0, rotations=0.0, return_covs=False, gamma=None, square=True
):
    """Create a batch of heatmaps and corresponding landmarks for testing."""
    # Create a generator with some example parameters
    if square:
        generator = GaussianHeatmapGenerator(
            nb_landmarks=3,
            sigmas=sigmas,
            rotation=rotations,
            heatmap_size=(64, 64),
            learnable=False,
            gamma=gamma,
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
    else:
        generator = GaussianHeatmapGenerator(
            nb_landmarks=3,
            sigmas=sigmas,
            rotation=rotations,
            heatmap_size=(64, 32),
            learnable=False,
            gamma=gamma,
        )

        landmarks_batch = torch.tensor(
            [
                [[10, 10], [30, 15], [50, 20]],
                [[16, 17], [24, 10], [30, 13]],
                [[24, 24], [14, 14], [54, 21]],
                [[30, 20], [36, 25], [20, 24]],
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


def create_batch_of_3d_heatmaps(
    subpixel=False, sigmas=1.0, rotations=0.0, return_covs=False, gamma=None, square=True
):
    """Create a batch of heatmaps and corresponding landmarks for testing."""
    # Create a generator with some example parameters
    if square:
        generator = GaussianHeatmapGenerator(
            nb_landmarks=3,
            sigmas=sigmas,
            rotation=rotations,
            heatmap_size=(64, 64, 64),
            learnable=False,
            gamma=gamma,
        )

        landmarks_batch = torch.tensor(
            [
                [[10, 10, 10], [30, 40, 30], [50, 60, 50]],
                [[16, 17, 16], [24, 35, 24], [30, 58, 30]],
                [[24, 24, 24], [14, 14, 14], [54, 30, 54]],
                [[30, 30, 30], [36, 46, 36], [20, 10, 20]],
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
    else:
        generator = GaussianHeatmapGenerator(
            nb_landmarks=3,
            sigmas=sigmas,
            rotation=rotations,
            heatmap_size=(64, 32, 40),
            learnable=False,
            gamma=gamma,
        )

        landmarks_batch = torch.tensor(
            [
                [[10, 10, 10], [30, 15, 20], [50, 20, 30]],
                [[16, 17, 16], [24, 10, 15], [30, 13, 10]],
                [[24, 24, 24], [14, 14, 14], [54, 21, 24]],
                [[30, 20, 30], [36, 25, 20], [20, 24, 10]],
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


def check_retriever_output(retriever_fun, atol=1e-2, rtol=0, spatial_dims=2, **kwargs):
    """Check the output of a retriever function."""
    # Create some example input data
    for square in [True, False]:
        if spatial_dims == 2:
            heatmaps, landmarks = create_heatmap(  # pylint: disable=unbalanced-tuple-unpacking
                subpixel=False, return_covs=False, square=square
            )
        else:
            heatmaps, landmarks = create_3d_heatmap(  # pylint: disable=unbalanced-tuple-unpacking
                subpixel=False, return_covs=False, square=square
            )

        # Call the method being tested
        coords = retriever_fun(heatmaps, spatial_dims=spatial_dims, **kwargs).float()

        # Check the output
        assert coords.shape == (1, 3, spatial_dims)
        assert torch.allclose(coords, landmarks, atol=atol, rtol=rtol)

        # Create some batched example input data
        if spatial_dims == 2:
            heatmaps, landmarks = (
                create_batch_of_heatmaps(  # pylint: disable=unbalanced-tuple-unpacking
                    subpixel=False, return_covs=False, square=square
                )
            )
        else:
            heatmaps, landmarks = (
                create_batch_of_3d_heatmaps(  # pylint: disable=unbalanced-tuple-unpacking
                    subpixel=False, return_covs=False, square=square
                )
            )
        # Call the method being tested
        coords = retriever_fun(heatmaps, spatial_dims=spatial_dims, **kwargs).float()

        # Check the output
        assert coords.shape == (4, 3, spatial_dims)
        assert torch.allclose(coords, landmarks, atol=atol, rtol=rtol)

        # Create some batched example input data with subpixel landmarks
        if spatial_dims == 2:
            heatmaps, landmarks = create_batch_of_heatmaps(
                subpixel=True, square=square
            )  # pylint: disable=unbalanced-tuple-unpacking
        else:
            heatmaps, landmarks = create_batch_of_3d_heatmaps(
                subpixel=True, square=square
            )  # pylint: disable=unbalanced-tuple-unpacking

        # Call the method being tested
        coords = retriever_fun(heatmaps, spatial_dims=spatial_dims, **kwargs).float()

        # Check the output
        assert coords.shape == (4, 3, spatial_dims)
        assert torch.allclose(coords.round(), landmarks.round(), atol=atol)


def test_coord_argmax():
    """Test the coord_argmax function."""
    check_retriever_output(coord_argmax, atol=1e-2, rtol=0)
    check_retriever_output(coord_argmax, atol=1e-2, rtol=0, spatial_dims=3)
    try:
        check_retriever_output(coord_argmax, atol=1e-2, rtol=0, spatial_dims=4)
        assert False
    except ValueError:
        assert True


def test_coord_local_soft_argmax():
    """Test the coord_local_soft_argmax function."""
    check_retriever_output(coord_local_soft_argmax, atol=1e-2, window=3, t=10)

    check_retriever_output(coord_local_soft_argmax, atol=1e-2, window=5, t=5)

    check_retriever_output(coord_local_soft_argmax, atol=1e-2, window=7, t=3)
    check_retriever_output(coord_local_soft_argmax, atol=1e-2, window=3, t=10, spatial_dims=3)

    check_retriever_output(coord_local_soft_argmax, atol=1e-2, window=5, t=5, spatial_dims=3)

    check_retriever_output(coord_local_soft_argmax, atol=1e-2, window=7, t=3, spatial_dims=3)
    try:
        check_retriever_output(coord_local_soft_argmax, atol=1e-2, window=7, t=3, spatial_dims=4)
        assert False
    except ValueError:
        assert True


def test_coord_weighted_spatial_mean():
    """Test the coord_weighted_spatial_mean function."""
    # TODO: the method seems to be kind of unstable hence the high tolerance
    check_retriever_output(coord_weighted_spatial_mean, atol=3)
    check_retriever_output(coord_weighted_spatial_mean, atol=3, spatial_dims=3)
    try:
        check_retriever_output(coord_weighted_spatial_mean, atol=3, spatial_dims=4)
        assert False
    except ValueError:
        assert True
    try:
        retriever_func = (
            lambda x, **kwargs: coord_weighted_spatial_mean(  # pylint: disable=unnecessary-lambda
                x, **kwargs, activation="wrong"
            )
        )
        check_retriever_output(retriever_func, atol=3)
        assert False
    except ValueError:
        assert True


def test_coord_cov_from_guassian_ls_scipy():
    """Test the coord_cov_from_guassian_ls_scipy function."""
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

    try:
        _ = coord_cov_from_gaussian_ls(heatmaps, gamma=None, ls_library="scipy", spatial_dims=3)
        assert False
    except ValueError:
        assert True

    try:
        _ = coord_cov_from_gaussian_ls(
            heatmaps, gamma=None, ls_library="Not implemented", spatial_dims=2
        )
        assert False
    except ValueError:
        assert True


def test_coord_cov_from_weighted_spatial_mean():
    """Test the coord_cov_from_weighted_spatial_mean function."""
    # Create some example input data
    rotations = np.array([0, 0, 0])
    sigmas = np.array([[1, 1], [1, 1], [1, 1]])

    h, w = 128, 128

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
    assert torch.allclose(coords, landmarks, atol=0.5)
    assert torch.allclose(covs, covs_true, atol=0.5, rtol=0.1)


def test_coord_cov_from_weighted_spatial_mean_3d():
    """Test the coord_cov_from_weighted_spatial_mean function."""
    # Create some example input data
    rotations = np.array([0, 0, 0])
    sigmas = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    h, w, d = 128, 128, 128

    heatmap_generator = GaussianHeatmapGenerator(
        nb_landmarks=3, sigmas=sigmas, rotation=rotations, heatmap_size=(h, w, d), learnable=False
    )
    landmarks = torch.tensor(
        [[[h // 2, w // 2, d // 2], [h // 2, w // 2, d // 2], [h // 2, w // 2, d // 2]]],
        dtype=torch.float,
    )
    heatmaps = heatmap_generator(landmarks)

    covs_true = heatmap_generator.get_covariance_matrix().unsqueeze(0)

    # Call the method being tested
    coords, covs = heatmap_to_coord_cov(heatmaps, method="weighted_spatial_mean", spatial_dims=3)
    coords = coords.float()

    # Check the output
    assert coords.shape == (1, 3, 3)
    assert covs.shape == (1, 3, 3, 3)
    assert torch.allclose(coords, landmarks, atol=0.5)
    assert torch.allclose(covs, covs_true, atol=0.5)

    # Create some example input data
    rotations = np.array([0, 0, 0])
    sigmas = np.array([[5, 5, 3], [1, 1, 2], [2, 1, 3]])
    heatmaps, landmarks, covs_true = (
        create_3d_heatmap(  # pylint: disable=unbalanced-tuple-unpacking
            subpixel=False, sigmas=sigmas, rotations=rotations, return_covs=True, gamma=None
        )
    )
    covs_true = covs_true.unsqueeze(0)
    # Call the method being tested
    coords, covs = heatmap_to_coord_cov(heatmaps, method="weighted_spatial_mean", spatial_dims=3)
    coords = coords.float()

    # Check the output
    assert coords.shape == (1, 3, 3)
    assert covs.shape == (1, 3, 3, 3)
    assert torch.allclose(covs, covs_true, atol=2)

    # Create some batched example input data
    heatmaps, landmarks = create_batch_of_3d_heatmaps(  # pylint: disable=unbalanced-tuple-unpacking
        subpixel=False
    )

    # Call the method being tested
    coords, covs = heatmap_to_coord_cov(heatmaps, method="weighted_spatial_mean", spatial_dims=3)
    coords = coords.float()

    # Check the output
    assert coords.shape == (4, 3, 3)
    assert covs.shape == (4, 3, 3, 3)
    assert torch.allclose(coords, landmarks, atol=0.5)

    try:
        _ = heatmap_to_coord_cov(heatmaps, method="weighted_spatial_mean", spatial_dims=4)
        assert False
    except ValueError:
        assert True


def test_windowed_weigthed_sample_cov():
    """Test the windowed_weigthed_sample_cov function."""
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
    covs = windowed_weigthed_sample_cov(heatmaps, landmarks)

    # Check the output
    assert covs.shape == (1, 3, 2, 2)
    assert torch.allclose(covs, covs_true, atol=0.5, rtol=0.1)

    heatmaps, landmarks, covs_true = create_heatmap(
        subpixel=False, sigmas=sigmas, rotations=rotations, return_covs=True, gamma=None
    )
    covs_true = covs_true.unsqueeze(0)
    # Call the method being tested
    covs = windowed_weigthed_sample_cov(heatmaps, landmarks)

    # Check the output
    assert covs.shape == (1, 3, 2, 2)
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


def test_weighted_sample_cov():
    """Test weighted_sample_cov."""
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
    covs = weighted_sample_cov(heatmaps, landmarks, spatial_dims=2)

    # Check the output
    assert covs.shape == (1, 3, 2, 2)
    assert torch.allclose(covs, covs_true, atol=0.5)

    # Call the method being tested
    covs = weighted_sample_cov(torch.logit(heatmaps, eps=1e-38), landmarks, activation="sigmoid")
    # Check the output
    assert covs.shape == (1, 3, 2, 2)
    assert torch.allclose(covs, covs_true, atol=0.5)


def test_weighted_sample_cov_3d():
    """Test weighted_sample_cov for 3D."""
    # Create some example input data
    rotations = np.array([0.0, 0, 1])
    sigmas = np.array([[4, 4, 5], [1, 1, 1], [2, 1, 1]])

    gamma = 1
    heatmap_generator = GaussianHeatmapGenerator(
        nb_landmarks=3,
        sigmas=sigmas,
        rotation=rotations,
        heatmap_size=(64, 64, 64),
        learnable=False,
        gamma=gamma,
    )
    landmarks = torch.tensor(
        [
            [
                [64 // 2 - 5, 64 // 2, 64 // 2],
                [64 // 2, 64 // 2 + 5, 64 // 2],
                [64 // 2, 64 // 2, 64 // 2 - 3],
            ]
        ],
        dtype=torch.float,
    )
    heatmaps = heatmap_generator(landmarks)
    covs_true = heatmap_generator.get_covariance_matrix().unsqueeze(0)

    # Call the method being tested
    covs = weighted_sample_cov(heatmaps, landmarks, spatial_dims=3)

    # Check the output
    assert covs.shape == (1, 3, 3, 3)
    assert torch.allclose(covs, covs_true, atol=0.5)

    # Call the method being tested
    covs = weighted_sample_cov(
        torch.logit(heatmaps, eps=1e-38), landmarks, activation="sigmoid", spatial_dims=3
    )
    # Check the output
    assert covs.shape == (1, 3, 3, 3)
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

    coord_1 = coord_weighted_spatial_mean(heatmap, activation="softmax")
    coord_2 = heatmap_to_coord(heatmap, method="soft_argmax")
    assert torch.allclose(coord_1, coord_2)

    coord_1 = coord_weighted_spatial_mean(heatmap, activation="ReLU")
    coord_2 = heatmap_to_coord(heatmap, method="weighted_spatial_mean_relu")

    coord_1 = coord_weighted_spatial_mean(heatmap, activation="sigmoid")
    coord_2 = heatmap_to_coord(heatmap, method="weighted_spatial_mean_sigmoid")

    try:
        _ = heatmap_to_coord(heatmap, method="wrong_method")
        assert False
    except ValueError:
        assert True


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
    check_retriever_output(
        heatmap_to_coord_enlarge,
        atol=1,
        rtol=0.1,
        method="argmax",
        enlarge_factor=2,
        spatial_dims=3,
        enlarge_mode="trilinear",
    )
    check_retriever_output(
        heatmap_to_coord_enlarge,
        atol=1,
        rtol=0.1,
        method="argmax",
        enlarge_dim=(128, 128, 128),
        spatial_dims=3,
        enlarge_mode="trilinear",
    )
    try:
        check_retriever_output(
            heatmap_to_coord_enlarge,
            atol=1,
            rtol=0.1,
            method="argmax",
            enlarge_dim=(128, 128),
            spatial_dims=5,
        )
        assert False
    except ValueError:
        assert True
