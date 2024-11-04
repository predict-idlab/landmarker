"""Test for the loss module."""

import torch
import torch.nn as nn

from src.landmarker.heatmap.generator import GaussianHeatmapGenerator
from src.landmarker.losses.losses import (
    EuclideanDistanceVarianceReg,
    GeneralizedNormalHeatmapLoss,
    MultivariateGaussianNLLLoss,
    NLLLoss,
    StackedLoss,
    StarLoss,
)


def test_generalized_normal_heatmap_loss():
    """Test the GeneralizedNormalHeatmapLoss class."""
    # create an instance of the loss function

    for distance in ["l1", "l2", "smooth-l1", "bce-with-logits", "bce"]:
        if "l1" == distance:
            reduction = "mean"
        else:
            reduction = "sum"
        loss_fn = GeneralizedNormalHeatmapLoss(alpha=5, distance=distance, reduction=reduction)

        # create some dummy input tensors
        if distance in "bce":
            pred = torch.rand(2, 3, 64, 64)
            target = torch.rand(2, 3, 64, 64)
        elif distance in ["bce-with-logits"]:
            pred = torch.randn(2, 3, 64, 64)
            target = torch.rand(2, 3, 64, 64)
        else:
            pred = torch.rand(2, 3, 64, 64) * 64
            target = torch.rand(2, 3, 64, 64) * 64
        sigmas = torch.rand(3, 2) * 5
        rotation = torch.rand(3) * 2 * torch.pi
        heatmap_generator = GaussianHeatmapGenerator(
            3, sigmas=sigmas, heatmap_size=(64, 64), rotation=rotation
        )
        covs = heatmap_generator.get_covariance_matrix()

        # calculate the loss
        loss = loss_fn(pred, covs, target)

        # check that the output has the correct shape
        assert loss.shape == torch.Size([])

        # check that the output is a scalar tensor
        assert loss.dim() == 0

        # check that the output is non-negative
        assert loss >= 0
    try:
        loss_fn = GeneralizedNormalHeatmapLoss(alpha=5, distance="invalid", reduction="mean")
        assert False
    except ValueError:
        assert True


def test_generalized_normal_heatmap_loss_3d():
    """Test the GeneralizedNormalHeatmapLoss class. For 3D inputs."""
    # create an instance of the loss function

    for distance in ["l1", "l2", "smooth-l1", "bce-with-logits", "bce"]:
        if "l1" == distance:
            reduction = "mean"
        else:
            reduction = "sum"
        loss_fn = GeneralizedNormalHeatmapLoss(alpha=5, distance=distance, reduction=reduction)

        # create some dummy input tensors
        if distance in "bce":
            pred = torch.rand(2, 3, 64, 64, 64)
            target = torch.rand(2, 3, 64, 64, 64)
        elif distance in ["bce-with-logits"]:
            pred = torch.randn(2, 3, 64, 64, 64)
            target = torch.rand(2, 3, 64, 64, 64)
        else:
            pred = torch.rand(2, 3, 64, 64, 64) * 64
            target = torch.rand(2, 3, 64, 64, 64) * 64
        sigmas = torch.rand(3, 3) * 5
        rotation = torch.rand(3, 3) * 2 * torch.pi
        heatmap_generator = GaussianHeatmapGenerator(
            3, sigmas=sigmas, heatmap_size=(64, 64, 64), rotation=rotation
        )
        covs = heatmap_generator.get_covariance_matrix()

        # calculate the loss
        loss = loss_fn(pred, covs, target)

        # check that the output has the correct shape
        assert loss.shape == torch.Size([])

        # check that the output is a scalar tensor
        assert loss.dim() == 0

        # check that the output is non-negative
        assert loss >= 0
    try:
        loss_fn = GeneralizedNormalHeatmapLoss(alpha=5, distance="invalid", reduction="mean")
        assert False
    except ValueError:
        assert True


def test_multivariate_gaussian_nll_loss():
    """Test the MultivariateGaussianNLLLoss class."""
    reduction = "mean"
    pred = torch.rand(1, 3, 2) * 64
    target = torch.rand(1, 3, 2) * 64
    cov_matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).view((1, 1, 2, 2)).repeat(1, 3, 1, 1)

    loss_fn = MultivariateGaussianNLLLoss(reduction=reduction)
    expected_output_shape = torch.Size([])

    loss = loss_fn(pred, cov_matrix, target)

    assert loss.shape == expected_output_shape

    pred = torch.rand((4, 5, 2)) * 64
    target = torch.rand((4, 5, 2)) * 64
    cov_matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).view((1, 1, 2, 2)).repeat(4, 5, 1, 1)

    loss_fn = MultivariateGaussianNLLLoss(reduction=reduction)
    expected_output_shape = torch.Size([])

    loss = loss_fn(pred, cov_matrix, target)

    assert loss.shape == expected_output_shape

    reduction = "sum"

    loss_fn = MultivariateGaussianNLLLoss(reduction=reduction)
    expected_output_shape = torch.Size([])

    loss = loss_fn(pred, cov_matrix, target)

    assert loss.shape == expected_output_shape

    reduction = "none"

    loss_fn = MultivariateGaussianNLLLoss(reduction=reduction)
    expected_output_shape = torch.Size([4, 5])

    loss = loss_fn(pred, cov_matrix, target)

    assert loss.shape == expected_output_shape


def test_multivariate_gaussian_nll_loss_3d():
    """Test the MultivariateGaussianNLLLoss class. For 3D inputs."""
    reduction = "mean"
    pred = torch.rand(1, 3, 3) * 64
    target = torch.rand(1, 3, 3) * 64
    cov_matrix = (
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        .view((1, 1, 3, 3))
        .repeat(1, 3, 1, 1)
    )

    loss_fn = MultivariateGaussianNLLLoss(reduction=reduction, spatial_dims=3)
    expected_output_shape = torch.Size([])

    loss = loss_fn(pred, cov_matrix, target)

    assert loss.shape == expected_output_shape

    pred = torch.rand((4, 5, 3)) * 64
    target = torch.rand((4, 5, 3)) * 64
    cov_matrix = (
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        .view((1, 1, 3, 3))
        .repeat(4, 5, 1, 1)
    )

    loss_fn = MultivariateGaussianNLLLoss(reduction=reduction, spatial_dims=3)
    expected_output_shape = torch.Size([])

    loss = loss_fn(pred, cov_matrix, target)

    assert loss.shape == expected_output_shape

    reduction = "sum"

    loss_fn = MultivariateGaussianNLLLoss(reduction=reduction, spatial_dims=3)
    expected_output_shape = torch.Size([])

    loss = loss_fn(pred, cov_matrix, target)

    assert loss.shape == expected_output_shape

    reduction = "none"

    loss_fn = MultivariateGaussianNLLLoss(reduction=reduction, spatial_dims=3)
    expected_output_shape = torch.Size([4, 5])

    loss = loss_fn(pred, cov_matrix, target)

    assert loss.shape == expected_output_shape


def test_euclidean_distance_variance_reg():
    """Test the EuclideanDistanceVarianceReg class."""
    reduction = "mean"
    sigma_t = 3
    # pred = torch.rand(1, 3, 2) * 64
    # target = torch.rand(1, 3, 2) * 64
    pred = torch.ones((1, 3, 2)) * (64 // 2)
    target = torch.ones((1, 3, 2)) * (64 // 2 - 5)

    heatmap_generator = GaussianHeatmapGenerator(3, sigmas=sigma_t, heatmap_size=(64, 64))
    cov_t = heatmap_generator.get_covariance_matrix()

    loss_fn = EuclideanDistanceVarianceReg(reduction=reduction, var_t=sigma_t**2)
    expected_output_shape = torch.Size([])

    loss = loss_fn(target, cov_t, target)

    assert loss.shape == expected_output_shape
    assert loss == 0

    loss = loss_fn(pred, cov_t, target)
    assert loss > 0

    reduction = "sum"
    sigma_t = 3
    # pred = torch.rand(1, 3, 2) * 64
    # target = torch.rand(1, 3, 2) * 64
    pred = torch.ones((1, 3, 2)) * (64 // 2)
    target = torch.ones((1, 3, 2)) * (64 // 2 - 5)

    heatmap_generator = GaussianHeatmapGenerator(3, sigmas=sigma_t, heatmap_size=(64, 64))
    cov_t = heatmap_generator.get_covariance_matrix()

    loss_fn = EuclideanDistanceVarianceReg(reduction=reduction, var_t=sigma_t**2)
    expected_output_shape = torch.Size([])

    loss = loss_fn(target, cov_t, target)

    assert loss.shape == expected_output_shape
    assert loss == 0

    loss = loss_fn(pred, cov_t, target)
    assert loss > 0

    reduction = "none"
    sigma_t = 3
    # pred = torch.rand(1, 3, 2) * 64
    # target = torch.rand(1, 3, 2) * 64
    pred = torch.ones((1, 3, 2)) * (64 // 2)
    target = torch.ones((1, 3, 2)) * (64 // 2 - 5)

    heatmap_generator = GaussianHeatmapGenerator(3, sigmas=sigma_t, heatmap_size=(64, 64))
    cov_t = heatmap_generator.get_covariance_matrix()

    loss_fn = EuclideanDistanceVarianceReg(reduction=reduction, var_t=sigma_t**2)
    expected_output_shape = target.shape[:-1]

    loss = loss_fn(target, cov_t, target)

    assert loss.shape == expected_output_shape
    assert torch.allclose(loss, torch.zeros_like(loss))

    loss = loss_fn(pred, cov_t, target)
    assert torch.all(loss > 0)

    try:
        loss_fn = EuclideanDistanceVarianceReg(reduction="invalid", var_t=-1)
        loss = loss_fn(pred, cov_t, target)
        assert False
    except ValueError:
        assert True


def test_euclidean_distance_variance_reg_3d():
    """Test the EuclideanDistanceVarianceReg class. For 3D inputs."""
    reduction = "mean"
    sigma_t = 3

    pred = torch.ones((1, 3, 3)) * (64 // 2)
    target = torch.ones((1, 3, 3)) * (64 // 2 - 5)

    heatmap_generator = GaussianHeatmapGenerator(3, sigmas=sigma_t, heatmap_size=(64, 64, 64))
    cov_t = heatmap_generator.get_covariance_matrix()

    loss_fn = EuclideanDistanceVarianceReg(reduction=reduction, var_t=sigma_t**2, spatial_dims=3)
    expected_output_shape = torch.Size([])

    loss = loss_fn(target, cov_t, target)

    assert loss.shape == expected_output_shape
    assert loss == 0

    loss = loss_fn(pred, cov_t, target)
    assert loss > 0

    reduction = "sum"
    sigma_t = 3

    pred = torch.ones((1, 3, 3)) * (64 // 2)
    target = torch.ones((1, 3, 3)) * (64 // 2 - 5)

    heatmap_generator = GaussianHeatmapGenerator(3, sigmas=sigma_t, heatmap_size=(64, 64, 64))
    cov_t = heatmap_generator.get_covariance_matrix()

    loss_fn = EuclideanDistanceVarianceReg(reduction=reduction, var_t=sigma_t**2, spatial_dims=3)
    expected_output_shape = torch.Size([])

    loss = loss_fn(target, cov_t, target)

    assert loss.shape == expected_output_shape
    assert loss == 0

    loss = loss_fn(pred, cov_t, target)
    assert loss > 0

    reduction = "none"
    sigma_t = 3

    pred = torch.ones((1, 3, 3)) * (64 // 2)
    target = torch.ones((1, 3, 3)) * (64 // 2 - 5)

    heatmap_generator = GaussianHeatmapGenerator(3, sigmas=sigma_t, heatmap_size=(64, 64, 64))
    cov_t = heatmap_generator.get_covariance_matrix()

    loss_fn = EuclideanDistanceVarianceReg(reduction=reduction, var_t=sigma_t**2, spatial_dims=3)
    expected_output_shape = target.shape[:-1]

    loss = loss_fn(target, cov_t, target)

    assert loss.shape == expected_output_shape
    assert torch.allclose(loss, torch.zeros_like(loss))

    loss = loss_fn(pred, cov_t, target)
    assert torch.all(loss > 0)

    try:
        loss_fn = EuclideanDistanceVarianceReg(reduction="invalid", var_t=-1, spatial_dims=3)
        loss = loss_fn(pred, cov_t, target)
        assert False
    except ValueError:
        assert True


def test_star_loss():
    """Test the StarLoss class.""" ""
    reduction = "mean"
    pred = torch.rand(1, 3, 2) * 64
    target = torch.rand(1, 3, 2) * 64
    cov_matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).view((1, 1, 2, 2)).repeat(1, 3, 1, 1)

    loss_fn = StarLoss(reduction=reduction)
    expected_output_shape = torch.Size([])

    loss = loss_fn(pred, cov_matrix, target)

    assert loss.shape == expected_output_shape

    pred = torch.rand((4, 5, 2)) * 64
    target = torch.rand((4, 5, 2)) * 64
    cov_matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).view((1, 1, 2, 2)).repeat(4, 5, 1, 1)

    loss_fn = StarLoss(reduction=reduction)
    expected_output_shape = torch.Size([])

    loss = loss_fn(pred, cov_matrix, target)

    assert loss.shape == expected_output_shape

    reduction = "sum"

    loss_fn = StarLoss(reduction=reduction)
    expected_output_shape = torch.Size([])

    loss = loss_fn(pred, cov_matrix, target)

    assert loss.shape == expected_output_shape

    reduction = "none"

    loss_fn = StarLoss(reduction=reduction)
    expected_output_shape = torch.Size([4, 5])

    loss = loss_fn(pred, cov_matrix, target)

    assert loss.shape == expected_output_shape

    for distance in ["l1", "l2", "smooth-l1"]:
        loss_fn = StarLoss(reduction=reduction, distance=distance)
        expected_output_shape = torch.Size([4, 5])

        loss = loss_fn(pred, cov_matrix, target)

        assert loss.shape == expected_output_shape

    try:
        loss_fn = StarLoss(reduction=reduction, distance="invalid")
        assert False
    except ValueError:
        assert True


def test_star_loss_3d():
    """Test the StarLoss class. For 3D inputs."""
    reduction = "mean"
    pred = torch.rand(1, 3, 3) * 64
    target = torch.rand(1, 3, 3) * 64
    cov_matrix = (
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        .view((1, 1, 3, 3))
        .repeat(1, 3, 1, 1)
    )

    loss_fn = StarLoss(reduction=reduction, spatial_dims=3)
    expected_output_shape = torch.Size([])

    loss = loss_fn(pred, cov_matrix, target)

    assert loss.shape == expected_output_shape

    pred = torch.rand((4, 5, 3)) * 64
    target = torch.rand((4, 5, 3)) * 64
    cov_matrix = (
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        .view((1, 1, 3, 3))
        .repeat(4, 5, 1, 1)
    )

    loss_fn = StarLoss(reduction=reduction, spatial_dims=3)
    expected_output_shape = torch.Size([])

    loss = loss_fn(pred, cov_matrix, target)

    assert loss.shape == expected_output_shape

    reduction = "sum"

    loss_fn = StarLoss(reduction=reduction, spatial_dims=3)
    expected_output_shape = torch.Size([])

    loss = loss_fn(pred, cov_matrix, target)

    assert loss.shape == expected_output_shape

    reduction = "none"

    loss_fn = StarLoss(reduction=reduction, spatial_dims=3)
    expected_output_shape = torch.Size([4, 5])

    loss = loss_fn(pred, cov_matrix, target)

    assert loss.shape == expected_output_shape

    for distance in ["l1", "l2", "smooth-l1"]:
        loss_fn = StarLoss(reduction=reduction, distance=distance, spatial_dims=3)
        expected_output_shape = torch.Size([4, 5])

        loss = loss_fn(pred, cov_matrix, target)

        assert loss.shape == expected_output_shape

    try:
        loss_fn = StarLoss(reduction=reduction, distance="invalid")
        assert False
    except ValueError:
        assert True


def test_stacked_loss():
    """Test the StackedLoss class."""
    # create an instance of the stacked loss function with mean reduction
    loss_fn = nn.MSELoss
    stacked_loss_fn = StackedLoss(loss_fn, reduction="mean")

    # create some dummy input tensors

    preds = [torch.rand(2, 4, 64, 64) for _ in range(3)]
    target = torch.rand(2, 4, 64, 64)

    # calculate the loss
    loss = stacked_loss_fn(preds, target)

    # check that the output has the correct shape
    assert loss.shape == torch.Size([])

    # check that the output is a scalar tensor
    assert loss.dim() == 0

    # check that the output is non-negative
    assert loss >= 0

    # create an instance of the stacked loss function with sum reduction
    stacked_loss_fn_sum = StackedLoss(loss_fn, reduction="sum")

    # calculate the loss
    loss_sum = stacked_loss_fn_sum(preds, target)

    # check that the output has the correct shape
    assert loss_sum.shape == torch.Size([])

    # check that the output is a scalar tensor
    assert loss_sum.dim() == 0

    # check that the output is non-negative
    assert loss_sum >= 0

    # check that the sum reduction is equivalent to 2*4*64*64 times the mean reduction
    assert torch.allclose(loss_sum, 2 * 3 * 4 * 64 * 64 * loss)

    # create an instance of the stacked loss function with element-wise reduction
    stacked_loss_fn_elements = StackedLoss(loss_fn, reduction="none")

    # calculate the loss
    loss = stacked_loss_fn_elements(preds, target)

    # check that the output has the correct shape
    assert loss.shape == torch.Size([3, 2, 4, 64, 64])

    # check that the output is non-negative
    assert (loss >= 0).all()


def test_stacked_loss_3d():
    """Test the StackedLoss class. For 3D inputs."""
    # create an instance of the stacked loss function with mean reduction
    loss_fn = nn.MSELoss
    stacked_loss_fn = StackedLoss(loss_fn, reduction="mean")

    # create some dummy input tensors

    preds = [torch.rand(2, 4, 64, 64, 64) for _ in range(3)]
    target = torch.rand(2, 4, 64, 64, 64)

    # calculate the loss
    loss = stacked_loss_fn(preds, target)

    # check that the output has the correct shape
    assert loss.shape == torch.Size([])

    # check that the output is a scalar tensor
    assert loss.dim() == 0

    # check that the output is non-negative
    assert loss >= 0

    # create an instance of the stacked loss function with sum reduction
    stacked_loss_fn_sum = StackedLoss(loss_fn, reduction="sum")

    # calculate the loss
    loss_sum = stacked_loss_fn_sum(preds, target)

    # check that the output has the correct shape
    assert loss_sum.shape == torch.Size([])

    # check that the output is a scalar tensor
    assert loss_sum.dim() == 0

    # check that the output is non-negative
    assert loss_sum >= 0

    # check that the sum reduction is equivalent to 2*4*64*64 times the mean reduction
    assert torch.allclose(loss_sum, 2 * 3 * 4 * 64 * 64 * 64 * loss)

    # create an instance of the stacked loss function with element-wise reduction
    stacked_loss_fn_elements = StackedLoss(loss_fn, reduction="none")

    # calculate the loss
    loss = stacked_loss_fn_elements(preds, target)

    # check that the output has the correct shape
    assert loss.shape == torch.Size([3, 2, 4, 64, 64, 64])

    # check that the output is non-negative
    assert (loss >= 0).all()


def test_NLLLoss_2d():
    """Test the NLLLoss class."""
    reduction = "mean"
    pred = torch.rand(2, 3, 64, 64)
    target = torch.rand(2, 3, 64, 64)

    loss_fn = NLLLoss(reduction=reduction)
    expected_output_shape = torch.Size([])
    loss = loss_fn(pred, target)
    assert loss.shape == expected_output_shape

    pred = torch.rand(2, 3, 64, 64)
    target = torch.rand(2, 3, 64, 64)

    loss_fn = NLLLoss(reduction="sum")
    expected_output_shape = torch.Size([])

    loss = loss_fn(pred, target)
    assert loss.shape == expected_output_shape

    loss_fn = NLLLoss(reduction="none")
    expected_output_shape = torch.Size([2, 3])

    loss = loss_fn(pred, target)
    assert loss.shape == expected_output_shape
