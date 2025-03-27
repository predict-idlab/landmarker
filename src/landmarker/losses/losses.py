"""Heatmap loss functions."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from landmarker.models.utils import LogSoftmaxND


class GeneralizedNormalHeatmapLoss(nn.Module):
    """
    Loss function for adaptive generalized normal direct heatmap regression. The loss function is an
    extension of the loss function proposed by Thaler et al. (2021) for adaptive heatmap regression,
    where they used a anistropic Gaussian distribution for adaptive heatmap regression. The loss
    function is defined as the sum of a specified distance function between the predicted heatmap
    and the target heatmap, additionaly a determinant of the supplied covariance matrix is added as
    regularization term to penalize the loss of the fitted covariance matrix.

    # TODO: add formula.
        source: Modeling Annotation Uncertainty with Gaussian Heatmaps in Landmark Localization

    Args:
        alpha (float, optional): Weight of the regularization term. Defaults to 5.0.
        distance (str, optional): Distance function to use for the loss calculation. Defaults to
            'l2'. Possible values are 'l2', 'l1', 'smooth-l1',
            'bce-with-logits' and 'bce'.
        reduction (str, optional): Specifies the reduction to apply to the output. Defaults to
            'mean'.
        **kwargs: Additional keyword arguments for the distance function.
    """

    def __init__(
        self, alpha: float = 5.0, distance: str = "l2", reduction: str = "mean", **kwargs
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.distance = distance
        self.reduction = reduction
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction: {reduction}")
        if self.distance == "l2":
            self.dist_func = lambda x, y: F.mse_loss(x, y, reduction="sum")
        elif self.distance == "l1":
            self.dist_func = lambda x, y: F.l1_loss(x, y, reduction="sum")
        elif self.distance == "smooth-l1":
            self.dist_func = lambda x, y: F.smooth_l1_loss(x, y, reduction="sum", **kwargs)
        elif self.distance == "bce-with-logits":
            self.dist_func = lambda x, y: F.binary_cross_entropy_with_logits(x, y, reduction="sum")
        elif self.distance == "bce":
            self.dist_func = lambda x, y: F.binary_cross_entropy(x, y, reduction="sum")
        else:
            raise ValueError(f"Invalid distance function: {self.distance}")

    def forward(
        self, heatmap: torch.Tensor, cov: torch.Tensor, heatmap_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            heatmap (torch.Tensor): Predicted heatmap.
            cov (torch.Tensor): Covariance matrix of the fitted heatmap.
            heatmap_target (torch.Tensor): Target heatmap.

        Returns:
            torch.Tensor: Loss value.
        """
        det_cov = torch.det(cov)
        assert torch.all(det_cov > 0), f"Determinant of covariance matrix is negative: {det_cov}"
        loss = self.dist_func(heatmap, heatmap_target) / heatmap.shape[0] + self.alpha * torch.sum(
            torch.sqrt(det_cov)
        )
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        return loss


class GaussianHeatmapL2Loss(nn.Module):
    """
    Loss function for Gaussian heatmap regression.
    source: http://arxiv.org/abs/2109.09533"""

    def __init__(self, alpha=5, reduction="mean", spatial_dims=2):
        super(GaussianHeatmapL2Loss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.spatial_dims = spatial_dims
        if spatial_dims not in [2, 3]:
            raise ValueError("spatial_dims must be 2 or 3")

    def forward(self, heatmap, sigmas, heatmap_target):
        if self.spatial_dims == 2:
            loss = F.mse_loss(heatmap, heatmap_target, reduction="sum") / heatmap.shape[
                0
            ] + self.alpha * torch.sum((sigmas[:, 0] * sigmas[:, 1]))
        else:
            loss = F.mse_loss(heatmap, heatmap_target, reduction="sum") / heatmap.shape[
                0
            ] + self.alpha * torch.sum((sigmas[..., 0] * sigmas[..., 1]) * sigmas[..., 2])
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        return loss


class EuclideanDistanceVarianceReg(nn.Module):
    """
    Euclidean distance loss with variance regularization. The regularization term is defined as the
    squared difference between the fitted/predicted variance and a predefined target variance, as
    proposed by Nibali et al. (2018). The authors point out that this regularization term does not
    directly constrain the specif shape of the learned heatmaps.
        source: Numerical Coordinate Regression with Convolutional Neural Networks - Nibali et al.
            (2018)

    Args:
        alpha (float, optional): Weight of the regularization term. Defaults to 1.0.
        var_t (float, optional): Target variance. Defaults to 1.0.
        reduction (str, optional): Specifies the reduction to apply to the output. Defaults to
            'mean'.
        eps (float, optional): Epsilon value to avoid division by zero. Defaults to 1e-6.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        var_t: float = 1.0,
        reduction: str = "mean",
        eps: float = 1e-6,
        spatial_dims=2,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.var_t = var_t
        self.reduction = reduction
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.eps = eps
        self.spatial_dims = spatial_dims
        if spatial_dims not in [2, 3]:
            raise ValueError("spatial_dims must be 2 or 3")

    def forward(self, pred: torch.Tensor, cov: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred (torch.Tensor): Predicted coordinates.
            cov (torch.Tensor): Related covariance matrix of the predicted coordinates.
            target (torch.Tensor): Target coordinates.
        """
        if self.spatial_dims == 2:
            reg = (cov[..., 0, 0] - self.var_t) ** 2 + (cov[..., 1, 1] - self.var_t) ** 2
        else:
            reg = (
                (cov[..., 0, 0] - self.var_t) ** 2
                + (cov[..., 1, 1] - self.var_t) ** 2
                + (cov[..., 2, 2] - self.var_t) ** 2
            )
        loss = _euclidean_distance(pred, target) + self.alpha * reg
        if self.reduction == "mean":
            return torch.mean(loss)
        if self.reduction == "sum":
            return torch.sum(loss)
        return loss


class MultivariateGaussianNLLLoss(nn.Module):
    """
    Negative log-likelihood loss for multivariate Gaussian distributions. The loss function is
    defined as the negative log-likelihood of the predicted coordinates given the predicted
    covariance matrix. The loss function is defined as:
        :math:`NLL = 0.5 * (log(det(Cov)) + (x - mu)^T * Cov^{-1} * (x - mu))`

        As proposed in: "UGLLI Face Alignment: Estimating Uncertainty with Gaussian Log-Likelihood
            Loss" - Kumar et al. (2019)

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output. Defaults to
            'mean'.
        eps (float, optional): Epsilon value to avoid division by zero. Defaults to 1e-6.
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-6, spatial_dims: int = 2):
        super().__init__()
        self.reduction = reduction
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.eps = eps
        self.spatial_dims = spatial_dims

    def forward(self, pred: torch.Tensor, cov: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred (torch.Tensor): Predicted coordinates.
            cov (torch.Tensor): Related covariance matrix of the predicted coordinates.
            target (torch.Tensor): Target coordinates.
        """
        loss = 0.5 * (
            torch.log(torch.det(cov) + self.eps)
            + (
                (target - pred).unsqueeze(-2)
                @ torch.linalg.inv(cov)
                @ (target - pred).unsqueeze(-2).transpose(-2, -1)
            )
            .squeeze(-1)
            .squeeze(-1)
        )
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        return loss


class WingLoss(nn.Module):
    r"""
    Wing loss, proposed for facial landmark detection by Feng et al. (2018), is a piece-wise loss
    function that focusses more attention on small and medium range erros compared to L2, L1, and
    smooth L1. It has large gradient when the error is small and a constant gradient when the error
    is large.
    The loss function is defined as:

    .. math::
        Wing(x, y) =
        \begin{cases}
            \omega * log(1 + \frac{|x - y|}{\epsilon}) & \text{if } |x - y| < \omega \\
            |x - y| - C & \text{otherwise}
        \end{cases}

    where :math:`C = \omega - \omega * log(1 + \frac{\omega}{\epsilon})`
        source: "Wing Loss for Robust Facial Landmark Localisation With Convolutional Neural
            Networks" - Feng et al. (2018)

    Args:
        omega (float, optional): Wing loss parameter. Defaults to 5.0.
        epsilon (float, optional): Wing loss parameter. Defaults to 0.5.
        reduction (str, optional): Specifies the reduction to apply to the output. Defaults to
            'mean'.
    """

    def __init__(
        self, omega: float = 5.0, epsilon: float = 0.5, reduction: str = "mean", spatial_dims=2
    ):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.reduction = reduction
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.c = self.omega * (1 - np.log(1 + self.omega / self.epsilon))
        self.spatial_dims = spatial_dims
        if self.spatial_dims == 2:
            raise ValueError("Only 2D heatmaps are supported.")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred (torch.Tensor): Predicted coordinates.
            target (torch.Tensor): Target coordinates.
        """
        diff_abs = torch.abs(pred - target)
        loss = torch.where(
            diff_abs < self.omega,
            self.omega * torch.log(1 + diff_abs / self.epsilon),
            diff_abs - self.c,
        )
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        return loss


class AdaptiveWingLoss(nn.Module):
    r"""
    Adaptive wing loss is a loss function that behaves like a smoothed Wing loss when the target is
    close to 1 and like the MSE loss when the target is close to 0.

    The loss function is defined as:

    .. math::
        AWing(x, y) =
        \begin{cases}
            \omega \log(1 + |\frac{x-y}{\epsilon}|^{\alpha-y} &
                \text{if } |x - y| < \theta \\
            A|x - y| - C & \text{otherwise}
        \end{cases}})
    where :math:`A = \omega (1/(1+(\theta/\epsilon)^{\alpha-y}))(\alpha - y)
                    ((\theta/\epsilon)^(\alpha-y-1))(1/\epsilon)` and
                :math:`C = (\theta * A - \omega * log(1 + (\theta/\epsilon)^{\alpha-y})))`
        source: "Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression" - Wang et al.
            (2019)

    Args:
        omega (float, optional): Adaptive wing loss parameter. Defaults to 5.0.
        epsilon (float, optional): Adaptive wing loss parameter. Defaults to 0.5.
        alpha (float, optional): Adaptive wing loss parameter. Defaults to 2.1.
        theta (float, optional): Adaptive wing loss parameter. Defaults to 0.5.
        reduction (str, optional): Specifies the reduction to apply to the output. Defaults to
            'mean'.
    """

    def __init__(
        self,
        omega: float = 5,
        epsilon: float = 0.5,
        alpha: float = 2.1,
        theta: float = 0.5,
        reduction: str = "mean",
        spatial_dims=2,
    ) -> None:
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.alpha = alpha
        self.theta = theta
        self.reduction = reduction
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.spatial_dims = spatial_dims
        if self.spatial_dims != 2:
            raise ValueError("Only 2D heatmaps are supported.")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred (torch.Tensor): Predicted coordinates.
            target (torch.Tensor): Target coordinates.
        """
        diff_abs = torch.abs(pred - target)
        a = (
            self.omega
            * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - target)))
            * (self.alpha - target)
            * torch.pow(self.theta / self.epsilon, self.alpha - target - 1)
            * (1 / self.epsilon)
        )
        c = self.theta * a - self.omega * torch.log(
            1 + torch.pow(self.theta / self.epsilon, self.alpha - target)
        )
        loss = torch.where(
            diff_abs < self.theta,
            self.omega * torch.log(1 + torch.pow(diff_abs / self.epsilon, self.alpha - target)),
            a * diff_abs - c,
        )
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        return loss


class StarLoss(nn.Module):
    """
    Self-adapTive Ambiguity Reduction (STAR) loss. Star loss takes into account the ambuigity
    (uncertainty) of the intermediate heatmap predictions by using the covariance matrix of the
    heatmap predictions, and extracting the eigenvectors and eigenvalues of the covariance matrix.

        source: "STAR Loss: Reducing Semantic Ambiguity in Facial Landmark Detection" - Zhou et al.
            (2023)

    Args:
        omega (float, optional): Weight of the regularization term. Defaults to 1.0.
        distance (str, optional): Distance function to use for the loss calculation. Defaults to
            'l2'. Possible values are 'l2', 'l1', 'smooth-l1'.
        reduction (str, optional): Specifies the reduction to apply to the output. Defaults to
            'mean'.
        **kwargs: Additional keyword arguments for the distance function.
    """

    def __init__(
        self,
        omega: float = 1.0,
        distance: str = "l2",
        epsilon: float = 1e-5,
        reduction: str = "mean",
        spatial_dims=2,
        **kwargs,
    ) -> None:
        super().__init__()
        self.omega = omega
        self.reduction = reduction
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.distance = distance
        self.spatial_dims = spatial_dims
        if self.distance == "l2":
            self.dist_func = lambda x, y: F.mse_loss(x, y, reduction="sum")
        elif self.distance == "l1":
            self.dist_func = lambda x, y: F.l1_loss(x, y, reduction="sum")
        elif self.distance == "smooth-l1":
            self.dist_func = lambda x, y: F.smooth_l1_loss(x, y, reduction="sum", **kwargs)
        else:
            raise ValueError(f"Invalid distance function: {self.distance}")
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, cov: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred (torch.Tensor): Predicted coordinates.
            cov (torch.Tensor): Related covariance matrix of the predicted coordinates.
            target (torch.Tensor): Target coordinates.
        """
        eig_values, eig_vecs = torch.linalg.eig(cov)
        eig_values = eig_values.float()
        eig_vecs = eig_vecs.float()
        if self.spatial_dims == 2:
            loss = (1 / torch.sqrt(eig_values[..., 0]) + self.epsilon) * self.dist_func(
                eig_vecs[..., 0].unsqueeze(-2) @ pred.unsqueeze(-1),
                eig_vecs[..., 0].unsqueeze(-2) @ target.unsqueeze(-1),
            ) + (1 / torch.sqrt(eig_values[..., 1]) + self.epsilon) * self.dist_func(
                eig_vecs[..., 1].unsqueeze(-2) @ pred.unsqueeze(-1),
                eig_vecs[..., 1].unsqueeze(-2) @ target.unsqueeze(-1),
            )
        else:
            loss = (
                (1 / torch.sqrt(eig_values[..., 0]) + self.epsilon)
                * self.dist_func(
                    eig_vecs[..., 0].unsqueeze(-2) @ pred.unsqueeze(-1),
                    eig_vecs[..., 0].unsqueeze(-2) @ target.unsqueeze(-1),
                )
                + (1 / torch.sqrt(eig_values[..., 1]) + self.epsilon)
                * self.dist_func(
                    eig_vecs[..., 1].unsqueeze(-2) @ pred.unsqueeze(-1),
                    eig_vecs[..., 1].unsqueeze(-2) @ target.unsqueeze(-1),
                )
                + (1 / torch.sqrt(eig_values[..., 2]) + self.epsilon)
                * self.dist_func(
                    eig_vecs[..., 2].unsqueeze(-2) @ pred.unsqueeze(-1),
                    eig_vecs[..., 2].unsqueeze(-2) @ target.unsqueeze(-1),
                )
            )
        loss += self.omega / 2 * torch.sum(eig_values, dim=-1)
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        return loss


class NLLLoss(nn.Module):
    """Negative log-likelihood loss for 2D/3D heatmaps. Assumes that the input is a probability
    distribution and calculates the negative log-likelihood of the predicted heatmap given the
    target heatmap.

    Args:
        spatial_dims (int, optional): Spatial dimension of the heatmaps. Defaults to 2.
        reduction (str, optional): Specifies the reduction to apply to the output. Defaults to
            'mean'.
    """

    def __init__(self, spatial_dims: int = 2, apply_softmax: bool = True, reduction: str = "mean"):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.apply_softmax = apply_softmax
        if self.apply_softmax:
            self.log_softmax = LogSoftmaxND(spatial_dims)
        if spatial_dims not in [2, 3]:
            raise ValueError("spatial_dims must be 2 or 3")
        self.reduction = reduction
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction: {reduction}")

    def forward(self, output, target):
        if self.apply_softmax:
            output = self.log_softmax(output)
        else:
            output = torch.log(output)
        nll = -target * output
        if self.spatial_dims == 2:
            dim = (2, 3)
        else:
            dim = (2, 3, 4)
        if self.reduction == "mean":
            return torch.mean(torch.sum(nll, dim=dim))
        if self.reduction == "sum":
            return torch.sum(torch.sum(nll, dim=dim))
        return torch.sum(nll, dim=dim)


class StackedLoss(nn.Module):
    """Stacked loss function. Applies a specified loss function to each list of predictions. This
    loss function is used to calculate the loss for each heatmap in the stacked heatmap regression,
    suchs as stacked hourglass and U-Net networks.

    Args:
        loss_fn (nn.Module): Loss function to use.
        reduction (str, optional): Specifies the reduction to apply to the output. Defaults to
            'mean'.
    """

    def __init__(self, loss_fn: nn.Module, reduction: str = "mean", **kwargs) -> None:
        super().__init__()
        self.loss_fn = loss_fn(reduction=reduction, **kwargs)
        self.reduction = reduction
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction: {reduction}")

    def forward(self, preds: list[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds(list[torch.Tensor]): List of predicted heatmaps or coordinates.
            target (torch.Tensor): Target heatmap.
        """
        losses = []
        for pred in preds:
            losses.append(self.loss_fn(pred, target))
        if self.reduction == "mean":
            return torch.stack(losses).mean()
        if self.reduction == "sum":
            return torch.stack(losses).sum()
        return torch.stack(losses)


def _euclidean_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Computes the euclidean distance between two tensors."""
    return torch.sqrt(torch.sum((x1 - x2) ** 2, dim=-1))
