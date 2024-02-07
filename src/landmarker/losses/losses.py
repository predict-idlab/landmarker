"""Heatmap loss functions."""

from typing import Optional

import numpy as np
import torch
from kornia.losses import js_div_loss_2d
from torch import nn
from torch.nn import functional as F

from landmarker.heatmap.generator import (
    GaussianHeatmapGenerator,
    HeatmapGenerator,
    LaplacianHeatmapGenerator,
)


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

    def __init__(self, alpha=5, reduction="mean"):
        super(GaussianHeatmapL2Loss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, heatmap, sigmas, heatmap_target):
        loss = F.mse_loss(heatmap, heatmap_target, reduction="sum") / heatmap.shape[
            0
        ] + self.alpha * torch.sum((sigmas[:, 0] * sigmas[:, 1]))
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
        self, alpha: float = 1.0, var_t: float = 1.0, reduction: str = "mean", eps: float = 1e-6
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.var_t = var_t
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred: torch.Tensor, cov: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred (torch.Tensor): Predicted coordinates.
            cov (torch.Tensor): Related covariance matrix of the predicted coordinates.
            target (torch.Tensor): Target coordinates.
        """
        reg = (cov[..., 0, 0] - self.var_t) ** 2 + (cov[..., 1, 1] - self.var_t) ** 2
        loss = _euclidean_distance(pred, target) + self.alpha * reg
        if self.reduction == "mean":
            return torch.mean(loss)
        if self.reduction == "sum":
            return torch.sum(loss)
        if self.reduction == "none":
            return loss
        raise ValueError(f"Invalid reduction: {self.reduction}")


class EuclideanDistanceJSDivergenceReg(nn.Module):
    r"""
    Euclidean distance loss with Jensen-Shannon divergence regularization. The regularization term
    imposes a Gaussian distribution on the predicted heatmaps and calculates the Jensen-Shannon
    divergence between the predicted and the target heatmap. The Jensen-Shannon divergence is a
    method to measure the similarity between two probability distributions. It is a symmetrized
    and smoothed version of the Kullback-Leibler divergence, and is defined as the average of the
    Kullback-Leibler divergence between the two distributions and a mixture M of the two
    distributions:
        :math:`JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)` where :math:`M = 0.5 * (P + Q)`
    Generalization of regularization term proposed by Nibali et al. (2018), which only considers
    Multivariate Gaussian distributions, to a generalized Gaussian distribution. (However, now
    we only consider the Gaussian and the Laplace distribution.)

        source: Numerical Coordinate Regression with Convolutional Neural Networks - Nibali et al.
            (2018)

    Args:
        alpha (float, optional): Weight of the regularization term. Defaults to 1.0.
        sigma_t (float, optional): Target sigma value. Defaults to 1.0.
        rotation_t (float, optional): Target rotation value. Defaults to 0.0.
        nb_landmarks (int, optional): Number of landmarks. Defaults to 1.
        heatmap_fun (str, optional): Specifies the type of heatmap function to use. Defaults to
            'gaussian'. Possible values are 'gaussian' and 'laplacian'.
        heatmap_size (tuple[int, int], optional): Size of the heatmap. Defaults to (512, 512).
        gamma (Optional[float], optional): Gamma value for the Laplace distribution. Defaults to
            1.0.
        reduction (str, optional): Specifies the reduction to apply to the output. Defaults to
            'mean'.
        eps (float, optional): Epsilon value to avoid division by zero. Defaults to 1e-6.
    """
    # TODO: Implement generalized Gaussian distribution. (Currently only Gaussian and Laplace)

    def __init__(
        self,
        alpha: float = 1.0,
        sigma_t: float | torch.Tensor = 1.0,
        rotation_t: float | torch.Tensor = 0.0,
        nb_landmarks: int = 1,
        heatmap_fun: str = "gaussian",
        heatmap_size: tuple[int, int] = (512, 512),
        gamma: Optional[float] = 1.0,
        reduction: str = "mean",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.eps = eps
        self.heatmap_fun: HeatmapGenerator
        if heatmap_fun == "gaussian":
            self.heatmap_fun = GaussianHeatmapGenerator(
                nb_landmarks=nb_landmarks,
                sigmas=sigma_t,
                rotation=rotation_t,
                heatmap_size=heatmap_size,
                gamma=gamma,
            )
        elif heatmap_fun == "laplacian":
            self.heatmap_fun = LaplacianHeatmapGenerator(
                nb_landmarks=nb_landmarks,
                sigmas=sigma_t,
                rotation=rotation_t,
                heatmap_size=heatmap_size,
                gamma=gamma,
            )
        else:
            raise ValueError(f"Invalid heatmap function: {heatmap_fun}")

    def forward(
        self, pred: torch.Tensor, pred_heatmap: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred (torch.Tensor): Predicted coordinates.
            pred_heatmap (torch.Tensor): Predicted heatmap.
            target (torch.Tensor): Target coordinates.
        """
        heatmap_t = self.heatmap_fun(target)
        # Normalize heatmaps to sum to 1
        heatmap_t = (heatmap_t + self.eps) / (heatmap_t + self.eps).sum(dim=(-2, -1), keepdim=True)
        pred_heatmap = (pred_heatmap + self.eps) / (
            pred_heatmap.sum(dim=(-2, -1), keepdim=True) + self.eps
        )
        reg = js_div_loss_2d(pred_heatmap, heatmap_t, reduction="none")
        loss = _euclidean_distance(pred, target) + self.alpha * reg
        if self.reduction == "mean":
            return torch.mean(loss)
        if self.reduction == "sum":
            return torch.sum(loss)
        if self.reduction == "none":
            return loss
        raise ValueError(f"Invalid reduction: {self.reduction}")


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

    def __init__(self, reduction: str = "mean", eps: float = 1e-6):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred: torch.Tensor, cov: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred (torch.Tensor): Predicted coordinates.
            cov (torch.Tensor): Related covariance matrix of the predicted coordinates.
            target (torch.Tensor): Target coordinates.
        """
        loss = 0.5 * (
            torch.log(torch.det(cov))
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

    def __init__(self, omega: float = 5.0, epsilon: float = 0.5, reduction: str = "mean"):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.reduction = reduction
        self.c = self.omega * (1 - np.log(1 + self.omega / self.epsilon))

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
    ) -> None:
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.alpha = alpha
        self.theta = theta
        self.reduction = reduction

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
        **kwargs,
    ) -> None:
        super().__init__()
        self.omega = omega
        self.reduction = reduction
        self.distance = distance
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
        loss = (1 / torch.sqrt(eig_values[..., 0]) + self.epsilon) * self.dist_func(
            eig_vecs[..., 0].unsqueeze(-2) @ pred.unsqueeze(-1),
            eig_vecs[..., 0].unsqueeze(-2) @ target.unsqueeze(-1),
        ) + (1 / torch.sqrt(eig_values[..., 1]) + self.epsilon) * self.dist_func(
            eig_vecs[..., 1].unsqueeze(-2) @ pred.unsqueeze(-1),
            eig_vecs[..., 1].unsqueeze(-2) @ target.unsqueeze(-1),
        )
        loss += self.omega / 2 * torch.sum(eig_values, dim=-1)
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        return loss


class StackedLoss(nn.Module):
    """Stacked loss function. Applies a specified loss function to each list of predictions. This
    loss function is used to calculate the loss for each heatmap in the stacked heatmap regression,
    suchs as stacked hourglass and U-Net networks.

    Args:
        loss_fn (nn.Module): Loss function to use.
        reduction (str, optional): Specifies the reduction to apply to the output. Defaults to
            'mean'.
    """

    def __init__(self, loss_fn: nn.Module, reduction: str = "mean") -> None:
        super().__init__()
        self.loss_fn = loss_fn(reduction=reduction)
        self.reduction = reduction

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
        if self.reduction == "none":
            return torch.stack(losses)
        raise ValueError(f"Invalid reduction: {self.reduction}")


def _euclidean_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Computes the euclidean distance between two tensors."""
    return torch.sqrt(torch.sum((x1 - x2) ** 2, dim=-1))
