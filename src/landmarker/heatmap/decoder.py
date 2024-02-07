"""
Decoder module for retrieving coordinates and other statistics from heatmaps.
"""

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy import optimize  # type: ignore

from landmarker.heatmap.generator import GaussianHeatmapGenerator


def coord_argmax(heatmap: torch.Tensor) -> torch.Tensor:
    """
    Returns the coordinates of the maximum value of the heatmap
    for each batch and channel (landmark).

    Args:
        heatmap (torch.Tensor): heatmap of shape (B, C, H, W)

    Returns:
        (torch.Tensor): coordinates of shape (B, C, 2)
    """
    # TODO: Think about way to deal with ties (multiple maxima) in the output
    #    now we just take the first one
    w = heatmap.shape[-1]
    idx_pred = torch.argmax(heatmap.flatten(-2, -1), dim=-1)
    idx_pred = torch.stack((idx_pred // w, idx_pred % w), dim=-1)
    return idx_pred.int()


def coord_local_soft_argmax(
    heatmap: torch.Tensor, window: int = 3, t: float = 10.0
) -> torch.Tensor:
    """
    Returns coordiantes through applying the local soft-argmax function on the heatmaps.
        Source: "Subpixel Heatmap Regression for Facial Landmark Localization" - Bulat et al. (2021)

    Args:
        heatmap (torch.Tensor): heatmap of shape (B, C, H, W)
        idx_pred (torch.Tensor): predicted indices of shape (B, C, 2)
        window (int): local window size
        t (float ): temperature that controls the resulting probability map

    Returns:
        (torch.Tensor): coordinates of shape (B, C, 2)
    """
    padded_heatmap = F.pad(heatmap, (window // 2, window // 2, window // 2, window // 2))
    b, c, h, w = padded_heatmap.shape

    argmax_coords = coord_argmax(heatmap) + window // 2

    mask = torch.zeros((b, c, h, w))
    for b1 in range(b):
        for c1 in range(c):
            for i in range(-(window // 2), (window // 2) + 1):
                for j in range(-(window // 2), (window // 2) + 1):
                    mask[b1, c1, argmax_coords[b1, c1, 0] + i, argmax_coords[b1, c1, 1] + j] = 1
    masked_output = F.softmax(t * padded_heatmap[mask > 0].view(b, c, -1), dim=2).view(
        b, c, window, window
    )
    x_values = torch.sum(
        torch.arange(0, window)
        .view(1, 1, 1, window)
        .repeat(b, c, window, 1)
        .to(padded_heatmap.device)
        * masked_output,
        dim=(2, 3),
    )
    y_values = torch.sum(
        torch.arange(0, window)
        .view(1, 1, window, 1)
        .repeat(b, c, 1, window)
        .to(padded_heatmap.device)
        * masked_output,
        dim=(2, 3),
    )
    return (
        torch.cat((y_values.unsqueeze(2), x_values.unsqueeze(2)), dim=2)
        + coord_argmax(heatmap)
        - torch.tensor([window // 2, window // 2]).to(heatmap.device).view(1, 1, 2).repeat(b, c, 1)
    )


def coord_weighted_spatial_mean(heatmap: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Returns the spatial weighted mean of the possitive elements
    of the heatmap by the heatmap values.
    Source: "UGLLI Face Alignment: Estimating Uncertainty with
        Gaussian Log-Likelihood Loss" - Kumar et al. (2019)

    Args:
        heatmap (torch.Tensor): heatmap of shape (B, C, H, W)
        eps (float): epsilon to avoid division by zero

    Returns:
        (torch.Tensor): coordinates of shape (B, C, 2)
    """
    b, c, h, w = heatmap.shape
    xs = torch.arange(0, w, dtype=torch.float32, requires_grad=False).to(heatmap.device)
    ys = torch.arange(0, h, dtype=torch.float32, requires_grad=False).to(heatmap.device)
    xy = (
        torch.stack(torch.meshgrid(xs, ys, indexing="xy"), dim=2)
        .view(1, 1, h, w, 2)
        .repeat(b, c, 1, 1, 1)
    )
    heatmap_adj = F.relu(heatmap)
    heatmap_adj = heatmap_adj / torch.clip(
        torch.sum(heatmap_adj, dim=(2, 3)).unsqueeze(-1).unsqueeze(-1), min=eps
    )
    return torch.sum(heatmap_adj.unsqueeze(4) * xy, dim=(2, 3)).flip(-1)


def coord_soft_argmax(
    heatmap: torch.Tensor, eps: float = 1e-6, logit_scale: bool = False, require_grad: bool = False
) -> torch.Tensor:
    """
    Returns the spatial mean over the softmax distribution of the heatmap.
    Source: “2D/3D Pose Estimation and Action Recognition using Multitask
        Deep Learning” - Luvizon et al. (2018)

    Args:
        heatmap (torch.Tensor): heatmap of shape (B, C, H, W)
        eps (float): epsilon to avoid division by zero
        logit_scale (bool): whether the input is logit scaled
        require_grad (bool): whether to require gradient for the coordinates
    """
    b, c, h, w = heatmap.shape
    xs = torch.arange(0, w, dtype=torch.float32, requires_grad=require_grad).to(heatmap.device)
    ys = torch.arange(0, h, dtype=torch.float32, requires_grad=require_grad).to(heatmap.device)
    xy = (
        torch.stack(torch.meshgrid(xs, ys, indexing="xy"), dim=2)
        .view(1, 1, h, w, 2)
        .repeat(b, c, 1, 1, 1)
    )
    if not logit_scale:
        heatmap = heatmap / torch.clip(
            torch.sum(heatmap, dim=(2, 3)).unsqueeze(-1).unsqueeze(-1), min=eps
        )
        heatmap = heatmap + eps
        heatmap = torch.logit(heatmap)
    out = F.softmax(heatmap.view(b, c, -1), dim=2).view(b, c, h, w, 1)

    return torch.sum(out * xy, dim=(2, 3)).flip(-1)


def coord_soft_argmax_2d(
    heatmap: torch.Tensor, eps: float = 1e-6, logit_scale: bool = False, require_grad: bool = False
) -> torch.Tensor:
    """
    Returns the spatial mean over the softmax distribution of the heatmap,
    but for a 2D heatmap, without the batch and channel dimensions.

    Args:
        heatmap (torch.Tensor): heatmap of shape (H, W)
        eps (float): epsilon to avoid division by zero
        logit_scale (bool): whether the input is logit scaled
        require_grad (bool): whether to require gradient for the coordinates
    """
    return coord_soft_argmax(
        heatmap.unsqueeze(0).unsqueeze(0),
        eps=eps,
        logit_scale=logit_scale,
        require_grad=require_grad,
    )[0, 0]


def heatmap_to_coord(heatmap: torch.Tensor, offset_coords: int = 0, method: str = "argmax"):
    """
    Returns the retrieved coordinates via specified method from a heatmap. The offset_coords is used
    to remove the first offset_coords coordinates from the heatmap. This is used to remove the
    background class (if present).

    Args:
        heatmap (torch.Tensor): heatmap of shape (B, C, H, W)
        offset_coords (int): number of coordinates to remove
        method (str): method to retrieve the coordinates

    Returns:
        (torch.Tensor): coordinates of shape (B, C, 2)
    """
    heatmap = heatmap[:, offset_coords:, :, :]
    if method == "argmax":
        return coord_argmax(heatmap)
    if method == "local_soft_argmax":
        return coord_local_soft_argmax(heatmap)
    if method == "weighted_spatial_mean":
        return coord_weighted_spatial_mean(heatmap)
    if method == "soft_argmax":
        return coord_soft_argmax(heatmap)
    if method == "soft_argmax_logit":
        return coord_soft_argmax(heatmap, logit_scale=True)
    raise ValueError("Method not implemented.")


def heatmap_to_coord_enlarge(
    heatmap: torch.Tensor,
    offset_coords: int = 0,
    method: str = "argmax",
    enlarge_factor: int = 1,
    enlarge_dim: Optional[tuple[int, int]] = None,
    enlarge_mode: str = "bilinear",
) -> torch.Tensor:
    """
    Returns the retrieved coordinates via specified method from an enlarged heatmap.
    The offset_coords is used to remove the first offset_coords coordinates from the heatmap.
    This is used to remove the background class (if present).

    Args:
        heatmap (torch.Tensor): heatmap of shape (B, C, H, W)
        offset_coords (int): number of coordinates to remove
        method (str): method to retrieve the coordinates
        enlarge_factor (int): factor to enlarge the heatmap
        enlarge_dim (tuple[int, int] or None): dimensions to enlarge the heatmap to
        enlarge_mode (str): interpolation mode to enlarge the heatmap

    Returns:
        (torch.Tensor): coordinates of shape (B, C, 2)
    """
    heatmap = heatmap[:, offset_coords:, :, :]
    if enlarge_dim is not None:
        heatmap_scaled = F.interpolate(
            heatmap, size=enlarge_dim, mode=enlarge_mode, align_corners=False
        )
    else:
        heatmap_scaled = F.interpolate(
            heatmap, scale_factor=enlarge_factor, mode=enlarge_mode, align_corners=False
        )
    coords = heatmap_to_coord(heatmap_scaled, method=method)
    if enlarge_dim is not None:
        coords = coords / (
            torch.tensor(enlarge_dim, dtype=torch.float).view((1, 1, 2)).to(heatmap.device)
            / torch.tensor(heatmap.shape[-2:], dtype=torch.float).view((1, 1, 2)).to(heatmap.device)
        )
    else:
        coords = coords / enlarge_factor
    return coords


def coord_soft_argmax_cov(
    heatmap: torch.Tensor, eps: float = 1e-6, logit_scale: bool = True, require_grad: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the spatial mean over the softmax distribution of the heatmap and the covariance matrix.
    The covariance matrix is the weigthed sample covariance matrix.

    Args:
        heatmap (torch.Tensor): heatmap of shape (B, C, H, W)
        eps (float): epsilon to avoid division by zero
        logit_scale (bool): whether the input is logit scaled
        require_grad (bool): whether to require gradient for the coordinates

    Returns:
        (torch.Tensor): coordinates of shape (B, C, 2)
        (torch.Tensor): covariance matrix of shape (B, C, 2, 2)
    """
    b, c, h, w = heatmap.shape
    xs = torch.arange(0, w, dtype=torch.float32, requires_grad=require_grad).to(heatmap.device)
    ys = torch.arange(0, h, dtype=torch.float32, requires_grad=require_grad).to(heatmap.device)
    xs, ys = torch.meshgrid(xs, ys, indexing="xy")
    yx = torch.stack((ys, xs), dim=2).view(1, 1, h, w, 2).repeat(b, c, 1, 1, 1)
    if not logit_scale:
        heatmap = heatmap / torch.clip(
            torch.sum(heatmap, dim=(2, 3)).unsqueeze(-1).unsqueeze(-1), min=eps
        )
        heatmap = heatmap + eps
        heatmap = torch.logit(heatmap)
    heatmap = F.softmax(heatmap.view(b, c, -1), dim=2).view(b, c, h, w)
    assert heatmap.isnan().sum() == 0
    mean_coords = torch.sum(heatmap.unsqueeze(-1) * yx, dim=(2, 3))

    dist_x = xs - mean_coords[:, :, 1].unsqueeze(-1).unsqueeze(-1)
    dist_y = ys - mean_coords[:, :, 0].unsqueeze(-1).unsqueeze(-1)
    dist = torch.stack(
        (dist_y.flatten(start_dim=-2, end_dim=-1), dist_x.flatten(start_dim=-2, end_dim=-1)), dim=-1
    )  # (B, C, H*W, 2)

    covariances = (
        (heatmap.flatten(start_dim=-2, end_dim=-1).unsqueeze(-1) * dist).transpose(-1, -2) @ dist
    ).view(b, c, 2, 2)

    v2 = torch.sum(heatmap**2, dim=(2, 3)).unsqueeze(-1).unsqueeze(-1)
    # v1 = torch.sum(heatmap, dim=(2, 3)).unsqueeze(-1).unsqueeze(-1)

    covariances = covariances / (1 - v2)
    return mean_coords, covariances


def coord_weighted_spatial_mean_cov(
    heatmap: torch.Tensor, eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the spatial weighted mean and the weighted sample covariance of the possitive elements
    of the heatmap by the heatmap values.
        source: https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_covariance

    Args:
        heatmap (torch.Tensor): heatmap of shape (B, C, H, W)
        eps (float): epsilon to avoid division by zero

    Returns:
        (torch.Tensor): coordinates of shape (B, C, 2)
        (torch.Tensor): covariance matrix of shape (B, C, 2, 2)
    """
    # TODO: The method seems to be kind of unstable. Probably inherent to the method.
    # TODO: Point close to the edges will proabbly suffer the most (find way to counteract this)
    b, c, h, w = heatmap.shape
    xs = torch.arange(0, w, dtype=torch.float32, requires_grad=True).to(heatmap.device)
    ys = torch.arange(0, h, dtype=torch.float32, requires_grad=True).to(heatmap.device)
    xs, ys = torch.meshgrid(xs, ys, indexing="xy")
    yx = torch.stack((ys, xs), dim=2).view(1, 1, h, w, 2).repeat(b, c, 1, 1, 1)
    heatmap = F.relu(heatmap)
    heatmap = heatmap / torch.clip(
        torch.sum(heatmap, dim=(2, 3)).unsqueeze(-1).unsqueeze(-1), min=eps
    )

    mean_coords = torch.sum(heatmap.unsqueeze(4) * yx, dim=(2, 3))

    dist_x = xs - mean_coords[:, :, 1].unsqueeze(-1).unsqueeze(-1)
    dist_y = ys - mean_coords[:, :, 0].unsqueeze(-1).unsqueeze(-1)
    dist = torch.stack(
        (dist_y.flatten(start_dim=-2, end_dim=-1), dist_x.flatten(start_dim=-2, end_dim=-1)), dim=-1
    )  # (B, C, H*W, 2)

    covariances = (
        (heatmap.flatten(start_dim=-2, end_dim=-1).unsqueeze(-1) * dist).transpose(-1, -2) @ dist
    ).view(b, c, 2, 2)

    v2 = torch.sum(heatmap**2, dim=(2, 3)).unsqueeze(-1).unsqueeze(-1)
    # v1 = torch.clip(torch.sum(heatmap, dim=(2, 3)).unsqueeze(-1).unsqueeze(-1), min=eps)

    return mean_coords, covariances / (1 - v2)


def heatmap_to_coord_cov(
    heatmap: torch.Tensor,
    method: str = "soft_argmax",
    eps: float = 1e-6,
    logit_scale: bool = True,
    require_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the modal coordinates and covariance matrix from a heatmap.

    Args:
        heatmap (torch.Tensor): heatmap of shape (B, C, H, W)
        method (str): method to retrieve the coordinates
        eps (float): epsilon to avoid division by zero
        logit_scale (bool): whether the input is logit scaled
        require_grad (bool): whether to require gradient for the coordinates

    Returns:
        (torch.Tensor): coordinates of shape (B, C, 2)
        (torch.Tensor): covariance matrix of shape (B, C, 2, 2)
    """
    if method == "soft_argmax":
        return coord_soft_argmax_cov(
            heatmap, eps=eps, logit_scale=logit_scale, require_grad=require_grad
        )
    if method == "weighted_spatial_mean":
        return coord_weighted_spatial_mean_cov(heatmap, eps=eps)
    raise ValueError("Method not implemented.")


def coord_cov_from_gaussian_ls(
    heatmap: torch.Tensor, gamma: Optional[float] = None, ls_library: str = "scipy"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the modal coordinates and covariance matrix from a heatmap through fitting the heatmap
    on Gaussian distribution with a specicic scaling factor gamma with help of least squares
    optimization.

    Args:
        heatmap (torch.Tensor): heatmap of shape (B, C, H, W)
        gamma (float): gamma parameter of the gaussian heatmap generator
        ls_library (str): library to use for least squares optimization. (scipy or pytorch)

    Returns:
        (torch.Tensor): coordinates of shape (B, C, 2)
        (torch.Tensor): covariance matrix of shape (B, C, 2, 2)
    """
    # TODO: see if we can use a pytorch implementation (e.g, pytorch-minimize seems to be broken)
    if ls_library == "scipy":
        return coord_cov_from_gaussian_ls_scipy(heatmap, gamma=gamma)
    raise ValueError("Method not implemented.")


def coord_cov_from_gaussian_ls_scipy(
    heatmap: torch.Tensor, gamma: Optional[float] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the modal coordinates and covariance matrix from a heatmap through fitting the heatmap
    on Gaussian distribution with a specicic scaling factor gamma with help of least squares
    optimization.

    Args:
        heatmap (torch.Tensor): heatmap of shape (B, C, H, W)
        gamma (float): gamma parameter of the gaussian heatmap generator

    Returns:
        (torch.Tensor): coordinates of shape (B, C, 2)
        (torch.Tensor): covariance matrix of shape (B, C, 2, 2)
    """

    def generate_gaussian(heatmap: torch.Tensor) -> Callable:
        """
        Returns the gaussian function for the given landmarks, sigma and rotation.
        """
        gaussian_generator = GaussianHeatmapGenerator(
            1,
            gamma=gamma,
            heatmap_size=(heatmap.shape[-2], heatmap.shape[-1]),
            device=heatmap.device,
            learnable=False,
        )

        def fun_to_minimize(x):
            gaussian_generator.set_sigmas(x[:2])
            gaussian_generator.set_rotation(x[2])
            return (
                (
                    heatmap
                    - gaussian_generator(torch.Tensor(x[3:]).view((1, 1, 2))).view(heatmap.shape)
                )
                .flatten()
                .detach()
                .cpu()
                .numpy()
            )

        return fun_to_minimize

    b, c, _, _ = heatmap.shape
    coords = torch.zeros((b, c, 2)).to(heatmap.device)
    covs = torch.zeros((b, c, 2, 2)).to(heatmap.device)
    for b1 in range(b):
        for c1 in range(c):
            init_coord = coord_argmax(heatmap[b1, c1].unsqueeze(0).unsqueeze(0))[0, 0]
            result = optimize.least_squares(
                generate_gaussian(heatmap[b1, c1]),
                np.array([1, 1, 0, init_coord[0].item(), init_coord[1].item()]),
                method="trf",
            )
            x = result.x
            coords[b1, c1] = torch.tensor([x[3], x[4]], dtype=torch.float)
            rotation = torch.tensor(
                [[np.cos(x[2]), -np.sin(x[2])], [np.sin(x[2]), np.cos(x[2])]], dtype=torch.float
            )
            diagonal = torch.diag(torch.tensor(x[:2] ** 2, dtype=torch.float))
            covs[b1, c1] = torch.mm(torch.mm(rotation, diagonal), rotation.t())
    return coords, covs


def cov_from_gaussian_ls(
    heatmap: torch.Tensor,
    coords: torch.Tensor,
    gamma: Optional[float] = None,
    ls_library: str = "scipy",
) -> torch.Tensor:
    """
    Returns covariance matrix from a heatmap through fitting the heatmap on Gaussian distribution
    with a specicic scaling factor gamma and specified coordinates
    with help of least squares optimization.

    Args:
        heatmap (torch.Tensor): heatmap of shape (B, C, H, W)
        coords (torch.Tensor): coordinates of shape (B, C, 2)
        gamma (float): gamma parameter of the gaussian heatmap generator
        ls_library (str): library to use for least squares optimization. (scipy or pytorch)

    Returns:
        (torch.Tensor): covariance matrix of shape (B, C, 2, 2)
    """
    # TODO: see if we can use a pytorch implementation (e.g, pytorch-minimize seems to be broken)
    if ls_library == "scipy":
        return cov_from_gaussian_ls_scipy(heatmap, coords, gamma=gamma)
    raise ValueError("Method not implemented.")


def cov_from_gaussian_ls_scipy(
    heatmap: torch.Tensor, coords: torch.Tensor, gamma: Optional[float] = None
) -> torch.Tensor:
    """
    Returns the covariance matrix from a heatmap through fitting the heatmap on Gaussian
    distribution with a specicic scaling factor gamma and retrieved coordinates with specified
    decoding method with help of least squares optimization.

    Args:
        heatmap (torch.Tensor): heatmap of shape (B, C, H, W)
        coords (torch.Tensor): coordinates of shape (B, C, 2)
        gamma (float): gamma parameter of the gaussian heatmap generator

    Returns:
        (torch.Tensor): covariance matrix of shape (B, C, 2, 2)
    """

    def generate_gaussian(landmarks: torch.Tensor, heatmap: torch.Tensor) -> Callable:
        """
        Returns the gaussian function for the given landmarks, sigma and rotation.
        """
        gaussian_generator = GaussianHeatmapGenerator(
            1,
            gamma=gamma,
            heatmap_size=(heatmap.shape[-2], heatmap.shape[-1]),
            device=heatmap.device,
            learnable=False,
        )

        def fun_to_minimize(x):
            gaussian_generator.set_sigmas(x[:2])
            gaussian_generator.set_rotation(x[2])
            return (
                (heatmap - gaussian_generator(landmarks.view(1, 1, 2)).view(heatmap.shape))
                .flatten()
                .detach()
                .cpu()
                .numpy()
            )

        return fun_to_minimize

    covs = torch.zeros((heatmap.shape[0], heatmap.shape[1], 2, 2)).to(heatmap.device)
    for b in range(heatmap.shape[0]):
        for c in range(heatmap.shape[1]):
            result = optimize.least_squares(
                generate_gaussian(coords[b, c], heatmap[b, c]), np.array([1, 1, 0]), method="trf"
            )
            x = result.x
            rotation = torch.tensor(
                [[np.cos(x[2]), -np.sin(x[2])], [np.sin(x[2]), np.cos(x[2])]], dtype=torch.float
            )
            diagonal = torch.diag(torch.tensor(x[:2] ** 2, dtype=torch.float))
            covs[b, c] = torch.mm(torch.mm(rotation, diagonal), rotation.t())
    return covs


# def extract_cov_pytroch(heatmap, mean_coords, gamma=None):
#     # TODO: fix python implementation
#     def generate_gaussian(landmarks, heatmap):
#         """
#         Returns the gaussian function for the given landmarks, sigma and rotation.
#         """
#         gaussian_generator = GaussianHeatmapGenerator(1, gamma=gamma, h
#                                                       eatmap_size=heatmap.shape[-2:],
#                                                       device=heatmap.device, learnable=False)
#         def fun_to_minimize(x):
#             gaussian_generator.set_sigmas(x[:2])
#             gaussian_generator.set_rotation(x[2])
#             return (heatmap -
#                     gaussian_generator(landmarks.view(1, 1, 2)).view(heatmap.shape)).flatten()
#         return fun_to_minimize
#     covs = torch.zeros((heatmap.shape[0], heatmap.shape[1], 2, 2)).to(heatmap.device)
#     for b in range(heatmap.shape[0]):
#         for c in range(heatmap.shape[1]):
#             result = optim.least_squares(generate_gaussian(mean_coords[b, c], heatmap[b, c]),
#                           torch.cat((torch.ones(2), torch.zeros(1))), method='trf', verbose=2)
#             x = result.x
#             rotation = torch.tensor([[np.cos(x[2]), -np.sin(x[2])],
#                                      [np.sin(x[2]), np.cos(x[2])]])
#             diagonal = torch.diag(x[:2]**2)
#             covs[b, c] = torch.mm(torch.mm(rotation, diagonal), rotation.t())
#     return covs


def heatmap_coord_to_weighted_sample_cov(
    heatmap: torch.Tensor, coords: torch.Tensor, eps: float = 1e-6, apply_softmax: bool = False
) -> torch.Tensor:
    """
    Returns the weighted sample covariance matrix from a heatmap and coordinates.
        source: https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_covariance

    Args:
        heatmap (torch.Tensor): heatmap of shape (B, C, H, W)
        coords (torch.Tensor): coordinates of shape (B, C, 2)
        eps (float): epsilon to avoid division by zero
        apply_softmax (bool): whether to apply softmax on the heatmap

    Returns:
        (torch.Tensor): covariance matrix of shape (B, C, 2, 2)
    """
    b, c, h, w = heatmap.shape
    if apply_softmax:
        return weighted_sample_cov(
            F.softmax(heatmap.view(b, c, -1), dim=2).view(b, c, h, w), coords, eps=eps
        )
    return weighted_sample_cov(heatmap, coords, eps=eps)


def weighted_sample_cov(
    heatmap: torch.Tensor, coords: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Calculate the covariance matrix from a heatmap by calculating the mean of the
    heatmap values weighted by the heatmap values.
    source: https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_covariance

    Args:
        heatmap (torch.Tensor): heatmap of shape (B, C, H, W)
        coords (torch.Tensor): coordinates of shape (B, C, 2)
        eps (float): epsilon to avoid division by zero

    Returns:
        (torch.Tensor): covariance matrix of shape (B, C, 2, 2)
    """
    heatmap = F.relu(heatmap)
    heatmap = heatmap / torch.clip(
        torch.sum(heatmap, dim=(2, 3)).unsqueeze(-1).unsqueeze(-1), min=eps
    )
    b, c, h, w = heatmap.shape
    xs = torch.arange(0, w, dtype=torch.float32, requires_grad=False).to(heatmap.device)
    ys = torch.arange(0, h, dtype=torch.float32, requires_grad=False).to(heatmap.device)
    xs, ys = torch.meshgrid(xs, ys, indexing="xy")

    dist_x = xs - coords[:, :, 1].unsqueeze(-1).unsqueeze(-1)
    dist_y = ys - coords[:, :, 0].unsqueeze(-1).unsqueeze(-1)
    dist = torch.stack(
        (dist_y.flatten(start_dim=-2, end_dim=-1), dist_x.flatten(start_dim=-2, end_dim=-1)), dim=-1
    )  # (B, C, H*W, 2)

    covariances = (
        (heatmap.flatten(start_dim=-2, end_dim=-1).unsqueeze(-1) * dist).transpose(-1, -2) @ dist
    ).view(b, c, 2, 2)

    v2 = torch.clip(torch.sum(heatmap**2, dim=(2, 3)).unsqueeze(-1).unsqueeze(-1), min=eps)
    assert covariances.shape == (b, c, 2, 2)
    return covariances / (1 - v2)


def coord_cov_windowed_weigthed_sample_cov(
    heatmap: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the covariance matrix from a heatmap by calculating the mean of the
    heatmap values weighted by the heatmap values. The window is determined by the
    distance to the closest edge.

    Args:
        heatmap (torch.Tensor): heatmap of shape (B, C, H, W)
        coords (torch.Tensor): coordinates of shape (B, C, 2)

    Returns:
        (torch.Tensor): coordinates of shape (B, C, 2)
        (torch.Tensor): covariance matrix of shape (B, C, 2, 2)
    """
    b, c, h, w = heatmap.shape
    coords = coord_weighted_spatial_mean(heatmap)
    covs = torch.zeros((b, c, 2, 2)).to(heatmap.device)
    for b in range(b):
        for c in range(c):
            window = int(
                min(h - coords[b, c, 0], coords[b, c, 0], w - coords[b, c, 1], coords[b, c, 1])
            )
            covs[b, c] = weighted_sample_cov(
                heatmap[
                    b,
                    c,
                    coords[b, c, 0].int() - window : coords[b, c, 0].int() + window + 1,
                    coords[b, c, 1].int() - window : coords[b, c, 1].int() + window + 1,
                ]
                .unsqueeze(0)
                .unsqueeze(0),
                torch.tensor([[[window, window]]], dtype=torch.float),
            )
    return coords, covs


def heatmap_to_multiple_coord(
    heatmaps: torch.Tensor,
    window: int = 5,
    threshold: Optional[float] = None,
    method: str = "argmax",
) -> tuple[list[list[list[torch.Tensor]]], list[list[list[float]]]]:
    """
    Returns the multiple coordinates of the maximum value of the heatmap
    for each batch and channel. Additionally, the scores of the maxima are returned.
    If threshold is not None, only the maxima with a score higher than the threshold are returned.

    Args:
        heatmap (torch.Tensor): heatmap of shape (B, C, H, W)
        window (tuple): window size
        threshold (float): threshold to remove points. If None, no threshold is applied. Default
            None.
        method (str): method to retrieve the coordinates

    Returns:
        (list(list(list(torch.Tensor)))): list of coordinates of the maxima for each batch and
            channel, with the first list the batch, the second list the channel and the third list
            the coordinates
        (list(list(list(float)))): list of scores of the maxima for each batch and channel, with
            the first list the batch, the second list the channel and the third list the scores.
    """
    argmax_fun: Callable[[torch.Tensor, int], list[torch.Tensor]]
    if method == "argmax":
        argmax_fun = non_maximum_surpression
    elif method == "local_soft_argmax":
        argmax_fun = non_maximum_surpression_local_soft_argmax
    else:
        raise ValueError("Method not implemented.")
    if heatmaps.dim() == 2:
        heatmaps = heatmaps.unsqueeze(0).unsqueeze(0)
    elif heatmaps.dim() == 3:
        heatmaps = heatmaps.unsqueeze(0)
    b, c, _, _ = heatmaps.shape
    out = []
    scores = []
    for b1 in range(b):
        out_classes = []
        scores_classes = []
        for c1 in range(c):
            out_class = argmax_fun(heatmaps[b1, c1], window)
            scores_class = [heatmaps[b1, c1, x[0].int(), x[1].int()].item() for x in out_class]
            sorted_idx = sorted(
                range(len(scores_class)), key=lambda k: scores_class[k], reverse=True
            )
            out_class = [out_class[i] for i in sorted_idx]
            scores_class = [scores_class[i] for i in sorted_idx]
            if threshold is not None:
                for i in range(len(scores_class)):
                    if scores_class[i] < threshold:
                        break
                    i += 1
                scores_class = scores_class[:i]
                out_class = out_class[:i]
            out_classes.append(out_class)
            scores_classes.append(scores_class)
        out.append(out_classes)
        scores.append(scores_classes)
    return out, scores


def non_maximum_surpression(heatmap: torch.Tensor, window: int = 3) -> list[torch.Tensor]:
    """Non-Maximum Surpression (NMS)

    source: Efficient Non-Maximum Suppression - Neubeck and Van Gool (2006)

    Args:
        heatmap (torch.Tensor): heatmap of shape (H, W)
        window (int): window size

    Returns:
        (list(tuple(int, int))): list of coordinates of the maxima
    """
    # TODO: Think about way to deal with ties (in block) (multiple maxima) in the output
    #    now we just take the first one
    assert heatmap.dim() == 2, f"Heatmap must be 2D, got {heatmap.dim()}"
    set_local_argmax = set()
    for i in range(0, heatmap.shape[0] - (window - 1) + 1, window):
        for j in range(0, heatmap.shape[1] - (window - 1) + 1, window):
            local_argmax = coord_argmax(
                heatmap[
                    i : min(i + window, heatmap.shape[0]), j : min(j + window, heatmap.shape[1])
                ]
            )
            local_argmax += torch.tensor([i, j])
            local_argmax_around = coord_argmax(
                heatmap[
                    torch.clamp(local_argmax[0] - window, min=0) : torch.clamp(
                        local_argmax[0] + window + 1, max=heatmap.shape[0]
                    ),
                    torch.clamp(local_argmax[1] - window, min=0) : torch.clamp(
                        local_argmax[1] + window + 1, max=heatmap.shape[1]
                    ),
                ]
            )
            local_argmax_around += torch.clamp(local_argmax - window, min=0)
            if torch.all(local_argmax == local_argmax_around):
                set_local_argmax.add(local_argmax)
    return [torch.Tensor((x[0].item(), x[1].item())) for x in set_local_argmax]


def non_maximum_surpression_local_soft_argmax(
    heatmap: torch.Tensor, window: int = 3
) -> list[torch.Tensor]:
    """
    Returns the coordinates of the maximum value of the heatmap for each batch and channel
    (landmark), with the soft-argmax function applied on the heatmaps.

    Args:
        heatmap (torch.Tensor): heatmap of shape (H, W)
        window (int): window size

    Returns:
        (list(tuple(int, int))): list of coordinates of the maxima
    """
    set_local_argmax = non_maximum_surpression(heatmap, window=window)
    list_local_soft_argmax = []
    heatmap_pad = F.pad(heatmap, (window // 2, window // 2, window // 2, window // 2))
    for local_argmax in list(set_local_argmax):
        local_argmax_tensor = torch.tensor(
            [int(local_argmax[0]), int(local_argmax[1])]
        ) + torch.tensor([window // 2, window // 2])
        local_argmax_around = coord_soft_argmax_2d(
            heatmap_pad[
                torch.clamp(local_argmax_tensor[0] - window, min=0) : torch.clamp(
                    local_argmax_tensor[0] + window + 1, max=heatmap_pad.shape[0]
                ),
                torch.clamp(local_argmax_tensor[1] - window, min=0) : torch.clamp(
                    local_argmax_tensor[1] + window + 1, max=heatmap_pad.shape[1]
                ),
            ]
        )
        local_argmax_around += torch.clamp(local_argmax_tensor - window, min=0) - torch.tensor(
            [window // 2, window // 2]
        )
        list_local_soft_argmax.append(local_argmax_around)
    return [torch.Tensor((x[0].item(), x[1].item())) for x in list_local_soft_argmax]
