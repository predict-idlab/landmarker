"""
Decoder module for retrieving coordinates and other statistics from heatmaps.
"""

from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import optimize  # type: ignore

from landmarker.heatmap.generator import GaussianHeatmapGenerator


def coord_argmax(heatmap: torch.Tensor, spatial_dims: int = 2) -> torch.Tensor:
    """
    Returns the coordinates of the maximum value of the heatmap
    for each batch and channel (landmark).

    Args:
        heatmap (torch.Tensor): heatmap of shape (B, C, H, W) or (B, C, D, H, W)
        spatial_dims (int): number of spatial dimensions (2 or 3)

    Returns:
        (torch.Tensor): coordinates of shape (B, C, 2) or (B, C, 3)
    """
    # TODO: Think about way to deal with ties (multiple maxima) in the output
    #    now we just take the first one
    if spatial_dims == 2:
        w = heatmap.shape[-1]
        idx_pred = torch.argmax(heatmap.flatten(-2, -1), dim=-1)
        idx_pred = torch.stack((idx_pred // w, idx_pred % w), dim=-1)
    elif spatial_dims == 3:
        d, h, w = heatmap.shape[-3:]
        idx_pred = torch.argmax(heatmap.flatten(-3, -1), dim=-1)
        idx_pred = torch.stack(
            (idx_pred // (h * w), (idx_pred % (h * w)) // w, idx_pred % w), dim=-1
        )
    else:
        raise ValueError(f"Spatial dimensions must be 2 or 3: {spatial_dims}")
    return idx_pred.int()


def coord_local_soft_argmax(
    heatmap: torch.Tensor, window: int = 5, t: float = 10.0, spatial_dims: int = 2
) -> torch.Tensor:
    """
    Returns coordiantes through applying the local soft-argmax function on the heatmaps.
        Source: "Subpixel Heatmap Regression for Facial Landmark Localization" - Bulat et al. (2021)

    Args:
        heatmap (torch.Tensor): heatmap of shape (B, C, H, W) or (B, C, D, H, W)
        window (int): local window size
        t (float ): temperature that controls the resulting probability map
        spatial_dims (int): number of spatial dimensions (2 or 3)

    Returns:
        (torch.Tensor): coordinates of shape (B, C, 2)
    """
    return coord_local_weighted(
        heatmap, window=window, spatial_dims=spatial_dims, activation="softmax", t=t
    )


def coord_local_weighted(
    heatmap: torch.Tensor,
    window: int = 9,
    spatial_dims: int = 2,
    activation: Optional[str] = None,
    t: float = 1.0,
) -> torch.Tensor:
    """
    Returns coordiantes through applying the local soft-argmax function on the heatmaps.
        Source: "Subpixel Heatmap Regression for Facial Landmark Localization" - Bulat et al. (2021)

    Args:
        heatmap (torch.Tensor): heatmap of shape (B, C, H, W) or (B, C, D, H, W)
        window (int): local window size
        t (float ): temperature that controls the resulting probability map
        spatial_dims (int): number of spatial dimensions (2 or 3)

    Returns:
        (torch.Tensor): coordinates of shape (B, C, 2)
    """
    if spatial_dims == 2:
        padded_heatmap = F.pad(heatmap, (window // 2, window // 2, window // 2, window // 2))
        b, c, h, w = padded_heatmap.shape

        argmax_coords = coord_argmax(heatmap) + window // 2

        mask = torch.zeros((b, c, h, w))
        for b1 in range(b):
            for c1 in range(c):
                for i in range(-(window // 2), (window // 2) + 1):
                    for j in range(-(window // 2), (window // 2) + 1):
                        mask[b1, c1, argmax_coords[b1, c1, 0] + i, argmax_coords[b1, c1, 1] + j] = 1
        masked_output = padded_heatmap[mask > 0].view(b, c, window, window)
        local_coords = coord_weighted_spatial_mean(
            masked_output, spatial_dims=spatial_dims, activation=activation, t=t
        )
        return local_coords + (argmax_coords - (window // 2) * 2)
    elif spatial_dims == 3:
        padding = [window // 2] * (2 * spatial_dims)  # Pad all spatial dimensions
        padded_heatmap = F.pad(heatmap, padding)
        b, c, d, h, w = padded_heatmap.shape

        argmax_coords = coord_argmax(heatmap, spatial_dims=3) + window // 2

        mask = torch.zeros((b, c, d, h, w))
        for b1 in range(b):
            for c1 in range(c):
                for z in range(-(window // 2), (window // 2) + 1):
                    for i in range(-(window // 2), (window // 2) + 1):
                        for j in range(-(window // 2), (window // 2) + 1):
                            mask[
                                b1,
                                c1,
                                argmax_coords[b1, c1, 0] + z,
                                argmax_coords[b1, c1, 1] + i,
                                argmax_coords[b1, c1, 2] + j,
                            ] = 1
        masked_output = padded_heatmap[mask > 0].view(b, c, window, window, window)

        local_coords = coord_weighted_spatial_mean(
            masked_output, spatial_dims=spatial_dims, activation=activation, t=t
        )
        return local_coords + (argmax_coords - (window // 2) * 2)
    else:
        raise ValueError(f"Spatial dimensions must be 2 or 3: {spatial_dims}")


def _activate_norm_heatmap(
    heatmap: torch.Tensor,
    spatial_dims: int = 2,
    activation: Optional[str] = "softmax",
    t: float = 1.0,
) -> torch.Tensor:
    dim = (-2, -1) if spatial_dims == 2 else (-3, -2, -1)
    if activation is not None:
        if activation == "softmax":
            heatmap = torch.exp(t * heatmap)
        elif activation == "sigmoid":
            heatmap = torch.sigmoid(heatmap)
        elif activation == "ReLU":
            heatmap = F.relu(heatmap)
        else:
            raise ValueError(f"Activation function {activation} not implemented.")
    return heatmap / torch.sum(heatmap, dim=dim, keepdim=True)


def coord_weighted_spatial_mean(
    heatmap: torch.Tensor,
    spatial_dims: int = 2,
    activation: Optional[str] = None,
    require_grad: bool = False,
    t: float = 1.0,
) -> torch.Tensor:
    """
    Returns the spatial weighted mean of the heatmap.
    Source: "UGLLI Face Alignment: Estimating Uncertainty with
        Gaussian Log-Likelihood Loss" - Kumar et al. (2019)

    Args:
        heatmap (torch.Tensor): heatmap of shape (B, C, H, W) or (B, C, D, H, W)
        spatial_dims (int): number of spatial dimensions (2 or 3)
        activation (str): activation function to apply to the heatmap
        require_grad (bool): whether to require gradient for the coordinates

    Returns:
        (torch.Tensor): coordinates of shape (B, C, 2) or (B, C, 3)
    """
    heatmap = _activate_norm_heatmap(heatmap, spatial_dims=spatial_dims, activation=activation, t=t)
    if spatial_dims == 2:
        b, c, h, w = heatmap.shape
        xs = torch.arange(0, w, dtype=torch.float32, requires_grad=require_grad).to(heatmap.device)
        ys = torch.arange(0, h, dtype=torch.float32, requires_grad=require_grad).to(heatmap.device)
        xy = (
            torch.stack(torch.meshgrid(xs, ys, indexing="xy"), dim=2)
            .view(1, 1, h, w, 2)
            .repeat(b, c, 1, 1, 1)
        )
        return torch.sum(heatmap.unsqueeze(4) * xy, dim=(2, 3)).flip(-1)
    elif spatial_dims == 3:
        b, c, d, h, w = heatmap.shape
        xs = torch.arange(0, w, dtype=torch.float32, requires_grad=require_grad).to(heatmap.device)
        ys = torch.arange(0, h, dtype=torch.float32, requires_grad=require_grad).to(heatmap.device)
        zs = torch.arange(0, d, dtype=torch.float32, requires_grad=require_grad).to(heatmap.device)
        xyz = (
            torch.stack(torch.meshgrid(zs, ys, xs, indexing="ij"), dim=3)
            .view(1, 1, d, h, w, 3)
            .repeat(b, c, 1, 1, 1, 1)
        )
        return torch.sum(heatmap.unsqueeze(5) * xyz, dim=(2, 3, 4))
    else:
        raise ValueError(f"Spatial dimensions must be 2 or 3: {spatial_dims}")


def coord_soft_argmax_2d(
    heatmap: torch.Tensor, logit_scale: bool = False, require_grad: bool = False
) -> torch.Tensor:
    """
    Returns the spatial mean over the softmax distribution of the heatmap,
    but for a 2D heatmap, without the batch and channel dimensions.

    Args:
        heatmap (torch.Tensor): heatmap of shape (H, W)
        logit_scale (bool): whether the input is logit scaled
        require_grad (bool): whether to require gradient for the coordinates
    """
    activation = "softmax" if not logit_scale else None
    return coord_weighted_spatial_mean(
        heatmap.unsqueeze(0).unsqueeze(0),
        spatial_dims=2,
        activation=activation,
        require_grad=require_grad,
    )[0, 0]


def heatmap_to_coord(
    heatmap: torch.Tensor,
    offset_coords: int = 0,
    method: str = "argmax",
    spatial_dims: int = 2,
    require_grad: bool = False,
) -> torch.Tensor:
    """
    Returns the retrieved coordinates via specified method from a heatmap. The offset_coords is used
    to remove the first offset_coords coordinates from the heatmap. This is used to remove the
    background class (if present).

    Args:
        heatmap (torch.Tensor): heatmap of shape (B, C, H, W) or (B, C, D, H, W)
        offset_coords (int): number of coordinates to remove
        method (str): method to retrieve the coordinates
        spatial_dims (int): number of spatial dimensions (2 or 3)
        require_grad (bool): whether to require gradient for the coordinates

    Returns:
        (torch.Tensor): coordinates of shape (B, C, 2) or (B, C, 3)
    """
    heatmap = heatmap[:, offset_coords:]
    if method == "argmax":
        return coord_argmax(heatmap, spatial_dims=spatial_dims)
    if method == "local_soft_argmax":
        return coord_local_soft_argmax(heatmap, spatial_dims=spatial_dims)
    if method == "weighted_spatial_mean_relu":
        return coord_weighted_spatial_mean(
            heatmap, spatial_dims=spatial_dims, activation="ReLU", require_grad=require_grad
        )
    if method == "soft_argmax":
        return coord_weighted_spatial_mean(
            heatmap, spatial_dims=spatial_dims, activation="softmax", require_grad=require_grad
        )
    if method == "weighted_spatial_mean":
        return coord_weighted_spatial_mean(
            heatmap, spatial_dims=spatial_dims, require_grad=require_grad
        )
    if method == "weighted_spatial_mean_sigmoid":
        return coord_weighted_spatial_mean(
            heatmap, spatial_dims=spatial_dims, activation="sigmoid", require_grad=require_grad
        )
    raise ValueError("Method not implemented.")


def heatmap_to_coord_enlarge(
    heatmap: torch.Tensor,
    offset_coords: int = 0,
    method: str = "argmax",
    enlarge_factor: int = 1,
    enlarge_dim: Optional[tuple[int, ...]] = None,
    enlarge_mode: str = "bilinear",
    spatial_dims: int = 2,
) -> torch.Tensor:
    """
    Returns the retrieved coordinates via specified method from an enlarged heatmap.
    The offset_coords is used to remove the first offset_coords coordinates from the heatmap.
    This is used to remove the background class (if present).

    Args:
        heatmap (torch.Tensor): heatmap of shape (B, C, H, W) or (B, C, D, H, W)
        offset_coords (int): number of coordinates to remove
        method (str): method to retrieve the coordinates
        enlarge_factor (int): factor to enlarge the heatmap
        enlarge_dim (tuple[int, ...] or None): dimensions to enlarge the heatmap to
        enlarge_mode (str): interpolation mode to enlarge the heatmap
        spatial_dims (int): number of spatial dimensions (2 or 3)

    Returns:
        (torch.Tensor): coordinates of shape (B, C, 2)
    """
    heatmap = heatmap[:, offset_coords:]
    if spatial_dims == 2:
        if enlarge_dim is not None:
            assert len(enlarge_dim) == 2, "enlarge_dim must be a tuple of 2 elements"
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
                / torch.tensor(heatmap.shape[-2:], dtype=torch.float)
                .view((1, 1, 2))
                .to(heatmap.device)
            )
        else:
            coords = coords / enlarge_factor
        return coords
    elif spatial_dims == 3:
        if enlarge_dim is not None:
            assert len(enlarge_dim) == 3, "enlarge_dim must be a tuple of 3 elements"
            heatmap_scaled = F.interpolate(
                heatmap, size=enlarge_dim, mode=enlarge_mode, align_corners=False
            )
        else:
            heatmap_scaled = F.interpolate(
                heatmap, scale_factor=enlarge_factor, mode=enlarge_mode, align_corners=False
            )
        coords = heatmap_to_coord(heatmap_scaled, method=method, spatial_dims=3)
        if enlarge_dim is not None:
            coords = coords / (
                torch.tensor(enlarge_dim, dtype=torch.float).view((1, 1, 3)).to(heatmap.device)
                / torch.tensor(heatmap.shape[-3:], dtype=torch.float)
                .view((1, 1, 3))
                .to(heatmap.device)
            )
        else:
            coords = coords / enlarge_factor
        return coords
    else:
        raise ValueError(f"Spatial dimensions must be 2 or 3: {spatial_dims}")


def coord_weighted_spatial_mean_cov(
    heatmap: torch.Tensor,
    spatial_dims: int = 2,
    require_grad: bool = False,
    activation: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the spatial weighted mean and the weighted sample covariance of the possitive elements
    of the heatmap by the heatmap values.
        source: https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_covariance

    Args:
        heatmap (torch.Tensor): heatmap of shape (B, C, H, W) or (B, C, D, H, W)
        spatial_dims (int): number of spatial dimensions (2 or 3)
        require_grad (bool): whether to require gradient for the coordinates
        activation (str): activation function to apply to the heatmap

    Returns:
        (torch.Tensor): coordinates of shape (B, C, 2) or (B, C, 3)
        (torch.Tensor): covariance matrix of shape (B, C, 2, 2) or (B, C, 3, 3)
    """
    heatmap = _activate_norm_heatmap(heatmap, spatial_dims=spatial_dims, activation=activation)
    if spatial_dims == 2:
        b, c, h, w = heatmap.shape
        xs = torch.arange(0, w, dtype=torch.float32, requires_grad=require_grad).to(heatmap.device)
        ys = torch.arange(0, h, dtype=torch.float32, requires_grad=require_grad).to(heatmap.device)
        xs, ys = torch.meshgrid(xs, ys, indexing="xy")
        yx = torch.stack((ys, xs), dim=2).view(1, 1, h, w, 2).repeat(b, c, 1, 1, 1)
        mean_coords = torch.sum(heatmap.unsqueeze(4) * yx, dim=(2, 3))
        dist_x = xs - mean_coords[:, :, 1].unsqueeze(-1).unsqueeze(-1)
        dist_y = ys - mean_coords[:, :, 0].unsqueeze(-1).unsqueeze(-1)
        dist = torch.stack(
            (dist_y.flatten(start_dim=-2, end_dim=-1), dist_x.flatten(start_dim=-2, end_dim=-1)),
            dim=-1,
        )  # (B, C, H*W, 2)

        covariances = (
            (heatmap.flatten(start_dim=-2, end_dim=-1).unsqueeze(-1) * dist).transpose(-1, -2)
            @ dist
        ).view(b, c, 2, 2)

        v2 = torch.sum(heatmap**2, dim=(2, 3)).unsqueeze(-1).unsqueeze(-1)
        return mean_coords, covariances / (1 - v2)
    elif spatial_dims == 3:
        b, c, d, h, w = heatmap.shape
        xs = torch.arange(0, w, dtype=torch.float32, requires_grad=require_grad).to(heatmap.device)
        ys = torch.arange(0, h, dtype=torch.float32, requires_grad=require_grad).to(heatmap.device)
        zs = torch.arange(0, d, dtype=torch.float32, requires_grad=require_grad).to(heatmap.device)
        zs, ys, xs = torch.meshgrid(zs, ys, xs, indexing="ij")
        zyx = torch.stack((zs, ys, xs), dim=3).view(1, 1, d, h, w, 3).repeat(b, c, 1, 1, 1, 1)

        mean_coords = torch.sum(heatmap.unsqueeze(5) * zyx, dim=(2, 3, 4))

        dist_x = xs - mean_coords[:, :, 2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        dist_y = ys - mean_coords[:, :, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        dist_z = zs - mean_coords[:, :, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        dist = torch.stack(
            (
                dist_z.flatten(start_dim=-3, end_dim=-1),
                dist_y.flatten(start_dim=-3, end_dim=-1),
                dist_x.flatten(start_dim=-3, end_dim=-1),
            ),
            dim=-1,
        )

        covariances = (
            (heatmap.flatten(start_dim=-3, end_dim=-1).unsqueeze(-1) * dist).transpose(-1, -2)
            @ dist
        ).view(b, c, 3, 3)

        v2 = torch.sum(heatmap**2, dim=(2, 3, 4)).unsqueeze(-1).unsqueeze(-1)

        return mean_coords, covariances / (1 - v2)
    else:
        raise ValueError(f"Spatial dimensions must be 2 or 3: {spatial_dims}")


def heatmap_to_coord_cov(
    heatmap: torch.Tensor,
    method: str = "soft_argmax",
    require_grad: bool = True,
    spatial_dims: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the modal coordinates and covariance matrix from a heatmap.

    Args:
        heatmap (torch.Tensor): heatmap of shape (B, C, H, W) or (B, C, D, H, W)
        method (str): method to retrieve the coordinates
        require_grad (bool): whether to require gradient for the coordinates
        spatial_dims (int): number of spatial dimensions (2 or 3)

    Returns:
        (torch.Tensor): coordinates of shape (B, C, 2) or (B, C, 3)
        (torch.Tensor): covariance matrix of shape (B, C, 2, 2) or (B, C, 3, 3)
    """
    if method == "soft_argmax":
        return coord_weighted_spatial_mean_cov(
            heatmap, spatial_dims=spatial_dims, require_grad=require_grad, activation="softmax"
        )
    if method == "weighted_spatial_mean":
        return coord_weighted_spatial_mean_cov(
            heatmap, spatial_dims=spatial_dims, require_grad=require_grad
        )
    if method == "weighted_spatial_mean_sigmoid":
        return coord_weighted_spatial_mean_cov(
            heatmap, spatial_dims=spatial_dims, require_grad=require_grad, activation="sigmoid"
        )
    if method == "weighted_spatial_mean_relu":
        return coord_weighted_spatial_mean_cov(
            heatmap, spatial_dims=spatial_dims, require_grad=require_grad, activation="ReLU"
        )
    raise ValueError("Method not implemented.")


def coord_cov_from_gaussian_ls(
    heatmap: torch.Tensor,
    gamma: Optional[float] = None,
    ls_library: str = "scipy",
    spatial_dims: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the modal coordinates and covariance matrix from a heatmap through fitting the heatmap
    on Gaussian distribution with a specicic scaling factor gamma with help of least squares
    optimization.

    Args:
        heatmap (torch.Tensor): heatmap of shape (B, C, H, W)
        gamma (float): gamma parameter of the gaussian heatmap generator
        ls_library (str): library to use for least squares optimization. (scipy or pytorch)
        spatial_dims (int): number of spatial dimensions (2 or 3)

    Returns:
        (torch.Tensor): coordinates of shape (B, C, 2)
        (torch.Tensor): covariance matrix of shape (B, C, 2, 2)
    """
    # TODO: see if we can use a pytorch implementation (e.g, pytorch-minimize seems to be broken)
    if spatial_dims != 2:
        raise ValueError(f"Spatial dimensions must be 2: {spatial_dims}")
    if ls_library == "scipy":
        return coord_cov_from_gaussian_ls_scipy(heatmap, gamma=gamma)
    raise ValueError("Method not implemented.")


def coord_cov_from_gaussian_ls_scipy(
    heatmap: torch.Tensor, gamma: Optional[float] = None, spatial_dims: int = 2
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the modal coordinates and covariance matrix from a heatmap through fitting the heatmap
    on Gaussian distribution with a specicic scaling factor gamma with help of least squares
    optimization.

    Args:
        heatmap (torch.Tensor): heatmap of shape (B, C, H, W)
        gamma (float): gamma parameter of the gaussian heatmap generator
        spatial_dims (int): number of spatial dimensions (2 or 3)

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
            learnable=False,
        ).to(heatmap.device)

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

    if spatial_dims != 2:
        raise ValueError(f"Spatial dimensions must be 2: {spatial_dims}")
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
    spatial_dims: int = 2,
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
    if spatial_dims != 2:
        raise ValueError(f"Spatial dimensions must be 2: {spatial_dims}")
    if ls_library == "scipy":
        return cov_from_gaussian_ls_scipy(heatmap, coords, gamma=gamma)
    raise ValueError("Method not implemented.")


def cov_from_gaussian_ls_scipy(
    heatmap: torch.Tensor,
    coords: torch.Tensor,
    gamma: Optional[float] = None,
    spatial_dims: int = 2,
) -> torch.Tensor:
    """
    Returns the covariance matrix from a heatmap through fitting the heatmap on Gaussian
    distribution with a specicic scaling factor gamma and retrieved coordinates with specified
    decoding method with help of least squares optimization.

    Args:
        heatmap (torch.Tensor): heatmap of shape (B, C, H, W)
        coords (torch.Tensor): coordinates of shape (B, C, 2)
        gamma (float): gamma parameter of the gaussian heatmap generator
        spatial_dims (int): number of spatial dimensions (2 or 3)

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
            learnable=False,
        ).to(heatmap.device)

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

    if spatial_dims != 2:
        raise ValueError(f"Spatial dimensions must be 2: {spatial_dims}")
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


def weighted_sample_cov(
    heatmap: torch.Tensor,
    coords: torch.Tensor,
    spatial_dims: int = 2,
    activation: Optional[str] = None,
) -> torch.Tensor:
    """
    Calculate the covariance matrix from a heatmap by calculating the mean of the
    heatmap values weighted by the heatmap values.
    source: https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_covariance

    Args:
        heatmap (torch.Tensor): heatmap of shape (B, C, H, W) or (B, C, D, H, W)
        coords (torch.Tensor): coordinates of shape (B, C, 2) or (B, C, 3)
        spatial_dims (int): number of spatial dimensions (2 or 3)
        activation (str): activation function to apply to the heatmap

    Returns:
        (torch.Tensor): covariance matrix of shape (B, C, 2, 2) or (B, C, 3, 3)
    """
    heatmap = _activate_norm_heatmap(heatmap, spatial_dims=spatial_dims, activation=activation)
    if spatial_dims == 2:
        assert coords.shape[-1] == 2, f"Coordinates must have 2 elements: {coords.shape[-1]}"
        b, c, h, w = heatmap.shape
        xs = torch.arange(0, w, dtype=torch.float32, requires_grad=False).to(heatmap.device)
        ys = torch.arange(0, h, dtype=torch.float32, requires_grad=False).to(heatmap.device)
        xs, ys = torch.meshgrid(xs, ys, indexing="xy")

        dist_x = xs - coords[:, :, 1].unsqueeze(-1).unsqueeze(-1)
        dist_y = ys - coords[:, :, 0].unsqueeze(-1).unsqueeze(-1)
        dist = torch.stack(
            (dist_y.flatten(start_dim=-2, end_dim=-1), dist_x.flatten(start_dim=-2, end_dim=-1)),
            dim=-1,
        )  # (B, C, H*W, 2)

        covariances = (
            (heatmap.flatten(start_dim=-2, end_dim=-1).unsqueeze(-1) * dist).transpose(-1, -2)
            @ dist
        ).view(b, c, 2, 2)

        v2 = torch.sum(heatmap**2, dim=(2, 3)).unsqueeze(-1).unsqueeze(-1)
        assert covariances.shape == (b, c, 2, 2)
        return covariances / (1 - v2)
    elif spatial_dims == 3:
        assert coords.shape[-1] == 3, f"Coordinates must have 3 elements: {coords.shape[-1]}"
        b, c, d, h, w = heatmap.shape
        xs = torch.arange(0, w, dtype=torch.float32, requires_grad=False).to(heatmap.device)
        ys = torch.arange(0, h, dtype=torch.float32, requires_grad=False).to(heatmap.device)
        zs = torch.arange(0, d, dtype=torch.float32, requires_grad=False).to(heatmap.device)
        zs, ys, xs = torch.meshgrid(zs, ys, xs, indexing="ij")

        dist_x = xs - coords[:, :, 2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        dist_y = ys - coords[:, :, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        dist_z = zs - coords[:, :, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        dist = torch.stack(
            (
                dist_z.flatten(start_dim=-3, end_dim=-1),
                dist_y.flatten(start_dim=-3, end_dim=-1),
                dist_x.flatten(start_dim=-3, end_dim=-1),
            ),
            dim=-1,
        )

        covariances = (
            (heatmap.flatten(start_dim=-3, end_dim=-1).unsqueeze(-1) * dist).transpose(-1, -2)
            @ dist
        ).view(b, c, 3, 3)

        v2 = torch.sum(heatmap**2, dim=(2, 3, 4)).unsqueeze(-1).unsqueeze(-1)
        assert covariances.shape == (b, c, 3, 3)
        return covariances / (1 - v2)
    else:
        raise ValueError(f"Spatial dimensions must be 2 or 3: {spatial_dims}")


def windowed_weigthed_sample_cov(
    heatmap: torch.Tensor,
    coords: torch.Tensor,
    spatial_dims: int = 2,
    activation: Optional[str] = None,
) -> torch.Tensor:
    """
    Calculate the covariance matrix from a heatmap by calculating the mean of the
    heatmap values weighted by the heatmap values. The window is determined by the
    distance to the closest edge.

    Args:
        heatmap (torch.Tensor): heatmap of shape (B, C, H, W)
        coords (torch.Tensor): coordinates of shape (B, C, 2)
        spatial_dims (int): number of spatial dimensions (2 or 3)
        activation (str): activation function to apply to the heatmap

    Returns:
        (torch.Tensor): covariance matrix of shape (B, C, 2, 2)
    """
    if spatial_dims != 2:
        raise ValueError(f"Spatial dimensions must be 2: {spatial_dims}")
    b, c, h, w = heatmap.shape
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
                torch.tensor([[[window, window]]], dtype=torch.float).to(heatmap.device),
                spatial_dims=spatial_dims,
                activation=activation,
            )
    return covs


def heatmap_to_multiple_coord(
    heatmaps: torch.Tensor,
    window: int = 5,
    threshold: Optional[float] = None,
    method: str = "argmax",
    spatial_dims: int = 2,
) -> Tuple[List[List[List[torch.Tensor]]], List[List[List[float]]]]:

    # Original input shape handling
    if spatial_dims != 2:
        raise ValueError(f"Spatial dimensions must be 2: {spatial_dims}")

    # Dimension handling identical to original
    orig_dim = heatmaps.dim()
    if orig_dim == 2:
        heatmaps = heatmaps.unsqueeze(0).unsqueeze(0)
    elif orig_dim == 3:
        heatmaps = heatmaps.unsqueeze(0)
    B, C, H, W = heatmaps.shape

    # Main processing
    if method == "argmax":
        coords, scores = non_maximum_surpression(heatmaps, window)
    else:
        raise ValueError("Method not implemented.")

    # Post-processing identical to original
    return process_multiple_coord_results(coords, scores, threshold, B, C, orig_dim)


def process_multiple_coord_results(coords, scores, threshold, B, C, orig_dim):
    # Identical sorting and threshold logic to original
    out = []
    score_out = []

    for b in range(B):
        batch_coords = []
        batch_scores = []
        for c in range(C):
            # Sort descending by score
            sc = scores[b][c]
            idx = sorted(range(len(sc)), key=lambda k: sc[k], reverse=True)
            sorted_coords = [coords[b][c][i] for i in idx]
            sorted_scores = [sc[i] for i in idx]

            # Apply threshold
            if threshold is not None:
                cutoff = next(
                    (i for i, s in enumerate(sorted_scores) if s < threshold), len(sorted_scores)
                )
                sorted_coords = sorted_coords[:cutoff]
                sorted_scores = sorted_scores[:cutoff]

            batch_coords.append(sorted_coords)
            batch_scores.append(sorted_scores)

        out.append(batch_coords)
        score_out.append(batch_scores)

    # Match original output shapes
    if orig_dim == 2:
        return out[0][0], score_out[0][0]
    elif orig_dim == 3:
        return out[0], score_out[0]
    return out, score_out


def non_maximum_surpression(
    heatmaps: torch.Tensor, window: int
) -> Tuple[List[List[List[torch.Tensor]]], List[List[List[float]]]]:

    B, C, H, W = heatmaps.shape
    device = heatmaps.device
    pad = window // 2
    padded = F.pad(heatmaps, (pad, pad, pad, pad), mode="constant", value=-float("inf"))

    # Step 1: Find initial candidates using original's block strategy
    grid_y = torch.arange(0, H, window, device=device)
    grid_x = torch.arange(0, W, window, device=device)
    gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    candidates = torch.stack([gy.flatten(), gx.flatten()], dim=1)  # [K, 2]

    # Step 2: Find local maxima within each block
    max_vals = torch.zeros(B, C, len(candidates), device=device)
    max_pos = torch.zeros(B, C, len(candidates), 2, dtype=torch.long, device=device)

    for i, (y, x) in enumerate(candidates):
        block = padded[:, :, y : y + window, x : x + window]
        vals, idx = block.flatten(2).max(dim=-1)  # [B, C]
        max_vals[:, :, i] = vals
        max_pos[:, :, i] = torch.stack([y + idx // window, x + idx % window], dim=-1)

    # Step 3: Verify maxima in their neighborhoods (original's 2nd check)
    valid = torch.zeros_like(max_vals, dtype=torch.bool)
    for i in range(len(candidates)):
        y, x = max_pos[:, :, i, 0], max_pos[:, :, i, 1]

        # Get neighborhood bounds (matching original's clamp logic)
        y0 = torch.clamp(y - pad, 0)
        y1 = torch.clamp(y + pad + 1, max=padded.shape[2])
        x0 = torch.clamp(x - pad, 0)
        x1 = torch.clamp(x + pad + 1, max=padded.shape[3])

        # Check if center is max in neighborhood
        for b in range(B):
            for c in range(C):
                neighborhood = padded[b, c, y0[b, c] : y1[b, c], x0[b, c] : x1[b, c]]
                if neighborhood.numel() > 0:
                    valid[b, c, i] = max_vals[b, c, i] >= neighborhood.max()

    # Step 4: Collect valid coordinates and deduplicate
    coords = [[[] for _ in range(C)] for _ in range(B)]  # type: ignore
    scores = [[[] for _ in range(C)] for _ in range(B)]  # type: ignore

    for b in range(B):
        for c in range(C):
            mask = valid[b, c]
            pos = max_pos[b, c][mask]
            vals = max_vals[b, c][mask]

            # Deduplicate using unique coordinates
            unique_coords, inverse = torch.unique(pos, dim=0, return_inverse=True)
            unique_vals = torch.zeros_like(unique_coords[:, 0], dtype=vals.dtype)
            unique_vals.scatter_reduce_(0, inverse, vals, reduce="amax", include_self=False)

            # Convert to original image coordinates
            final_coords = unique_coords.float() - pad  # Remove padding offset
            coords[b][c] = [final_coords[i] for i in range(len(final_coords))]
            scores[b][c] = unique_vals.tolist()

    return coords, scores
