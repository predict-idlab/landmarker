"""Heatmap generator"""

from abc import abstractmethod
from typing import Optional

import numpy as np
import torch
from torch import nn


class HeatmapGenerator(nn.Module):
    """
    Heatmap generator abstract class for generating heatmaps from landmarks

    Args:
        nb_landmarks (int): number of landmarks
        sigmas (float or list[float] or torch.Tensor or np.ndarray): sigmas of the heatmap function
        gamma (float or None): scaling factor of the heatmap function
        rotation (float or list[float] or torch.Tensor or np.ndarray): rotation of the heatmap
            function
        heatmap_size (tuple[int, ...]): size of the heatmap
        learnable (bool): whether the sigmas and rotation are learnable
        background (bool): whether to add a background channel to the heatmap
        all_points (bool): whether to add a channel with the sum of all the landmarks
        continuous (bool): whether to use continuous or discrete landmarks
        na_zero (bool): whether to set the value of the landmarks to zero if they are not available
    """

    def __init__(
        self,
        nb_landmarks: int,
        sigmas: float | list[float] | torch.Tensor | np.ndarray = 1.0,
        gamma: Optional[float] = None,
        rotation: float | list[float] | torch.Tensor | np.ndarray = 0,
        heatmap_size: tuple[int, ...] = (512, 512),
        learnable: bool = False,
        background: bool = False,
        all_points: bool = False,
        continuous: bool = True,
        na_zero: bool = False,
    ) -> None:
        super(HeatmapGenerator, self).__init__()
        self.nb_landmarks = nb_landmarks
        self.learnable = learnable
        self.heatmap_size = heatmap_size
        self.spatial_dims = len(heatmap_size)
        self.set_sigmas(sigmas)
        self.set_rotation(rotation)
        self.gamma = gamma
        self.background = background
        self.all_points = all_points
        self.continuous = continuous
        self.na_zero = na_zero

    def set_sigmas(self, sigmas: float | list[float] | torch.Tensor | np.ndarray) -> None:
        """
        Set the sigmas of the heatmap function.

        Args:
            sigmas (float or list[float] or torch.Tensor or np.ndarray): sigmas of the heatmap
                function
        """
        if isinstance(sigmas, torch.Tensor):
            device = sigmas.device
        else:
            device = torch.device("cpu")
        sigmas = torch.tensor(
            sigmas, device=device, dtype=torch.float, requires_grad=self.learnable
        )
        sigmas = (
            torch.ones(
                (self.nb_landmarks, self.spatial_dims),
                requires_grad=self.learnable,
                device=device,
                dtype=torch.float,
            )
            * sigmas
        )
        if self.learnable:
            self.sigmas = torch.nn.Parameter(sigmas)
        else:
            self.register_buffer("sigmas", sigmas)

    def set_rotation(self, rotation: float | list[float] | torch.Tensor | np.ndarray) -> None:
        """
        Set the rotation of the heatmap function.

        Args:
            rotation (float or list[float] or torch.Tensor or np.ndarray): rotation of the heatmap
                function
        """
        if hasattr(self, "sigmas"):
            device = self.sigmas.device
        elif isinstance(rotation, torch.Tensor):
            device = rotation.device
        else:
            device = torch.device("cpu")
        rotation = torch.tensor(
            rotation, device=device, dtype=torch.float, requires_grad=self.learnable
        )
        if self.spatial_dims == 2:
            rotation = (
                torch.ones(
                    (self.nb_landmarks,),
                    requires_grad=self.learnable,
                    device=device,
                    dtype=torch.float,
                )
                * rotation
            )
        else:
            rotation = (
                torch.ones(
                    (self.nb_landmarks, self.spatial_dims),
                    requires_grad=self.learnable,
                    device=device,
                    dtype=torch.float,
                )
                * rotation
            )
        if self.learnable:
            self.rotation = torch.nn.Parameter(rotation)
        else:
            self.register_buffer("rotation", rotation)

    def __call__(
        self, landmarks: torch.Tensor, affine_matrix: torch.Tensor = torch.eye(4)
    ) -> torch.Tensor:
        affine_matrix = affine_matrix.to(landmarks.device)
        assert affine_matrix.shape[-1] == affine_matrix.shape[-2]
        if len(affine_matrix.shape) == 2:
            affine_matrix = affine_matrix.unsqueeze(0)
        if affine_matrix.shape[-1] == 2:
            # go from 2 by 2 affine matrix to 4 by 4
            affine_matrix = from_2by2_to_4by4(affine_matrix)
        elif affine_matrix.shape[-1] == 3:
            # go from 3 by 3 affine matrix to 4 by 4
            affine_matrix = from_3by3_to_4by4(affine_matrix)
        affine_matrix = affine_matrix.unsqueeze(1)
        heatmaps = torch.zeros(
            (landmarks.shape[0], landmarks.shape[1], *self.heatmap_size), device=self.sigmas.device
        )
        if len(landmarks.shape) == 2:
            heatmaps = self.create_heatmap(landmarks.unsqueeze(1), self.gamma, affine_matrix)
        elif len(landmarks.shape) == 3:
            heatmaps = self.create_heatmap(landmarks, self.gamma, affine_matrix)
        else:
            heatmaps = self.create_heatmap(landmarks, self.gamma, affine_matrix).nansum(dim=2)
        if self.all_points:
            heatmaps = torch.cat((heatmaps.sum(dim=1, keepdim=True), heatmaps), 1)
        elif self.background:
            heatmaps = torch.cat(
                (
                    torch.ones(
                        (heatmaps.shape[0], 1, *self.heatmap_size), device=self.sigmas.device
                    )
                    - heatmaps.sum(dim=1, keepdim=True),
                    heatmaps,
                ),
                1,
            )
        if self.na_zero:
            heatmaps[torch.isnan(heatmaps)] = 0
        return heatmaps

    @abstractmethod
    def heatmap_fun(
        self,
        landmark_t: torch.Tensor,
        coords: torch.Tensor,
        covariance: torch.Tensor,
        gamma: Optional[float],
    ):
        """Abstract heatmap function

        Args:
            landmark_t (torch.Tensor): coordinates of the landmark (y, x) or (y, x, z)
            coords (torch.Tensor): coordinates of the pixel (y, x) or (y, x, z)
            covariance (torch.Tensor): covariance matrix
            gamma (float or None): scaling factor of the heatmap function
        """

    def get_covariance_matrix(self, return4by4: bool = False) -> torch.Tensor:
        """
        Get the covariance matrix of the heatmap function.

        Args:
            return4by4 (bool): whether to return a 4 by 4 covariance matrix or a spatial_dims by
                spatial_dims covariance matrix.

        Returns:
            torch.Tensor: covariance matrix
        """
        if self.spatial_dims == 2:
            rotation = torch.stack(
                (
                    torch.stack((torch.cos(self.rotation), -torch.sin(self.rotation)), dim=-1),
                    torch.stack((torch.sin(self.rotation), torch.cos(self.rotation)), dim=-1),
                ),
                dim=-2,
            )

            diagonal = torch.diag_embed((self.sigmas**2))
            if return4by4:
                rotation = from_2by2_to_4by4(rotation)
                diagonal = from_2by2_to_4by4(diagonal)
        else:  # 3D case
            # see: https://msl.cs.uiuc.edu/planning/node102.html
            rotation_yaw = torch.stack(
                (
                    torch.stack(
                        (
                            torch.cos(self.rotation[..., 0]),
                            -torch.sin(self.rotation[..., 0]),
                            torch.zeros_like(self.rotation[..., 0]),
                        ),
                        dim=-1,
                    ),
                    torch.stack(
                        (
                            torch.sin(self.rotation[..., 0]),
                            torch.cos(self.rotation[..., 0]),
                            torch.zeros_like(self.rotation[..., 0]),
                        ),
                        dim=-1,
                    ),
                    torch.stack(
                        (
                            torch.zeros_like(self.rotation[..., 0]),
                            torch.zeros_like(self.rotation[..., 0]),
                            torch.ones_like(self.rotation[..., 0]),
                        ),
                        dim=-1,
                    ),
                ),
                dim=-2,
            )
            rotation_pitch = torch.stack(
                (
                    torch.stack(
                        (
                            torch.cos(self.rotation[..., 1]),
                            torch.zeros_like(self.rotation[..., 1]),
                            torch.sin(self.rotation[..., 1]),
                        ),
                        dim=-1,
                    ),
                    torch.stack(
                        (
                            torch.zeros_like(self.rotation[..., 1]),
                            torch.ones_like(self.rotation[..., 1]),
                            torch.zeros_like(self.rotation[..., 1]),
                        ),
                        dim=-1,
                    ),
                    torch.stack(
                        (
                            -torch.sin(self.rotation[..., 1]),
                            torch.zeros_like(self.rotation[..., 1]),
                            torch.cos(self.rotation[..., 1]),
                        ),
                        dim=-1,
                    ),
                ),
                dim=-2,
            )
            rotation_roll = torch.stack(
                (
                    torch.stack(
                        (
                            torch.ones_like(self.rotation[..., 2]),
                            torch.zeros_like(self.rotation[..., 2]),
                            torch.zeros_like(self.rotation[..., 2]),
                        ),
                        dim=-1,
                    ),
                    torch.stack(
                        (
                            torch.zeros_like(self.rotation[..., 2]),
                            torch.cos(self.rotation[..., 2]),
                            -torch.sin(self.rotation[..., 2]),
                        ),
                        dim=-1,
                    ),
                    torch.stack(
                        (
                            torch.zeros_like(self.rotation[..., 2]),
                            torch.sin(self.rotation[..., 2]),
                            torch.cos(self.rotation[..., 2]),
                        ),
                        dim=-1,
                    ),
                ),
                dim=-2,
            )
            rotation = rotation_yaw @ rotation_pitch @ rotation_roll

            diagonal = torch.diag_embed((self.sigmas**2))
            if return4by4:
                rotation = from_3by3_to_4by4(rotation)
                diagonal = from_3by3_to_4by4(diagonal)
        covariance = rotation @ diagonal @ rotation.transpose(-2, -1)
        return covariance

    def create_heatmap(
        self, landmarks: torch.Tensor, gamma: Optional[float], affine_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Create a heatmap for a given landmark in an image returns a heatmap with the same size as
        the image. Works with batches and multiple landmarks.

        Args:
            landmarks (torch.Tensor): landmarks of shape (B, C, M, S) or (B, C, S)
            gamma (float or None): scaling factor of the heatmap function
            affine_matrix (torch.Tensor): affine matrix of shape (B, 1, 4, 4)

        Returns:
            torch.Tensor: heatmap of shape (B, C, M, H, W,) or (B, C, H, W) if landmarks are 2D
                else (B, C, M, D, H, W) or (B, C, D, H, W) if landmarks are 3D
        """
        covariance = self.get_covariance_matrix(return4by4=True)
        covariance = affine_matrix @ covariance @ affine_matrix.transpose(-2, -1)

        if self.spatial_dims == 2:
            x = landmarks[..., 1]
            y = landmarks[..., 0]
            x_round = torch.round(x).int()
            y_round = torch.round(y).int()
            xs = torch.arange(
                0, self.heatmap_size[1], 1, dtype=torch.float32, device=self.sigmas.device
            )
            ys = torch.arange(
                0, self.heatmap_size[0], 1, dtype=torch.float32, device=self.sigmas.device
            )
            xs, ys = torch.meshgrid(xs, ys, indexing="xy")
            pre_shape = tuple(1 for _ in range(len(landmarks.shape[:-1])))
            xs = xs.view(*pre_shape, *xs.shape).repeat(*landmarks.shape[:-1], 1, 1)
            ys = ys.view(*pre_shape, *ys.shape).repeat(*landmarks.shape[:-1], 1, 1)
            if self.continuous:
                x_t, y_t = x, y
            else:
                x_t, y_t = x_round, y_round
            x_t = x_t.view(*landmarks.shape[:-1], 1, 1)
            y_t = y_t.view(*landmarks.shape[:-1], 1, 1)
            heatmap = self.heatmap_fun(
                torch.stack((y_t, x_t), -1), torch.stack((ys, xs), -1), covariance, gamma
            )
        else:
            z = landmarks[..., 0]
            y = landmarks[..., 1]
            x = landmarks[..., 2]
            z_round = torch.round(z).int()
            y_round = torch.round(y).int()
            x_round = torch.round(x).int()
            zs = torch.arange(
                0, self.heatmap_size[0], 1, dtype=torch.float32, device=self.sigmas.device
            )
            ys = torch.arange(
                0, self.heatmap_size[1], 1, dtype=torch.float32, device=self.sigmas.device
            )
            xs = torch.arange(
                0, self.heatmap_size[2], 1, dtype=torch.float32, device=self.sigmas.device
            )
            zs, ys, xs = torch.meshgrid(zs, ys, xs, indexing="ij")
            pre_shape = tuple(1 for _ in range(len(landmarks.shape[:-1])))
            zs = zs.view(*pre_shape, *zs.shape).repeat(*landmarks.shape[:-1], 1, 1, 1)
            ys = ys.view(*pre_shape, *ys.shape).repeat(*landmarks.shape[:-1], 1, 1, 1)
            xs = xs.view(*pre_shape, *xs.shape).repeat(*landmarks.shape[:-1], 1, 1, 1)
            if self.continuous:
                z_t, y_t, x_t = z, y, x
            else:
                z_t, y_t, x_t = z_round, y_round, x_round
            z_t = z_t.view(*landmarks.shape[:-1], 1, 1, 1)
            y_t = y_t.view(*landmarks.shape[:-1], 1, 1, 1)
            x_t = x_t.view(*landmarks.shape[:-1], 1, 1, 1)
            heatmap = self.heatmap_fun(
                torch.stack((z_t, y_t, x_t), -1), torch.stack((zs, ys, xs), -1), covariance, gamma
            )
        return heatmap


class GaussianHeatmapGenerator(HeatmapGenerator):
    """
    Gaussian heatmap generator for generating heatmaps from landmarks.

    Args:
        nb_landmarks (int): number of landmarks
        sigmas (float or list[float] or torch.Tensor or np.ndarray): sigmas of the gaussian heatmap
            function
        gamma (float or None): scaling factor of the gaussian heatmap function
        rotation (float or list[float] or torch.Tensor or np.ndarray): rotation of the gaussian
            heatmap function
        heatmap_size (tuple[int, int]): size of the heatmap
        learnable (bool): whether the sigmas and rotation are learnable
        background (bool): whether to add a background channel to the heatmap
        all_points (bool): whether to add a channel with the sum of all the landmarks
        continuous (bool): whether to use continuous or discrete landmarks
        na_zero (bool): whether to set the value of the landmarks to zero if they are not available
    """

    def __init__(
        self,
        nb_landmarks: int,
        sigmas: float | list[float] | torch.Tensor | np.ndarray = 1.0,
        gamma: Optional[float] = None,
        rotation: float | list[float] | torch.Tensor | np.ndarray = 0,
        heatmap_size: tuple[int, ...] = (512, 512),
        learnable: bool = False,
        background: bool = False,
        all_points: bool = False,
        continuous: bool = True,
        na_zero: bool = False,
    ) -> None:
        super(GaussianHeatmapGenerator, self).__init__(
            nb_landmarks,
            sigmas,
            gamma,
            rotation,
            heatmap_size,
            learnable,
            background,
            all_points,
            continuous,
            na_zero=na_zero,
        )

    def heatmap_fun(
        self,
        landmark_t: torch.Tensor,
        coords: torch.Tensor,
        covariance: torch.Tensor,
        gamma: Optional[float],
    ) -> torch.Tensor:
        """Gaussian heatmap function

        Args:
            landmark_t (torch.Tensor): coordinates of the landmark (y, x) or (z, y, x)
            coords (torch.Tensor): coordinates of the pixel (y, x) or (z, y, x)
            covariance (torch.Tensor): covariance matrix (y, x) or (z, y, x)
            gamma (float or None): scaling factor of the heatmap function


        Returns:
            torch.Tensor: value of the gaussian heatmap function for the given pixel
        """
        if len(covariance.shape) == len(landmark_t.shape[: -(self.spatial_dims - 1)]):
            inverse_covariance = (
                torch.inverse(covariance[..., : self.spatial_dims, : self.spatial_dims])
                .unsqueeze(-3)
                .unsqueeze(-3)
            )
        else:
            # multiple of the same landmarks
            inverse_covariance = (
                torch.inverse(covariance[..., : self.spatial_dims, : self.spatial_dims])
                .unsqueeze(-3)
                .unsqueeze(-3)
                .unsqueeze(-3)
            )
        if self.spatial_dims == 3:
            inverse_covariance = inverse_covariance.unsqueeze(-3)
        diff = (landmark_t - coords).unsqueeze(-2)
        # if self.spatial_dims == 3:
        #     assert False, f"diff shape: {diff.shape}" + f"inv shape: {inverse_covariance.shape}"
        if gamma is not None:
            if self.spatial_dims == 2:
                scale = gamma / (
                    (2 * torch.pi) ** (self.spatial_dims / 2)
                    * torch.sqrt(
                        torch.det(covariance[..., : self.spatial_dims, : self.spatial_dims])
                    )
                ).unsqueeze(-1).unsqueeze(-1)
            else:
                scale = gamma / (
                    (2 * torch.pi) ** (self.spatial_dims / 2)
                    * torch.sqrt(
                        torch.det(covariance[..., : self.spatial_dims, : self.spatial_dims])
                    )
                ).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        else:
            scale = 1
        return scale * torch.exp(-0.5 * (diff @ inverse_covariance @ diff.transpose(-2, -1))).view(
            *landmark_t.shape[: -(self.spatial_dims + 1)],
            *coords.shape[-(self.spatial_dims + 1) : -1],
        )


class LaplacianHeatmapGenerator(HeatmapGenerator):
    """
    Laplacian heatmap generator for generating heatmaps from landmarks.

    Args:
        nb_landmarks (int): number of landmarks
        sigmas (float or list[float] or torch.Tensor or np.ndarray): sigmas of the Laplacian heatmap
            function
        gamma (float or None): scaling factor of the Laplacian heatmap function
        rotation (float or list[float] or torch.Tensor or np.ndarray): rotation of the laplacian
            heatmap function
        heatmap_size (tuple[int, int]): size of the heatmap
        learnable (bool): whether the sigmas and rotation are learnable
        background (bool): whether to add a background channel to the heatmap
        all_points (bool): whether to add a channel with the sum of all the landmarks
        continuous (bool): whether to use continuous or discrete landmarks
        na_zero (bool): whether to set the value of the landmarks to zero if they are not available
    """

    def __init__(
        self,
        nb_landmarks: int,
        sigmas: float | list[float] | torch.Tensor | np.ndarray = 1.0,
        gamma: Optional[float] = None,
        rotation: float | list[float] | torch.Tensor | np.ndarray = 0,
        heatmap_size: tuple[int, ...] = (512, 512),
        learnable: bool = False,
        background: bool = False,
        all_points: bool = False,
        continuous: bool = True,
        na_zero: bool = False,
    ) -> None:
        super(LaplacianHeatmapGenerator, self).__init__(
            nb_landmarks,
            sigmas,
            gamma,
            rotation,
            heatmap_size,
            learnable,
            background,
            all_points,
            continuous,
            na_zero=na_zero,
        )
        if self.spatial_dims != 2:
            raise ValueError("Laplacian heatmap generator only works in 2D")

    def heatmap_fun(
        self,
        landmark_t: torch.Tensor,
        coords: torch.Tensor,
        covariance: torch.Tensor,
        gamma: Optional[float] = None,
    ) -> torch.Tensor:
        """Laplacian heatmap function

        Args:
            landmark_t (torch.Tensor): coordinates of the landmark (y, x)
            coords (torch.Tensor): coordinates of the pixel (y, x)
            covariance (torch.Tensor): covariance matrix (y, x)
            gamma (float or None): scaling factor of the heatmap function


        Returns:
            torch.Tensor: value of the gaussian heatmap function for the given pixel
        """
        if len(covariance.shape) == len(landmark_t.shape[:-1]):
            inverse_covariance = (
                torch.inverse(covariance[..., : self.spatial_dims, : self.spatial_dims])
                .unsqueeze(-(self.spatial_dims + 1))
                .unsqueeze(-(self.spatial_dims + 1))
            )
        else:
            # multiple of the same landmarks
            inverse_covariance = (
                torch.inverse(covariance[..., : self.spatial_dims, : self.spatial_dims])
                .unsqueeze(-(self.spatial_dims + 1))
                .unsqueeze(-(self.spatial_dims + 1))
                .unsqueeze(-(self.spatial_dims + 1))
            )
        diff = (landmark_t - coords).unsqueeze(-2)
        if gamma is not None:
            return (
                gamma
                / ((2 / 3) * torch.pi * torch.sqrt(torch.det(covariance[..., :2, :2])))
                .unsqueeze(-1)
                .unsqueeze(-1)
                * torch.exp(
                    -1 * torch.sqrt(3 * (diff @ inverse_covariance @ diff.transpose(-2, -1)))
                ).view(*landmark_t.shape[:-3], *coords.shape[-3:-1])
            )
        return torch.exp(
            -1 * torch.sqrt(3 * (diff @ inverse_covariance @ diff.transpose(-2, -1)))
        ).view(*landmark_t.shape[:-3], *coords.shape[-3:-1])


def from_2by2_to_4by4(affine_matrix: torch.Tensor) -> torch.Tensor:
    """Converts a 2 by 2 affine matrix to a 4 by 4 affine matrix.

    Args:
        affine_matrix (torch.Tensor): 2 by 2 affine matrix

    Returns:
        torch.Tensor: 4 by 4 affine matrix
    """
    assert affine_matrix.shape[-1] == 2
    assert affine_matrix.shape[-2] == 2
    if len(affine_matrix.shape) == 2:
        return torch.cat(
            (
                torch.cat((affine_matrix, torch.zeros((2, 2), device=affine_matrix.device)), dim=1),
                torch.cat(
                    (
                        torch.zeros((2, 2), device=affine_matrix.device),
                        torch.diag(torch.ones(2, device=affine_matrix.device)),
                    ),
                    dim=1,
                ),
            ),
            dim=0,
        )
    if len(affine_matrix.shape) == 3:
        return torch.cat(
            (
                torch.cat(
                    (affine_matrix, torch.zeros(affine_matrix.shape, device=affine_matrix.device)),
                    dim=2,
                ),
                torch.cat(
                    (
                        torch.zeros(affine_matrix.shape, device=affine_matrix.device),
                        torch.diag(torch.ones(2, device=affine_matrix.device)).repeat(
                            affine_matrix.shape[0], 1, 1
                        ),
                    ),
                    dim=2,
                ),
            ),
            dim=1,
        )
    if len(affine_matrix.shape) == 4:
        return from_2by2_to_4by4(affine_matrix.view(-1, 2, 2)).view(
            affine_matrix.shape[0], affine_matrix.shape[1], 4, 4
        )
    raise ValueError("Affine matrix should be of shape (2,2) or (B,2,2) or (B,C,2,2))")


def from_3by3_to_4by4(affine_matrix: torch.Tensor) -> torch.Tensor:
    """Converts a 3 by 3 affine matrix to a 4 by 4 affine matrix.

    Args:
        affine_matrix (torch.Tensor): 3 by 3 affine matrix

    Returns:
        torch.Tensor: 4 by 4 affine matrix
    """
    assert affine_matrix.shape[-1] == 3
    assert affine_matrix.shape[-2] == 3
    if len(affine_matrix.shape) == 2:
        return torch.cat(
            (
                torch.cat((affine_matrix, torch.zeros((3, 1), device=affine_matrix.device)), dim=1),
                torch.cat(
                    (
                        torch.zeros((1, 3), device=affine_matrix.device),
                        torch.diag(torch.ones(1, device=affine_matrix.device)),
                    ),
                    dim=1,
                ),
            ),
            dim=0,
        )
    if len(affine_matrix.shape) == 3:
        return torch.cat(
            (
                torch.cat(
                    (
                        affine_matrix,
                        torch.zeros((affine_matrix.shape[0], 3, 1), device=affine_matrix.device),
                    ),
                    dim=2,
                ),
                torch.cat(
                    (
                        torch.zeros((affine_matrix.shape[0], 1, 3), device=affine_matrix.device),
                        torch.diag(torch.ones(1, device=affine_matrix.device)).repeat(
                            affine_matrix.shape[0], 1, 1
                        ),
                    ),
                    dim=2,
                ),
            ),
            dim=1,
        )
    if len(affine_matrix.shape) == 4:
        return from_3by3_to_4by4(affine_matrix.view(-1, 3, 3)).view(
            affine_matrix.shape[0], affine_matrix.shape[1], 4, 4
        )
    raise ValueError("Affine matrix should be of shape (3,3) or (B,3,3), (B,C,3,3)")
