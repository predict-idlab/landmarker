"""Heatmap generator"""

from abc import abstractmethod
from typing import Optional

import numpy as np
import torch
from typing_extensions import Self


class HeatmapGenerator:
    """
    Heatmap generator abstract class for generating heatmaps from landmarks

    Args:
        nb_landmarks (int): number of landmarks
        sigmas (float or list[float] or torch.Tensor or np.ndarray): sigmas of the heatmap function
        gamma (float or None): scaling factor of the heatmap function
        rotation (float or list[float] or torch.Tensor or np.ndarray): rotation of the heatmap
            function
        heatmap_size (tuple[int, int]): size of the heatmap
        full_map (bool): whether to return the full heatmap or only the part around the landmark
        learnable (bool): whether the sigmas and rotation are learnable
        background (bool): whether to add a background channel to the heatmap
        all_points (bool): whether to add a channel with the sum of all the landmarks
        continuous (bool): whether to use continuous or discrete landmarks
        device (str): device to use for the heatmap generator
    """

    def __init__(
        self,
        nb_landmarks: int,
        sigmas: float | list[float] | torch.Tensor | np.ndarray = 1.0,
        gamma: Optional[float] = None,
        rotation: float | list[float] | torch.Tensor | np.ndarray = 0,
        heatmap_size: tuple[int, int] = (512, 512),
        full_map: bool = True,
        learnable: bool = False,
        background: bool = False,
        all_points: bool = False,
        continuous: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        self.nb_landmarks = nb_landmarks
        self.learnable = learnable
        self.device = device
        self.set_sigmas(sigmas)
        self.set_rotation(rotation)
        self.sigmas: torch.Tensor = self.sigmas.to(device)
        self.rotation: torch.Tensor = self.rotation.to(device)
        self.gamma = gamma
        self.heatmap_size = heatmap_size
        self.full_map = full_map
        self.background = background
        self.all_points = all_points
        self.continuous = continuous
        self.bound = 3
        self.epsilon = 1e-2

    def set_sigmas(self, sigmas: float | list[float] | torch.Tensor | np.ndarray) -> None:
        """
        Set the sigmas of the heatmap function.

        Args:
            sigmas (float or list[float] or torch.Tensor or np.ndarray): sigmas of the heatmap
                function
        """
        sigmas = torch.tensor(
            sigmas, device=self.device, dtype=torch.float, requires_grad=self.learnable
        )
        self.sigmas = (
            torch.ones(
                (self.nb_landmarks, 2),
                requires_grad=self.learnable,
                device=self.device,
                dtype=torch.float,
            )
            * sigmas
        )
        if self.learnable:
            self.sigmas = torch.nn.Parameter(self.sigmas)

    def set_rotation(self, rotation: float | list[float] | torch.Tensor | np.ndarray) -> None:
        """
        Set the rotation of the heatmap function.

        Args:
            rotation (float or list[float] or torch.Tensor or np.ndarray): rotation of the heatmap
                function
        """
        rotation = torch.tensor(
            rotation, device=self.device, dtype=torch.float, requires_grad=self.learnable
        )
        self.rotation = (
            torch.ones(
                (self.nb_landmarks),
                requires_grad=self.learnable,
                device=self.device,
                dtype=torch.float,
            )
            * rotation
        )
        if self.learnable:
            self.rotation = torch.nn.Parameter(self.rotation)

    def to(self, device: str | torch.device) -> Self:
        """
        Move the heatmap generator to a given device.

        Args:
            device (str): device to move the heatmap generator to
        """
        self.device = device
        self.sigmas = self.sigmas.to(device)
        self.rotation = self.rotation.to(device)
        return self

    def __call__(
        self, landmarks: torch.Tensor, affine_matrix: torch.Tensor = torch.eye(4)
    ) -> torch.Tensor:
        assert affine_matrix.shape[-1] == affine_matrix.shape[-2]
        if len(affine_matrix.shape) == 2:
            affine_matrix = affine_matrix.unsqueeze(0)
        if affine_matrix.shape[-1] == 2:
            # go from 2 by 2 affine matrix to 4 by 4
            affine_matrix = from_2by2_to_4by4(affine_matrix)
        elif affine_matrix.shape[-1] == 3:
            # go from 3 by 3 affine matrix to 4 by 4
            affine_matrix = from_3by3_to_4by4(affine_matrix)
        affine_matrix = affine_matrix.to(self.device).unsqueeze(1)
        heatmaps = torch.zeros((landmarks.shape[0], landmarks.shape[1], *self.heatmap_size)).to(
            self.device
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
                    torch.ones((heatmaps.shape[0], 1, *heatmaps.shape[2:]), device=self.device)
                    - heatmaps.sum(dim=1, keepdim=True),
                    heatmaps,
                ),
                1,
            )
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
            landmark_t (torch.Tensor): coordinates of the landmark (y, x)
            coords (torch.Tensor): coordinates of the pixel (y, x)
            covariance (torch.Tensor): covariance matrix (y, x)
            gamma (float or None): scaling factor of the heatmap function
        """

    def get_covariance_matrix(self, return4by4: bool = False) -> torch.Tensor:
        """
        Get the covariance matrix of the heatmap function.

        Args:
            return4by4 (bool): whether to return a 4 by 4 covariance matrix or a 2 by 2 covariance
                matrix

        Returns:
            torch.Tensor: covariance matrix
        """
        rotation = torch.stack(
            (
                torch.stack((torch.cos(self.rotation), -torch.sin(self.rotation)), dim=-1),
                torch.stack((torch.sin(self.rotation), torch.cos(self.rotation)), dim=-1),
            ),
            dim=-2,
        )

        diagonal = torch.diag_embed((self.sigmas**2))
        if return4by4:
            rotation = from_2by2_to_4by4(rotation).to(self.device)
            diagonal = from_2by2_to_4by4(diagonal).to(self.device)
        covariance = rotation @ diagonal @ rotation.transpose(-2, -1)
        return covariance

    def create_heatmap(
        self, landmarks: torch.Tensor, gamma: Optional[float], affine_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Create a heatmap for a given landmark in an image returns a heatmap with the same size as
        the image. Works with batches and multiple landmarks.

        Args:
            landmarks (torch.Tensor): landmarks of shape (B, C, M, 2) or (B, C, 2)
            gamma (float or None): scaling factor of the heatmap function
            affine_matrix (torch.Tensor): affine matrix of shape (B, 1, 4, 4)

        Returns:
            torch.Tensor: heatmap of shape (B, C, M, H, W) or (B, C, H, W)
        """
        covariance = self.get_covariance_matrix(return4by4=True)
        covariance = affine_matrix @ covariance @ affine_matrix.transpose(-2, -1)

        x = landmarks[..., 1]
        y = landmarks[..., 0]
        x_round = torch.round(x).int()
        y_round = torch.round(y).int()
        if self.full_map:
            xs = torch.arange(0, self.heatmap_size[1], 1, dtype=torch.float32, device=self.device)
            ys = torch.arange(0, self.heatmap_size[0], 1, dtype=torch.float32, device=self.device)
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
            return heatmap
        raise NotImplementedError()
        # heatmap = torch.zeros((*landmarks.shape[:-1], *self.heatmap_size)).to(self.device)
        # max_dist_x = torch.round(self.bound * sigmas[1]).int().to(self.device)
        # max_dist_y = torch.round(self.bound * sigmas[0]).int().to(self.device)
        # xs = torch.arange(0, 2 * max_dist_x + 1, 1, dtype=torch.float32, device=self.device)
        # ys = torch.arange(0, 2 * max_dist_y + 1, 1, dtype=torch.float32, device=self.device)
        # xs, ys = torch.meshgrid(xs, ys, indexing='xy')
        # xs = xs.view(1, *xs.shape).repeat(*landmarks.shape[:-1], 1, 1)
        # ys = ys.view(1, *ys.shape).repeat(*landmarks.shape[:-1], 1, 1)
        # x_t, y_t = max_dist_x.float(), max_dist_y.float()
        # if self.continuous:
        #     x_t = x_t + (x - x_round)
        #     y_t = y_t + (y - y_round)
        # x_t = x_t.view(*landmarks.shape[:-1], 1, 1)
        # y_t = y_t.view(*landmarks.shape[:-1], 1, 1)
        # g = self.heatmap_fun(x_t, y_t, xs, ys, covariance, gamma)
        # xs_min = torch.maximum(torch.tensor(
        #     [0], device=self.device), x_round - max_dist_x).int()
        # xs_max = torch.minimum(torch.tensor(
        #     [self.heatmap_size[1]], device=self.device), x_round + max_dist_x + 1).int()
        # ys_min = torch.maximum(torch.tensor(
        #     [0], device=self.device), y_round - max_dist_y).int()
        # ys_max = torch.minimum(torch.tensor(
        #     [self.heatmap_size[0]], device=self.device), y_round + max_dist_y + 1).int()

        # g_min_x, g_max_x = max_dist_x + xs_min - x_round, max_dist_x + xs_max - x_round
        # g_min_y, g_max_y = max_dist_y + ys_min - y_round, max_dist_y + ys_max - y_round

        # # all g's below epsilon are set to 0
        # g[g < self.epsilon] = 0

        # # Set heatmap to values, by looping through the batch
        # for b in range(landmarks.shape[0]):
        #     for c in range(landmarks.shape[1]):
        #         if len(landmarks.shape) > 3:
        #             for m in range(landmarks.shape[2]):
        #                 heatmap[b, c, m, ys_min[b, c, m]:ys_max[b, c, m],
        #                         xs_min[b, c, m]:xs_max[b, c, m]
        #                         ] = g[b, g_min_y[b, c, m]:g_max_y[b, c, m],
        #                               g_min_x[b, c, m]:g_max_x[b, c, m]]
        #         else:
        #             heatmap[b, c, ys_min[b, c]:ys_max[b, c], xs_min[b, c]:xs_max[b, c]
        #                     ] = g[b, g_min_y[b, c]:g_max_y[b, c], g_min_x[b, c]:g_max_x[b, c]]
        # return heatmap


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
        full_map (bool): whether to return the full heatmap or only the part around the landmark
        learnable (bool): whether the sigmas and rotation are learnable
        background (bool): whether to add a background channel to the heatmap
        all_points (bool): whether to add a channel with the sum of all the landmarks
        continuous (bool): whether to use continuous or discrete landmarks
        device (str): device to use for the heatmap generator
    """

    def __init__(
        self,
        nb_landmarks: int,
        sigmas: float | list[float] | torch.Tensor | np.ndarray = 1.0,
        gamma: Optional[float] = None,
        rotation: float | list[float] | torch.Tensor | np.ndarray = 0,
        heatmap_size: tuple[int, int] = (512, 512),
        full_map: bool = True,
        learnable: bool = False,
        background: bool = False,
        all_points: bool = False,
        continuous: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(
            nb_landmarks,
            sigmas,
            gamma,
            rotation,
            heatmap_size,
            full_map,
            learnable,
            background,
            all_points,
            continuous,
            device,
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
            landmark_t (torch.Tensor): coordinates of the landmark (y, x)
            coords (torch.Tensor): coordinates of the pixel (y, x)
            covariance (torch.Tensor): covariance matrix (y, x)
            gamma (float or None): scaling factor of the heatmap function


        Returns:
            torch.Tensor: value of the gaussian heatmap function for the given pixel
        """
        if len(covariance.shape) == len(landmark_t.shape[:-1]):
            inverse_covariance = torch.inverse(covariance[..., :2, :2]).unsqueeze(-3).unsqueeze(-3)
        else:
            # multiple of the same landmarks
            inverse_covariance = (
                torch.inverse(covariance[..., :2, :2]).unsqueeze(-3).unsqueeze(-3).unsqueeze(-3)
            )
        diff = (landmark_t - coords).unsqueeze(-2)
        if gamma is not None:
            return (
                gamma
                / (2 * torch.pi * torch.sqrt(torch.det(covariance[..., :2, :2])))
                .unsqueeze(-1)
                .unsqueeze(-1)
                * torch.exp(-0.5 * (diff @ inverse_covariance @ diff.transpose(-2, -1))).view(
                    *landmark_t.shape[:-3], *coords.shape[-3:-1]
                )
            )
        return torch.exp(-0.5 * (diff @ inverse_covariance @ diff.transpose(-2, -1))).view(
            *landmark_t.shape[:-3], *coords.shape[-3:-1]
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
        full_map (bool): whether to return the full heatmap or only the part around the landmark
        learnable (bool): whether the sigmas and rotation are learnable
        background (bool): whether to add a background channel to the heatmap
        all_points (bool): whether to add a channel with the sum of all the landmarks
        continuous (bool): whether to use continuous or discrete landmarks
        device (str): device to use for the heatmap generator
    """

    def __init__(
        self,
        nb_landmarks: int,
        sigmas: float | list[float] | torch.Tensor | np.ndarray = 1.0,
        gamma: Optional[float] = None,
        rotation: float | list[float] | torch.Tensor | np.ndarray = 0,
        heatmap_size: tuple[int, int] = (512, 512),
        full_map: bool = True,
        learnable: bool = False,
        background: bool = False,
        all_points: bool = False,
        continuous: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(
            nb_landmarks,
            sigmas,
            gamma,
            rotation,
            heatmap_size,
            full_map,
            learnable,
            background,
            all_points,
            continuous,
            device,
        )

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
            inverse_covariance = torch.inverse(covariance[..., :2, :2]).unsqueeze(-3).unsqueeze(-3)
        else:
            # multiple of the same landmarks
            inverse_covariance = (
                torch.inverse(covariance[..., :2, :2]).unsqueeze(-3).unsqueeze(-3).unsqueeze(-3)
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
