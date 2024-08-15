"""
Landmark transforms.
"""

from typing import Optional

import torch


def resize_landmarks(
    landmarks: torch.Tensor,
    orig_dim: tuple[int, ...] | torch.Tensor,
    new_dim: tuple[int, ...] | torch.Tensor,
    padding: Optional[tuple[int, ...] | torch.Tensor] = None,
) -> torch.Tensor:
    """
    Resize landmarks to ``new_dim`` and apply padding if necessary (supplied).

    Args:
        landmarks (torch.Tensor): landmarks to resize.
        orig_dim (tuple[int, int]): original dimension of the images.
        new_dim (tuple[int, int]): new dimension of the images.
        padding (tuple[int, int]): padding applied to the images.

    Returns:
        landmarks (torch.Tensor): resized and padded code landmarks.
    """
    orig_dim = torch.Tensor(orig_dim)
    new_dim = torch.Tensor(new_dim)
    if padding is None:
        padding = torch.zeros(len(orig_dim.size()))
    else:
        padding = torch.Tensor(padding)
    t_landmarks = landmarks + padding
    t_landmarks = t_landmarks * (new_dim / (orig_dim + 2 * padding))
    return t_landmarks


def affine_landmarks(
    landmarks: torch.Tensor,
    affine_matrix: torch.Tensor,  # push affine matrix
) -> torch.Tensor:
    """
    Apply an affine transform to landmarks. The affine matrix is assumed to be a push affine
    transform.

    Args:
        landmarks (torch.Tensor): landmarks to transform.
        affine_matrix (torch.Tensor): 4 by 4 affine matrix to apply to the landmarks.

    Returns:
        landmarks (torch.Tensor): transformed landmarks.
    """
    assert affine_matrix.shape == (4, 4)
    # push affine transform
    if landmarks.shape[-1] == 2:
        landmarks_t = affine_matrix @ torch.cat(
            [
                landmarks,
                torch.zeros((*landmarks.shape[:-1], 1)),
                torch.ones((*landmarks.shape[:-1], 1)),
            ],
            dim=-1,
        ).unsqueeze(-1)
        return landmarks_t[..., :2, 0]
    elif landmarks.shape[-1] == 3:
        landmarks_t = affine_matrix @ torch.cat(
            [
                landmarks,
                torch.ones((*landmarks.shape[:-1], 1)),
            ],
            dim=-1,
        ).unsqueeze(-1)
        return landmarks_t[..., :3, 0]
    raise ValueError(f"Landmarks shape {landmarks.shape} not supported")


def elastic_deform_landmarks(
    landmarks: torch.Tensor,
    grid_displacement: torch.Tensor,
) -> torch.Tensor:
    """
    Apply an elastic deformation to landmarks.

    Args:
        landmarks (torch.Tensor): landmarks to transform.
        displacement (torch.Tensor): displacement field to apply to the landmarks.

    Returns:
        landmarks (torch.Tensor): transformed landmarks.
    """
    raise NotImplementedError(
        "Elastic deformation not implemented yet. MONAI does not"
        " support backtracking the grid sample and affine transform."
    )
    return landmarks
