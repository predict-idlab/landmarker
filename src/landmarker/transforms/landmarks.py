"""
Landmark transforms.
"""


from typing import Optional, Sequence

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


def flip_landmarks_with_affine(
    landmarks: torch.Tensor,
    affine_matrix: torch.Tensor,
    flip_indices: Sequence[int],
):
    """
    Flip landmarks.

    Args:
        landmarks (torch.Tensor): landmarks to flip.
        affine_matrix (torch.Tensor): affine matrix to apply to the landmarks.
        flip_indices (Sequence[int]): indices of the landmarks to flip.

    Returns:
        landmarks (torch.Tensor): flipped landmarks.
    """
    t_landmarks = affine_landmarks(landmarks, affine_matrix)
    return t_landmarks[..., flip_indices]


def flip_landmarks(
    landmarks: torch.Tensor,
    flip_code: int,
    flip_indices_v: Optional[Sequence[int]] = None,
    flip_indices_h: Optional[Sequence[int]] = None,
):
    """
    Flip landmarks.

    Args:
        landmarks (torch.Tensor): landmarks to flip.
        flip_code (int): flip code (0, 1, 2, 3) to apply to the landmarks. 0 is no flip. 1 is
            horizontal flip. 2 is vertical flip. 3 is horizontal and vertical flip.
        flip_indices_v (Sequence[int]): indices of the landmarks to flip vertically.
        flip_indices_h (Sequence[int]): indices of the landmarks to flip horizontally.

    Returns:
        landmarks (torch.Tensor): flipped landmarks.
    """
    if flip_indices_h is None:
        flip_indices_h = [i for i in range(landmarks.shape[1])]
    if flip_indices_v is None:
        flip_indices_v = [i for i in range(landmarks.shape[1])]
    if flip_code == 1 or flip_code == 3:
        return landmarks[..., flip_indices_h]
    if flip_code == 2 or flip_code == 3:
        return landmarks[..., flip_indices_v]
    return landmarks
