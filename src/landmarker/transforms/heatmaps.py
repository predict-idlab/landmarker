from typing import Optional, Sequence

import torch


def flip_heatmaps(
    heatmap: torch.Tensor,
    flip_code: int,
    flip_indices_v: Optional[Sequence[int]] = None,
    flip_indices_h: Optional[Sequence[int]] = None,
):
    """
    Flip heatmap.

    Args:
        heatmap (torch.Tensor): heatmap to flip.
        flip_code (int): flip code (0, 1, 2, 3) to apply to the heatmap. 0 is no flip. 1 is
            horizontal flip. 2 is vertical flip. 3 is horizontal and vertical flip.
        flip_indices_v (Sequence[int]): indices of the heatmap to flip vertically.
        flip_indices_h (Sequence[int]): indices of the heatmap to flip horizontally.

    Returns:
        heatmap (torch.Tensor): flipped heatmap.
    """
    if flip_indices_h is None:
        flip_indices_h = [i for i in range(heatmap.shape[1])]
    if flip_indices_v is None:
        flip_indices_v = [i for i in range(heatmap.shape[1])]
    if flip_code == 1 or flip_code == 3:
        return heatmap[..., flip_indices_h]
    if flip_code == 2 or flip_code == 3:
        return heatmap[..., flip_indices_v]
    return heatmap
