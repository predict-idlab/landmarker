"""
Image transforms.
"""

import torch
import torchvision.transforms as T  # type: ignore


def resize_with_pad(
    images: torch.Tensor, dim: tuple[int, int]
) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Resize images to ``dim`` and pad them to preserve the aspect ratio.
        source: https://github.com/pytorch/vision/issues/6236

    Args:
        images (torch.Tensor): images to resize and pad.
        dim (tuple[int, int]): dimension of the images.

    Returns:
        images (torch.Tensor): resized and padded images.
        padding (tuple[int, int]): padding applied to the images.
    """
    h_orig, w_orig = images.shape[-2], images.shape[-1]
    h, w = dim[0], dim[1]
    ratio_in = h_orig / w_orig
    ratio_out = h / w

    # check if the original and final aspect ratios are the same within a margin
    if round(ratio_in, 2) != round(ratio_out, 2):
        # padding to preserve aspect ratio
        w_padding = int(h_orig / ratio_out - w_orig)
        h_padding = int(ratio_out * w_orig - h_orig)
        if w_padding > 0 and h_padding < 0:
            w_padding = w_padding // 2
            images = T.functional.pad(images, (w_padding, 0), 0, "constant")
            return T.functional.resize(images, (h, w)), (0, w_padding)

        if w_padding < 0 and h_padding > 0:
            h_padding = h_padding // 2
            images = T.functional.pad(images, (0, h_padding), 0, "constant")
            return T.functional.resize(images, (h, w)), (h_padding, 0)
    return T.functional.resize(images, (h, w)), (0, 0)
