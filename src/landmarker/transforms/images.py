"""
Image transforms.
"""

from collections.abc import Mapping
from typing import Hashable

import monai
import monai.transforms
import torch
import torchvision.transforms as T  # type: ignore
from monai.transforms import Resize, SpatialPad


def resize_with_pad(
    images: torch.Tensor, dim: tuple[int, ...]
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """
    Resize images to ``dim`` and pad them to preserve the aspect ratio.
        source: https://github.com/pytorch/vision/issues/6236

    Args:
        images (torch.Tensor): images to resize and pad.
        dim (tuple[int, ...]): dimension of the images.

    Returns:
        images (torch.Tensor): resized and padded images.
        padding (tuple[int, int]): padding applied to the images.
    """
    if len(dim) == 2:
        return resize_with_pad_2d(images, dim)
    elif len(dim) == 3:
        return resize_with_pad_3d(images, dim)
    raise ValueError("spatial_dims must be 2 or 3.")


def resize_with_pad_2d(
    images: torch.Tensor, dim: tuple[int, ...]
) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Resize 2D images to ``dim`` and pad them to preserve the aspect ratio.
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


def resize_with_pad_3d(
    images: torch.Tensor, dim: tuple[int, ...]
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """
    Resize 3D images to ``dim`` and pad them to preserve the aspect ratio.

    Args:
        images (torch.Tensor): images to resize and pad.
        dim (tuple[int, int, int]): dimension of the images (D, H, W).

    Returns:
        images (torch.Tensor): resized and padded images.
        padding (tuple[int, int, int]): padding applied to the images (d_pad, h_pad, w_pad).
    """
    d_orig, h_orig, w_orig = images.shape[-3], images.shape[-2], images.shape[-1]
    d, h, w = dim[0], dim[1], dim[2]

    # Calculate aspect ratios for depth-height and depth-width
    ratio_dh_in = d_orig / h_orig
    ratio_dw_in = d_orig / w_orig
    ratio_hw_in = h_orig / w_orig

    ratio_dh_out = d / h
    ratio_dw_out = d / w
    ratio_hw_out = h / w

    padding = [0, 0, 0]  # [d_padding, h_padding, w_padding]
    padded_images = images

    # Check and adjust depth-height ratio
    if round(ratio_dh_in, 2) != round(ratio_dh_out, 2):
        h_padding = int(d_orig / ratio_dh_out - h_orig)
        d_padding = int(ratio_dh_out * h_orig - d_orig)

        if h_padding > 0 and d_padding < 0:
            h_padding = h_padding // 2
            padding[1] = h_padding
        elif h_padding < 0 and d_padding > 0:
            d_padding = d_padding // 2
            padding[0] = d_padding

    # Check and adjust depth-width ratio
    if round(ratio_dw_in, 2) != round(ratio_dw_out, 2):
        w_padding = int(d_orig / ratio_dw_out - w_orig)
        d_padding = int(ratio_dw_out * w_orig - d_orig)

        if w_padding > 0 and d_padding < 0:
            w_padding = w_padding // 2
            padding[2] = w_padding
        elif (
            w_padding < 0 and d_padding > 0 and padding[0] == 0
        ):  # Only pad depth if not already padded
            d_padding = d_padding // 2
            padding[0] = d_padding

    # Check and adjust height-width ratio if needed and not already adjusted
    if round(ratio_hw_in, 2) != round(ratio_hw_out, 2) and padding[1] == 0 and padding[2] == 0:
        w_padding = int(h_orig / ratio_hw_out - w_orig)
        h_padding = int(ratio_hw_out * w_orig - h_orig)

        if w_padding > 0 and h_padding < 0:
            w_padding = w_padding // 2
            padding[2] = w_padding
        elif w_padding < 0 and h_padding > 0:
            h_padding = h_padding // 2
            padding[1] = h_padding

    # Final resize to target dimensions
    padded_images = SpatialPad((d_orig + padding[0], h_orig + padding[1], w_orig + padding[2]))(
        padded_images
    )
    resized_images = Resize((d, h, w))(padded_images)
    return resized_images, tuple(padding)


class UseOnlyFirstChannel(monai.transforms.MapTransform):
    """
    Select only the first channel of the image.
    """

    def __init__(self, keys: monai.config.KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(
        self, data: Mapping[Hashable, monai.config.type_definitions.NdarrayOrTensor]
    ) -> dict[Hashable, monai.config.type_definitions.NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = d[key][0].unsqueeze(0)
        return d
