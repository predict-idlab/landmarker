"""
Metrics for evaluating the performance of landmark detection models.
"""

from collections.abc import Iterable
from typing import Callable, Optional

import torch

from landmarker.utils.utils import pixel_to_unit


def point_error(
    true_landmarks: torch.Tensor,
    pred_landmarks: torch.Tensor,
    dim: Optional[tuple[int, ...] | torch.Tensor] = None,
    dim_orig: Optional[torch.Tensor] = None,
    pixel_spacing: Optional[torch.Tensor] = None,
    padding: Optional[torch.Tensor] = None,
    reduction: str = "mean",
):
    """
    Calculates the point error between true and predicted landmarks. The point error is the mean
    Euclidean distance between the true and predicted landmarks. If the pixel spacing is given, the
    point error is calculated in mm. If the pixel spacing is not given, the point error is
    calculated in pixels. If the dim and dim_orig are given, the landmarks are rescaled to the
    original image size.

    Args:
        true_landmarks (torch.Tensor): true landmarks
        pred_landmarks (torch.Tensor): predicted landmarks
        dim (Optional[tuple[int, ...] | torch.Tensor], optional): image size. Defaults to None.
        dim_orig (Optional[torch.Tensor], optional): original image size. Defaults to None.
        pixel_spacing (Optional[torch.Tensor], optional): pixel spacing. Defaults to None.
        padding (Optional[torch.Tensor], optional): padding. Defaults to None.
        reduction (str, optional): reduction method. Defaults to "mean". Can be "mean", "nanmean",
            or "none".
    """
    if pixel_spacing is None:
        pixel_spacing = torch.ones((len(true_landmarks), true_landmarks.shape[-1]))
    true_landmarks = pixel_to_unit(
        true_landmarks, pixel_spacing=pixel_spacing, dim=dim, dim_orig=dim_orig, padding=padding
    )
    pred_landmarks = pixel_to_unit(
        pred_landmarks, pixel_spacing=pixel_spacing, dim=dim, dim_orig=dim_orig, padding=padding
    )

    # Calculate MRE
    if reduction == "mean":
        return torch.mean(torch.sqrt(torch.sum((true_landmarks - pred_landmarks) ** 2, -1)))
    if reduction == "none":
        return torch.sqrt(torch.sum((true_landmarks - pred_landmarks) ** 2, -1))
    if reduction == "nanmean":
        return torch.nanmean(torch.sqrt(torch.sum((true_landmarks - pred_landmarks) ** 2, -1)))
    raise ValueError(f"Reduction method {reduction} not supported.")


def multi_instance_point_error(
    true_landmarks: torch.Tensor,
    pred_landmarks: list | torch.Tensor,
    dim: Optional[tuple[int, ...] | torch.Tensor | list[tuple[int, ...]]] = None,
    dim_orig: Optional[torch.Tensor] = None,
    pixel_spacing: Optional[torch.Tensor] = None,
    padding: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    threshold: Optional[float] = None,
) -> (
    tuple[float, float, float, float, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """
    Calculates the multi-instance point error between true and predicted landmarks. The multi-
    instance point error is the mean Euclidean distance between the true and predicted landmarks
    for each instance. If the pixel spacing is given, the point error is calculated in mm. If the
    pixel spacing is not given, the point error is calculated in pixels. If the dim and dim_orig
    are given, the landmarks are rescaled to the original image size. The multi-instance point
    error is the mean of the instance point errors. Next to the multi-instance point error, the
    true positive and false positive rates are returned.

    Args:
        true_landmarks (list | torch.Tensor): true landmarks
        pred_landmarks (list | torch.Tensor): predicted landmarks
        dim (Optional[tuple[int, ...] | torch.Tensor | list[tuple[int, ...]]], optional): image
            size. Defaults to None.
        dim_orig (Optional[torch.Tensor], optional): original image size. Defaults to None.
        pixel_spacing (Optional[torch.Tensor], optional): pixel spacing. Defaults to None.
        padding (Optional[torch.Tensor], optional): padding. Defaults to None.
        reduction (str, optional): reduction method. Defaults to "mean". Can be "mean" or "none".

    Returns:
        tuple[float, float, float, float, torch.Tensor] | tuple[torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor, torch.Tensor]: multi-instance point error, true positive,
            false positive, false negative, predicted landmarks
    """
    if isinstance(pred_landmarks, torch.Tensor):
        pred_landmarks = transform_multi_instance_tensor(pred_landmarks)
    if isinstance(pred_landmarks[0], torch.Tensor):
        pred_landmarks = [[pred_landmarks]]
    if isinstance(pred_landmarks[0][0], torch.Tensor):
        pred_landmarks = [pred_landmarks]  # type: ignore
    assert len(true_landmarks) == len(pred_landmarks)
    assert len(true_landmarks[0]) == len(pred_landmarks[0])

    if dim is None:
        dim = [None] * len(true_landmarks)  # type: ignore
    if dim_orig is None:
        dim_orig = [None] * len(true_landmarks)  # type: ignore
    if pixel_spacing is None:
        pixel_spacing = [None] * len(true_landmarks)  # type: ignore
    if padding is None:
        padding = [None] * len(true_landmarks)  # type: ignore

    if isinstance(dim, tuple):
        dim = [dim] * len(true_landmarks)
    if isinstance(dim_orig, torch.Tensor) and len(dim_orig.shape) == 1:
        dim_orig = dim_orig.unsqueeze(0).repeat(len(true_landmarks), 1)
    if isinstance(pixel_spacing, torch.Tensor) and len(pixel_spacing.shape) == 1:
        pixel_spacing = pixel_spacing.unsqueeze(0).repeat(len(true_landmarks), 1)
    if isinstance(padding, torch.Tensor) and len(padding.shape) == 1:
        padding = padding.unsqueeze(0).repeat(len(true_landmarks), 1)

    pe = torch.zeros((len(true_landmarks), len(true_landmarks[0])))
    pe[pe == 0] = float("nan")
    tp = torch.zeros((len(true_landmarks), len(true_landmarks[0])))
    fp = torch.zeros((len(true_landmarks), len(true_landmarks[0])))
    fn = torch.zeros((len(true_landmarks), len(true_landmarks[0])))
    pred_landmarks_torch = torch.zeros_like(true_landmarks)
    pred_landmarks_torch[pred_landmarks_torch == 0] = float("nan")
    for b, landmark_batch in enumerate(true_landmarks):
        for c, landmark_class in enumerate(landmark_batch):
            picked_true = torch.zeros(len(true_landmarks[b][c]))
            for i, pred_landmark_instance in enumerate(pred_landmarks[b][c]):
                best_pe = float("inf")
                best_j = -1
                for j, true_landmark_instance in enumerate(landmark_class):
                    if picked_true[j]:
                        continue
                    assert pred_landmark_instance.shape == true_landmark_instance.shape
                    pe_inst = point_error(
                        true_landmark_instance.unsqueeze(0),
                        pred_landmark_instance.unsqueeze(0),
                        dim=dim[b],
                        dim_orig=dim_orig[b],  # type: ignore
                        pixel_spacing=pixel_spacing[b],  # type: ignore
                        padding=padding[b],  # type: ignore
                    )  # type: ignore
                    if pe_inst < best_pe:
                        best_pe = pe_inst
                        best_j = j
                if best_j != -1:
                    picked_true[best_j] = 1
                    if pe[b, c].isnan():
                        pe[b, c] = 0
                    pe[b, c] += best_pe
                    pred_landmarks_torch[
                        b,
                        c,
                        best_j,
                    ] = pred_landmark_instance
            if len(pred_landmarks[b][c]) > 0:
                pe[b, c] /= min(len(pred_landmarks[b][c]), len(landmark_class))
            tp[b, c] = torch.nansum(picked_true).item()
            fn[b, c] = len(true_landmarks[b][c]) - tp[b, c]
            fp[b, c] = max(len(pred_landmarks[b][c]) - torch.nansum(picked_true).item(), 0)
    if reduction == "mean":
        return (
            torch.nanmean(pe).item(),
            torch.sum(tp).item(),
            torch.sum(fp).item(),
            torch.sum(fn).item(),
            pred_landmarks_torch,
        )
    return pe, tp, fp, fn, pred_landmarks_torch


def transform_multi_instance_tensor(
    landmarks: torch.Tensor,
) -> list[torch.Tensor] | list[list[torch.Tensor]] | list[list[list[torch.Tensor]]] | list:
    """
    Transforms a multi-instance landmarks into a lists of landmarks. Where nan values are not
        included, since they are not landmarks.

    Args:
        tensor (torch.Tensor): multi-instance tensor

    Returns:
        list[torch.Tensor] | list[list[torch.Tensor]] | list[list[list[torch.Tensor]]]: list of
            tensors
    """
    if len(landmarks.shape) > 4 or len(landmarks.shape) == 1:
        raise ValueError(
            "If true_landmarks is a torch.Tensor, it must have at most 4 "
            + "dimensions. and at least 2 dimensions."
        )
    landmarks_list: (
        list[torch.Tensor] | list[list[torch.Tensor]] | list[list[list[torch.Tensor]]] | list
    ) = []
    for i in range(landmarks.shape[0]):
        if len(landmarks.shape) == 2:
            landmarks_list.append(landmarks[i])  # type: ignore
            continue
        landmarks_list.append([])  # type: ignore
        for j in range(landmarks.shape[1]):
            if len(landmarks.shape) == 3:
                if not torch.any(landmarks[i, j].isnan()):
                    landmarks_list[i].append(landmarks[i, j])  # type: ignore
                continue
            landmarks_list[i].append([])  # type: ignore
            for k in range(landmarks.shape[2]):
                if not torch.any(landmarks[i, j, k].isnan()):
                    landmarks_list[i][j].append(landmarks[i, j, k])  # type: ignore
    return landmarks_list


def sdr(
    radius: Iterable | float,
    true_landmarks: torch.Tensor,
    pred_landmarks: torch.Tensor,
    dim: Optional[tuple[int, ...] | torch.Tensor] = None,
    dim_orig: Optional[torch.Tensor] = None,
    pixel_spacing: Optional[torch.Tensor] = None,
    padding: Optional[torch.Tensor] = None,
    nanmean: bool = False,
) -> float | dict[float, float]:
    """
    Calculates the success detection rate (SDR), which is the percentage of
    predicted landmarks with a point error smaller or equal than the radius.

    Args:
        radius (Iterable | float): radius or list of radii
        true_landmarks (torch.Tensor): true landmarks
        pred_landmarks (torch.Tensor): predicted landmarks
        dim (Optional[tuple[int, ...] | torch.Tensor], optional): image size. Defaults to None.
        dim_orig (Optional[torch.Tensor], optional): original image size. Defaults to None.
        pixel_spacing (Optional[torch.Tensor], optional): pixel spacing. Defaults to None.
        padding (Optional[torch.Tensor], optional): padding. Defaults to None.
        nanmean (bool, optional): use nanmean instead of mean. Defaults to False.

    Returns:
        float | dict[float, float]: SDR or dictionary of SDRs for each radius
    """
    point_error_ = point_error(
        true_landmarks,
        pred_landmarks,
        dim=dim,
        dim_orig=dim_orig,
        pixel_spacing=pixel_spacing,
        padding=padding,
        reduction="none",
    )
    if nanmean:
        reduce_func: Callable = torch.nanmean
    else:
        reduce_func = torch.mean
    if isinstance(radius, Iterable):
        return {r: reduce_func((point_error_ <= r).float()).item() * 100.0 for r in radius}
    return (reduce_func((point_error_ <= radius).float()) * 100).item()
