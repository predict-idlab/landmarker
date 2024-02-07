"""Evaluation visualization functions."""

from typing import Optional

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import seaborn as sns  # type: ignore
import torch

from landmarker.metrics.metrics import point_error, sdr  # type: ignore


def plot_cpe(
    true_landmarks: torch.Tensor,
    pred_landmarks: torch.Tensor,
    dim: tuple[int, ...] | torch.Tensor,
    dim_orig: torch.Tensor,
    pixel_spacing: torch.Tensor,
    padding: torch.Tensor,
    class_names: Optional[list[str]] = None,
    group: bool = True,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    stat: str = "proportion",
    unit: str = "mm",
    kind: str = "ecdf",
):
    """Calculate the cumulative point-to-point error (CPE) and plot the CPE curve.

    Args:
        y_true : numpy.ndarray
            The true values of the target variable, with shape (n, p, 2).
            n is the number of samples, p is the number of landmarks,
            and 2 is respectively the y and x coordinates of the landmarks.
        y_pred : numpy.ndarray
            The predicted values of the target variable, with shape (n, p, 2).
            n is the number of samples, p is the number of landmarks,
            and 2 is respectively the y and x coordinates of the landmarks.
        class_names: list
            The names of the landmarks. The default is None.
        group : bool, optional
            Whether to group the CPE curves by landmarks. The default is True.
        title : str, optional
            The title of the plot. The default is None.
        save_path : str, optional
            The path to save the plot. The default is None.
        stat : str, optional
            The type of statistic to plot. The default is 'proportion'.
        unit : str, optional
            The unit of distance. The default is 'mm'.
        kind : str, optional
            The type of plot. The default is 'ecdf'.
            Possible values are 'ecdf', 'kde', and 'hist'.
    """
    # Calculate the point-to-point error (PE)
    pe = point_error(
        true_landmarks, pred_landmarks, dim, dim_orig, pixel_spacing, padding, reduction="none"
    )
    pe = pe.cpu().detach().numpy()
    if len(pe.shape) == 3:
        pe = np.nansum(pe, axis=-1)

    # Plot the CPE curve
    if group:
        # Calculate the mean point-to-point error (MPE) for each sample
        pe = np.nanmean(pe, axis=1)
        if kind == "ecdf":
            ax = sns.ecdfplot(x=pe, stat=stat, label="Sample MPE")
        elif kind == "kde":
            ax = sns.kdeplot(x=pe, cumulative=True, label="Sample MPE")
        elif kind == "hist":
            ax = sns.histplot(x=pe, stat=stat, cumulative=True, label="Sample MPE")
        else:
            raise ValueError(f"Invalid kind: {kind!r}.")
    else:
        if class_names is None:
            class_names = [f"Landmark {i}" for i in range(true_landmarks.shape[1])]
        fig, ax = plt.subplots()
        for i, class_name in enumerate(class_names):
            if kind == "ecdf":
                sns.ecdfplot(x=pe[:, i], stat=stat, ax=ax, label=class_name)
            elif kind == "kde":
                sns.kdeplot(x=pe[:, i], cumulative=True, ax=ax, label=class_name)
            elif kind == "hist":
                sns.histplot(x=pe[:, i], stat=stat, cumulative=True, ax=ax, label=class_name)
            else:
                raise ValueError(f"Invalid kind: {kind!r}.")
    ax.set(xlabel=f"IPE in {unit}", ylabel="Proportion of images", title=title)
    ax.grid()
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if save_path is not None:
        fig.savefig(save_path)
    plt.show()


def detection_report(
    true_landmarks: torch.Tensor,
    pred_landmarks: torch.Tensor,
    dim: tuple[int, ...] | torch.Tensor,
    dim_orig: torch.Tensor,
    pixel_spacing: torch.Tensor,
    padding: torch.Tensor,
    class_names: Optional[list[str]] = None,
    radius: list[float] = [2, 2.5, 3, 4],
    digits: int = 2,
    unit: str = "mm",
    output_dict: bool = False,
):
    """Calculate the detection report.

    Args:
        true_landmarks : torch.Tensor
            The true landmarks, with shape (n, p, 2).
            n is the number of samples, p is the number of landmarks,
            and 2 is respectively the y and x coordinates of the landmarks.
        pred_landmarks : torch.Tensor
            The predicted landmarks, with shape (n, p, 2).
            n is the number of samples, p is the number of landmarks,
            and 2 is respectively the y and x coordinates of the landmarks.
        dim : tuple[int, ...] | torch.Tensor
            The dimension of the image, with shape (2,).
            2 is respectively the height and width of the image.
        dim_orig : torch.Tensor
            The original dimension of the image, with shape (2,).
            2 is respectively the height and width of the image.
        pixel_spacing : torch.Tensor
            The pixel spacing of the image, with shape (2,).
            2 is respectively the height and width of the image.
        padding : torch.Tensor
            The padding of the image, with shape (2,).
            2 is respectively the height and width of the image.
        class_names: list
            The names of the landmarks. The default is None.
        radius : list[float], optional
            The radius of the success detection rate (SDR). The default is [2, 2.5, 3, 4].
        digits : int, optional
            The number of digits to round the statistics. The default is 2.
        unit : str, optional
            The unit of distance. The default is 'mm'.
        output_dict : bool, optional
            Whether to output the detection report as a dictionary. The default is False.

    Returns:
        report : dict
            The detection report. Only output when output_dict is True.
    """
    # Calculate the cumulative point-to-point error (CPE)
    pe = point_error(
        true_landmarks, pred_landmarks, dim, dim_orig, pixel_spacing, padding, reduction="none"
    )
    pe = pe.cpu().detach().numpy()
    if class_names is None:
        class_names = [f"Landmark {i}" for i in range(true_landmarks.shape[1])]
    # Calculate the detection report
    report: dict[str, dict[str, float | dict[float, float]]] = {}
    for i, class_name in enumerate(class_names):  # type: ignore
        report[class_name] = {}
        report[class_name]["Mean"] = np.mean(pe[:, i])
        report[class_name]["Median"] = np.median(pe[:, i])
        report[class_name]["Std"] = np.std(pe[:, i])
        report[class_name]["Min"] = np.min(pe[:, i])
        report[class_name]["Max"] = np.max(pe[:, i])
        sdr_class = sdr(
            radius,
            true_landmarks=true_landmarks[:, i],
            pred_landmarks=pred_landmarks[:, i],
            dim=dim,
            dim_orig=dim_orig,
            pixel_spacing=pixel_spacing,
            padding=padding,
        )
        report[class_name]["SDR"] = sdr_class

    # Print the detection report in a nice table
    sdr_names = [f"SDR (PE≤{r}{unit})" for r in radius]
    print("Detection report:")
    print("1# Point-to-point error (PE) statistics:")
    print("=" * (20 + 10 * 5))
    print(f"{'Class':<20}{'Mean':<10}{'Median':<10}{'Std':<10}{'Min':<10}{'Max':<10}")
    print("-" * (20 + 10 * 5))
    for class_name in report:
        print(
            f"{class_name:<20}"
            f"{report[class_name]['Mean']:<10.{digits}f}"
            f"{report[class_name]['Median']:<10.{digits}f}"
            f"{report[class_name]['Std']:<10.{digits}f}"
            f"{report[class_name]['Min']:<10.{digits}f}"
            f"{report[class_name]['Max']:<10.{digits}f}"
        )
    print("=" * (20 + 10 * 5))

    print("\n2# Success detection rate (SDR):")
    print("=" * (20 + 15 * len(radius)))
    print("".join([f"{'Class':<20}"] + [f"{sdr_name:<15}" for sdr_name in sdr_names]))
    print("-" * (20 + 15 * len(radius)))
    for class_name in report:
        print(
            "".join(
                [f"{class_name:<20}"]
                + [f"{report[class_name]['SDR'][r]:<15.{digits}f}" for r in radius]  # type: ignore
            )
        )
    print("=" * (20 + 15 * len(radius)))

    if output_dict:
        return report


def multi_instance_detection_report(
    true_landmarks: torch.Tensor,
    pred_landmarks: torch.Tensor,
    true_positives: torch.Tensor,
    false_positives: torch.Tensor,
    false_negatives: torch.Tensor,
    dim: tuple[int, ...] | torch.Tensor,
    dim_orig: torch.Tensor,
    pixel_spacing: torch.Tensor,
    padding: torch.Tensor,
    class_names: Optional[list[str]] = None,
    radius: list[float] = [2, 2.5, 3, 4],
    digits: int = 2,
    unit: str = "mm",
    output_dict: bool = False,
):
    """Calculate the multi instance detection report.

    Args:
        true_landmarks : torch.Tensor
            The true landmarks, with shape (n, p, i, 2). n is the number of samples, p is the number
            of classes of landmarks, i is the number of instances, and 2 is respectively the y and
            x coordinates of the landmarks.
        pred_landmarks : torch.Tensor
            The predicted landmarks, with shape (n, p, i, 2). n is the number of samples, p is the
            number of classes of landmarks, i is the number of instances, and 2 is respectively the
            y and x coordinates of the landmarks.
        true_positives : torch.Tensor
            The true positives, with shape (n, p, i). n is the number of samples, p is the number of
            classes of landmarks, and i is the number of instances.
        false_positives : torch.Tensor
            The false positives, with shape (n, p, i). n is the number of samples, p is the number
            of classes of landmarks, and i is the number of instances.
        false_negatives : torch.Tensor
            The false negatives, with shape (n, p, i). n is the number of samples, p is the number
            of classes of landmarks, and i is the number of instances.
        dim : tuple[int, ...] | torch.Tensor
            The dimension of the image, with shape (2,).
            2 is respectively the height and width of the image.
        dim_orig : torch.Tensor
            The original dimension of the image, with shape (2,).
            2 is respectively the height and width of the image.
        pixel_spacing : torch.Tensor
            The pixel spacing of the image, with shape (2,).
            2 is respectively the height and width of the image.
        padding : torch.Tensor
            The padding of the image, with shape (2,).
            2 is respectively the height and width of the image.
        class_names: list
            The names of the landmarks. The default is None.
        radius : list[float], optional
            The radius of the success detection rate (SDR). The default is [2, 2.5, 3, 4].
        digits : int, optional
            The number of digits to round the statistics. The default is 2.
        unit : str, optional
            The unit of distance. The default is 'mm'.
        output_dict : bool, optional
            Whether to output the detection report as a dictionary. The default is False.

    Returns:
        report : dict
            The detection report. Only output when output_dict is True.
    """
    # Calculate the cumulative point-to-point error (CPE)
    pe = point_error(
        true_landmarks, pred_landmarks, dim, dim_orig, pixel_spacing, padding, reduction="none"
    )
    pe = pe.cpu().detach().numpy()

    # Calculate the detection report
    report: dict[str, dict[str, float | dict[float, float]]] = {}
    for i, class_name in enumerate(class_names):  # type: ignore
        report[class_name] = {}
        report[class_name]["Mean"] = np.nanmean(pe[:, i])
        report[class_name]["Median"] = np.nanmedian(pe[:, i])
        report[class_name]["Std"] = np.nanstd(pe[:, i])
        report[class_name]["Min"] = np.nanmin(pe[:, i])
        report[class_name]["Max"] = np.nanmax(pe[:, i])
        report[class_name]["TP"] = torch.sum(true_positives[:, i]).item()
        report[class_name]["FP"] = torch.sum(false_positives[:, i]).item()
        report[class_name]["FN"] = torch.sum(false_negatives[:, i]).item()
        report[class_name]["TPR"] = report[class_name]["TP"] / (  # type: ignore
            report[class_name]["TP"] + report[class_name]["FN"]  # type: ignore
        )
        sdr_class = sdr(
            radius,
            true_landmarks=true_landmarks[:, i],
            pred_landmarks=pred_landmarks[:, i],
            dim=dim,
            dim_orig=dim_orig,
            pixel_spacing=pixel_spacing,
            padding=padding,
        )
        report[class_name]["SDR"] = sdr_class

    # Print the detection report in a nice table
    sdr_names = [f"SDR (PE≤{r}{unit})" for r in radius]
    print("Detection report:")
    print("1# Instance detection statistics:")
    print("=" * (20 + 10 * 4))
    print(f"{'Class':<20}{'TP':<10}{'FP':<10}{'FN':<10}{'TPR':<10}")
    print("-" * (20 + 10 * 4))
    for class_name in report:
        print(
            f"{class_name:<20}"
            f"{report[class_name]['TP']:<10}"
            f"{report[class_name]['FP']:<10}"
            f"{report[class_name]['FN']:<10}"
            f"{report[class_name]['TPR']:<10.{digits}f}"
        )
    print("=" * (20 + 10 * 4))

    print("\n2# Point-to-point error (PE) statistics:")
    print("=" * (20 + 10 * 5))
    print(f"{'Class':<20}{'Mean':<10}{'Median':<10}{'Std':<10}{'Min':<10}{'Max':<10}")
    print("-" * (20 + 10 * 5))
    for class_name in report:
        print(
            f"{class_name:<20}"
            f"{report[class_name]['Mean']:<10.{digits}f}"
            f"{report[class_name]['Median']:<10.{digits}f}"
            f"{report[class_name]['Std']:<10.{digits}f}"
            f"{report[class_name]['Min']:<10.{digits}f}"
            f"{report[class_name]['Max']:<10.{digits}f}"
        )
    print("=" * (20 + 10 * 5))

    print("\n3# Success detection rate (SDR):")
    print("=" * (20 + 15 * len(radius)))
    print("".join([f"{'Class':<20}"] + [f"{sdr_name:<15}" for sdr_name in sdr_names]))
    print("-" * (20 + 15 * len(radius)))
    for class_name in report:
        print(
            "".join(
                [f"{class_name:<20}"]
                + [f"{report[class_name]['SDR'][r]:<15.{digits}f}" for r in radius]  # type: ignore
            )
        )
    print("=" * (20 + 15 * len(radius)))

    if output_dict:
        return report
