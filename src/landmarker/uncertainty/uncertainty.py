"""Coppied from https://github.com/predict-idlab/landmark-uq"""

from typing import Optional

import cv2
import numpy as np
import torch
from scipy.stats import norm, spearmanr  # type: ignore
from scipy.stats.mstats import mquantiles  # type: ignore
from skimage import measure  # type: ignore
from tqdm import tqdm  # type: ignore

from landmarker.utils import pixel_to_unit_numpy


def mahalanobis_distance(pred, pred_cov, target):
    """Calculate the Mahalanobis distance between the predicted mean and the target.

    Args:
        pred: numpy array of shape (..., D) containing predicted means
        pred_cov: numpy array of shape (..., D, D) containing covariance matrices
        target: numpy array of shape (..., D) containing target values

    Returns:
        numpy array of shape (...) containing Mahalanobis distances
    """
    # Add a dimension to make diff shape (..., 1, D)
    diff = np.expand_dims(pred - target, axis=-2)
    # Calculate inverse of covariance matrix
    cov_inv = np.linalg.inv(pred_cov)
    # Calculate Mahalanobis distance
    # diff @ cov_inv @ diff.T -> shape (..., 1, 1)
    dist = np.sqrt(
        (diff @ cov_inv @ np.transpose(diff, axes=(*range(diff.ndim - 2), -1, -2)))
        .squeeze(axis=-1)
        .squeeze(axis=-1)
    )
    return dist


def euclidean_distance(pred, target):
    """Calculate the Euclidean distance between the predicted mean and the target.

    Args:
        pred: numpy array of shape (..., D)
        target: numpy array of shape (..., D)

    Returns:
        numpy array of shape (...)
    """
    return np.sqrt(np.sum((pred - target) ** 2, axis=-1))


def absolute_error_max(pred, target, sigmas=None):
    """Calculate the absolute error between the predicted mean and the target. The
    maximum absolute error is taken over the last dimension.

    Args:
        pred: numpy array of any shape
        target: numpy array of the same shape as pred
        sigmas: numpy array of the same shape as pred, containing the standard deviation

    Returns:
        numpy array of the same shape as inputs
    """
    if sigmas is None:
        return np.abs(pred - target).max(axis=-1)
    return (np.abs(pred - target) / sigmas).max(axis=-1)


class ConformalRegressorMahalanobis:
    def __init__(self, spatial_dims=2, nb_landmarks=1):
        self.spatial_dims = spatial_dims
        self.nb_landmarks = nb_landmarks
        self.alphas = None

    def check_input(self, pred, pred_cov):
        if pred.ndim != 3:
            raise ValueError("pred must have shape (N, C, D)")
        if pred_cov.ndim != 4:
            raise ValueError("pred_cov must have shape (N, C, D, D)")
        if pred.shape[2] != pred_cov.shape[2] and self.spatial_dims != pred.shape[2]:
            raise ValueError("pred and pred_cov must have the same second dimension")
        if pred_cov.shape[2] != pred_cov.shape[3]:
            raise ValueError("pred_cov must be square")
        if pred.shape[0] != pred_cov.shape[0]:
            raise ValueError("pred and pred_cov must have the same first dimension")
        if pred.shape[1] != self.nb_landmarks:
            raise ValueError("pred must have the same second dimension as nb_landmarks")
        if pred_cov.shape[1] != self.nb_landmarks:
            raise ValueError("pred_cov must have the same second dimension as nb_landmarks")

    def fit(self, pred, pred_cov, target):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().numpy()
        if isinstance(pred_cov, torch.Tensor):
            pred_cov = pred_cov.detach().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().numpy()
        self.check_input(
            pred, pred_cov
        )  # check if the target has the same shape as the calibration set
        assert pred.shape == target.shape
        self.alphas = mahalanobis_distance(pred, pred_cov, target)
        # Sort the non-conformity scores in descending order
        self.alphas = np.sort(self.alphas, axis=0)[::-1]

    def bound(self, pred_cov, confidence=0.95):
        """Get the bound for the given confidence level."""
        if self.alphas is None:
            raise ValueError("You must call fit before calling predict")
        return pred_cov * (
            self.alphas[int((1 - confidence) * (len(self.alphas) + 1)) - 1].reshape((1, -1, 1, 1))
            ** 2
        )

    def area_prediction_region(self, pred_cov, spacing=None, confidence=0.95):
        """
        Calculate the area of the prediction region, accounting for pixel spacing.

        Parameters:
        pred_cov (ndarray): The predicted covariance matrix.
        spacing (ndarray, optional): The spacing between pixels in millimeters,
                         with shape (Batch_size, spatial_size).
                         Defaults to None.
        confidence (float, optional): The confidence level for the prediction region.
                         Defaults to 0.95.

        Returns:
        float: The area of the prediction region.
        """
        pred_cov_scaled = self.bound(pred_cov, confidence)
        if spacing is not None:
            if isinstance(spacing, torch.Tensor):
                spacing = spacing.detach().numpy()
            assert spacing.shape == (pred_cov.shape[0], self.spatial_dims)
            # Make a diagonal matrix with the pixel spacing (Batch_size, spatial_size, spatial_size)
            # e.g. for 2D: [[s1, 0], [0, s2]], for 3D: [[s1, 0, 0], [0, s2, 0], [0, 0, s3]]
            spacing_matrix = np.stack(
                [np.diag(spacing[i]) for i in range(spacing.shape[0])], axis=0
            ).reshape((pred_cov.shape[0], 1, self.spatial_dims, self.spatial_dims))
            # Scale the covariance matrix by the pixel spacing
            pred_cov_scaled = (
                spacing_matrix @ pred_cov_scaled @ np.transpose(spacing_matrix, (0, 1, 3, 2))
            )
        eig = np.linalg.eigvalsh(pred_cov_scaled)
        return np.pi * np.prod(np.sqrt(eig), axis=-1)

    def predict(self, pred, pred_cov, confidence=0.95):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().numpy()
        if isinstance(pred_cov, torch.Tensor):
            pred_cov = pred_cov.detach().numpy()
        self.check_input(pred, pred_cov)
        return pred, self.bound(pred_cov, confidence)

    def predict_contour(self, pred, pred_cov, confidence=0.95, sample_points=200):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().numpy()
        if isinstance(pred_cov, torch.Tensor):
            pred_cov = pred_cov.detach().numpy()
        self.check_input(pred, pred_cov)

        if self.spatial_dims == 2:
            angles = np.linspace(0, 2 * np.pi, sample_points)
            directions = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
            directions = directions.reshape((1, -1, 1, 1, self.spatial_dims))
            error_bound = directions @ np.linalg.cholesky(self.bound(pred_cov, confidence)).reshape(
                (-1, 1, self.nb_landmarks, self.spatial_dims, self.spatial_dims)
            )
            return pred.reshape(
                (-1, 1, self.nb_landmarks, self.spatial_dims)
            ) + error_bound.reshape((-1, directions.shape[1], self.nb_landmarks, self.spatial_dims))
        elif self.spatial_dims == 3:
            eigenvalues, eigenvectors = np.linalg.eigh(self.bound(pred_cov, confidence))

            # Create a grid of points for a unit sphere
            u = np.linspace(0, 2 * np.pi, sample_points)
            v = np.linspace(0, np.pi, sample_points // 2)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones_like(u), np.cos(v))

            # Scale and rotate the unit sphere to match the ellipsoid
            radii = np.sqrt(eigenvalues)
            transform = eigenvectors @ np.stack(
                [
                    np.stack(
                        [radii[..., 0], np.zeros_like(radii[..., 0]), np.zeros_like(radii[..., 0])],
                        axis=-1,
                    ),
                    np.stack(
                        [np.zeros_like(radii[..., 0]), radii[..., 1], np.zeros_like(radii[..., 0])],
                        axis=-1,
                    ),
                    np.stack(
                        [np.zeros_like(radii[..., 0]), np.zeros_like(radii[..., 0]), radii[..., 2]],
                        axis=-1,
                    ),
                ],
                axis=-2,
            )
            ellipsoid = transform @ np.array([z.ravel(), y.ravel(), x.ravel()])
            # Reshape to 3D grid
            x_ellipsoid = ellipsoid[..., 2, :].reshape((-1, self.nb_landmarks, *x.shape))
            y_ellipsoid = ellipsoid[..., 1, :].reshape((-1, self.nb_landmarks, *y.shape))
            z_ellipsoid = ellipsoid[..., 0, :].reshape((-1, self.nb_landmarks, *z.shape))

            return np.stack([z_ellipsoid, y_ellipsoid, x_ellipsoid], axis=-1) + pred.reshape(
                (-1, self.nb_landmarks, 1, 1, self.spatial_dims)
            )
        else:
            raise ValueError("Unsupported spatial dimension")

    def in_region(self, pred, pred_cov, target, confidence=0.95):
        return mahalanobis_distance(pred, self.bound(pred_cov, confidence), target) <= 1

    def evaluate(self, pred, pred_cov, target, spacing=None, confidence=0.95, return_summary=True):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().numpy()
        if isinstance(pred_cov, torch.Tensor):
            pred_cov = pred_cov.detach().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().numpy()
        self.check_input(pred, pred_cov)
        assert pred.shape == target.shape
        error = euclidean_distance(
            pixel_to_unit_numpy(pred, pixel_spacing=spacing),
            pixel_to_unit_numpy(target, pixel_spacing=spacing),
        )
        area = self.area_prediction_region(pred_cov, spacing=spacing, confidence=confidence)
        in_region = self.in_region(pred, pred_cov, target, confidence=confidence)
        coverage = in_region.mean()
        efficiency_mean = area.mean()
        efficiency_median = np.median(area)
        efficiency_q1 = np.quantile(area, 0.25)
        efficiency_q3 = np.quantile(area, 0.75)
        adaptivity = spearmanr(error.flatten(), area.flatten()).statistic
        print(f"Coverage: {coverage}")
        print(f"Efficiency (mean): {efficiency_mean}")
        print(f"Efficiency (median): {efficiency_median}")
        print(f"Efficiency (Q1): {efficiency_q1}")
        print(f"Efficiency (Q3): {efficiency_q3}")
        print(f"Adeptivity: {adaptivity}")
        if not return_summary:
            return (in_region, area, error)
        return {
            "coverage": coverage,
            "efficiency_mean": efficiency_mean,
            "efficiency_median": efficiency_median,
            "efficiency_q1": efficiency_q1,
            "efficiency_q3": efficiency_q3,
            "adaptivity": adaptivity,
        }


class ConformalRegressorBonferroni:
    def __init__(
        self,
        spatial_dims=2,
        nb_landmarks=1,
    ):
        self.alphas = None
        self.spatial_dims = spatial_dims
        self.nb_landmarks = nb_landmarks

    def check_input(self, pred):
        if pred.ndim != 3:
            raise ValueError("pred must have shape (N, C, D)")
        if pred.shape[1] != self.nb_landmarks:
            raise ValueError("pred must have the same second dimension as nb_landmarks")
        if pred.shape[2] != self.spatial_dims:
            raise ValueError("pred must have the same third dimension as spatial_dims")

    def fit(self, pred, target, sigmas=None):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().numpy()
        self.check_input(pred)
        if sigmas is None:
            sigmas = np.ones_like(target)
        elif isinstance(sigmas, torch.Tensor):
            sigmas = sigmas.detach().numpy()
        if sigmas.shape != target.shape:
            raise ValueError("sigmas must have the same shape as target")
        self.alphas = []
        for d in range(self.spatial_dims):
            alphas = np.abs(target[:, :, d] - pred[:, :, d]) / sigmas[:, :, d]
            alphas = np.sort(alphas, axis=0)[::-1]
            self.alphas.append(alphas)

    def bound(self, sigmas=None, confidence=0.95):
        """Get the bound for the given confidence level."""
        if self.alphas is None:
            raise ValueError("You must call fit before calling predict")
        if sigmas is None:
            sigmas = np.ones((self.spatial_dims,))
        bonferroni_confidence = 1 - (1 - confidence) / self.spatial_dims
        return (
            np.stack(
                [
                    alphas[int((1 - bonferroni_confidence) * (len(alphas) + 1)) - 1]
                    for alphas in self.alphas
                ],
                axis=-1,
            ).reshape((1, -1, self.spatial_dims))
            * sigmas
        )

    def area_prediction_region(self, sigmas=None, spacing=None, confidence=0.95):
        if spacing is None:
            return np.prod(self.bound(sigmas, confidence) * 2, axis=-1)
        return np.prod(self.bound(sigmas, confidence) * 2, axis=-1) * np.prod(
            spacing, axis=-1
        ).reshape(-1, 1)

    def predict(self, pred, sigma=None, confidence=0.95):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().numpy()
        return pred + self.bound(sigma, confidence) * np.array([-1, 1])

    def predict_contour(self, pred, sigmas, confidence=0.95, sample_points=200):
        # Convert torch tensor to numpy if needed
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().numpy()

        # Calculate error bounds - shape: (batch_size, nb_landmarks, spatial_dims)
        bounds = self.bound(sigmas, confidence)

        batch_size = pred.shape[0]

        if self.spatial_dims == 2:
            line_points = sample_points // 4

            # Initialize output arrays with the new shape
            x_coords = np.zeros((batch_size, line_points * 4, self.nb_landmarks))
            y_coords = np.zeros((batch_size, line_points * 4, self.nb_landmarks))

            # Generate contours for each batch and landmark
            for b in range(batch_size):
                for i in range(self.nb_landmarks):
                    # Get center point and bounds for current landmark
                    center_y = pred[b, i, 0]  # y coordinate
                    center_x = pred[b, i, 1]  # x coordinate
                    bound_y = bounds[b, i, 0]  # y bound
                    bound_x = bounds[b, i, 1]  # x bound

                    # Generate points for each edge
                    x_points = np.linspace(-bound_x, bound_x, line_points)
                    y_points = np.linspace(-bound_y, bound_y, line_points)

                    # Create rectangular contour
                    x_contour = np.concatenate(
                        [
                            center_x + x_points,  # Top edge
                            center_x + bound_x * np.ones(line_points),  # Right edge
                            center_x + x_points[::-1],  # Bottom edge
                            center_x - bound_x * np.ones(line_points),  # Left edge
                        ]
                    )

                    y_contour = np.concatenate(
                        [
                            center_y + bound_y * np.ones(line_points),  # Top edge
                            center_y + y_points,  # Right edge
                            center_y - bound_y * np.ones(line_points),  # Bottom edge
                            center_y + y_points[::-1],  # Left edge
                        ]
                    )

                    # Store the contour points in the new shape format
                    x_coords[b, :, i] = x_contour
                    y_coords[b, :, i] = y_contour

            return np.stack([y_coords, x_coords], axis=-1)

        elif self.spatial_dims == 3:
            bounds = self.bound(sigmas=sigmas, confidence=confidence)

            offsets = np.array(
                [
                    [-1, -1, -1],  # Vertex 0
                    [1, -1, -1],  # Vertex 1
                    [1, 1, -1],  # Vertex 2
                    [-1, 1, -1],  # Vertex 3
                    [-1, -1, 1],  # Vertex 4
                    [1, -1, 1],  # Vertex 5
                    [1, 1, 1],  # Vertex 6
                    [-1, 1, 1],  # Vertex 7
                ]
            )

            # Expand dimensions for broadcasting
            offsets = offsets[np.newaxis, np.newaxis, :, :]

            # Generate all vertices by scaling offsets with bounds and adding center
            vertices = pred[:, :, np.newaxis, :] + bounds[:, :, np.newaxis, :] * offsets

            # Define the face connectivity in terms of vertex indices
            face_indices = np.array(
                [
                    [0, 1, 5, 4],  # Bottom face
                    [2, 3, 7, 6],  # Top face
                    [0, 4, 7, 3],  # Left face
                    [1, 5, 6, 2],  # Right face
                    [0, 1, 2, 3],  # Front face
                    [4, 5, 6, 7],  # Back face
                ]
            )
            return vertices, face_indices
        else:
            raise ValueError("Unsupported spatial dimension")

    def in_region(self, pred, sigmas, target, confidence=0.95):
        return np.all(np.abs(target - pred) <= self.bound(sigmas, confidence), axis=-1)

    def evaluate(self, pred, sigmas, target, spacing=None, confidence=0.95, return_summary=True):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().numpy()
        if isinstance(sigmas, torch.Tensor):
            sigmas = sigmas.detach().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().numpy()
        self.check_input(pred)
        assert pred.shape == target.shape
        error = euclidean_distance(
            pixel_to_unit_numpy(pred, pixel_spacing=spacing),
            pixel_to_unit_numpy(target, pixel_spacing=spacing),
        )
        area = self.area_prediction_region(sigmas, spacing=spacing, confidence=confidence)
        in_region = self.in_region(pred, sigmas, target, confidence=confidence)
        coverage = in_region.mean()
        efficiency_mean = area.mean()
        efficiency_median = np.median(area)
        efficiency_q1 = np.quantile(area, 0.25)
        efficiency_q3 = np.quantile(area, 0.75)
        adaptivity = spearmanr(error.flatten(), area.flatten()).statistic
        print(f"Coverage: {coverage}")
        print(f"Efficiency (mean): {efficiency_mean}")
        print(f"Efficiency (median): {efficiency_median}")
        print(f"Efficiency (Q1): {efficiency_q1}")
        print(f"Efficiency (Q3): {efficiency_q3}")
        print(f"Adeptivity: {adaptivity}")
        if not return_summary:
            return (in_region, area, error)
        return {
            "coverage": coverage,
            "efficiency_mean": efficiency_mean,
            "efficiency_median": efficiency_median,
            "efficiency_q1": efficiency_q1,
            "efficiency_q3": efficiency_q3,
            "adaptivity": adaptivity,
        }


class ConformalRegressorMaxNonconformity:
    def __init__(
        self,
        spatial_dims=2,
        nb_landmarks=1,
    ):
        self.spatial_dims = spatial_dims
        self.nb_landmarks = nb_landmarks
        self.alphas = None

    def check_input(self, pred):
        if pred.ndim != 3:
            raise ValueError("pred must have shape (N, C, D)")
        if pred.shape[1] != self.nb_landmarks:
            raise ValueError("pred must have the same second dimension as nb_landmarks")
        if pred.shape[2] != self.spatial_dims:
            raise ValueError("pred must have the same third dimension as spatial_dims")

    def fit(self, pred, target, sigmas=None):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().numpy()
        self.check_input(pred)
        if sigmas is None:
            sigmas = np.ones_like(target)
        elif isinstance(sigmas, torch.Tensor):
            sigmas = sigmas.detach().numpy()
        if sigmas.shape != target.shape:
            raise ValueError("sigmas must have the same shape as target")
        self.alphas = absolute_error_max(pred, target, sigmas)
        self.alphas = np.sort(self.alphas, axis=0)[::-1]

    def bound(self, sigmas=None, confidence=0.95):
        """Get the bound for the given confidence level."""
        if self.alphas is None:
            raise ValueError("You must call fit before calling predict")
        if sigmas is None:
            sigmas = np.ones((self.spatial_dims,))
        return (
            self.alphas[int((1 - confidence) * (len(self.alphas) + 1)) - 1].reshape((1, -1, 1))
            * sigmas
        )

    def area_prediction_region(self, sigmas=None, spacing=None, confidence=0.95):
        if spacing is None:
            return np.prod(self.bound(sigmas, confidence) * 2, axis=-1)
        return np.prod(self.bound(sigmas, confidence) * 2, axis=-1) * np.prod(
            spacing, axis=-1
        ).reshape(-1, 1)

    def predict(self, pred, sigma=None, confidence=0.95):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().numpy()
        return pred + self.bound(sigma, confidence) * np.array([-1, 1])

    def predict_contour(self, pred, sigmas, confidence=0.95, sample_points=200):
        # Convert torch tensor to numpy if needed
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().numpy()

        # Calculate error bounds - shape: (batch_size, nb_landmarks, spatial_dims)
        bounds = self.bound(sigmas, confidence)

        batch_size = pred.shape[0]

        if self.spatial_dims == 2:
            line_points = sample_points // 4

            # Initialize output arrays with the new shape
            x_coords = np.zeros((batch_size, line_points * 4, self.nb_landmarks))
            y_coords = np.zeros((batch_size, line_points * 4, self.nb_landmarks))

            # Generate contours for each batch and landmark
            for b in range(batch_size):
                for i in range(self.nb_landmarks):
                    # Get center point and bounds for current landmark
                    center_y = pred[b, i, 0]  # y coordinate
                    center_x = pred[b, i, 1]  # x coordinate
                    bound_y = bounds[b, i, 0]  # y bound
                    bound_x = bounds[b, i, 1]  # x bound

                    # Generate points for each edge
                    x_points = np.linspace(-bound_x, bound_x, line_points)
                    y_points = np.linspace(-bound_y, bound_y, line_points)

                    # Create rectangular contour
                    x_contour = np.concatenate(
                        [
                            center_x + x_points,  # Top edge
                            center_x + bound_x * np.ones(line_points),  # Right edge
                            center_x + x_points[::-1],  # Bottom edge
                            center_x - bound_x * np.ones(line_points),  # Left edge
                        ]
                    )

                    y_contour = np.concatenate(
                        [
                            center_y + bound_y * np.ones(line_points),  # Top edge
                            center_y + y_points,  # Right edge
                            center_y - bound_y * np.ones(line_points),  # Bottom edge
                            center_y + y_points[::-1],  # Left edge
                        ]
                    )

                    # Store the contour points in the new shape format
                    x_coords[b, :, i] = x_contour
                    y_coords[b, :, i] = y_contour

            return np.stack([y_coords, x_coords], axis=-1)

        elif self.spatial_dims == 3:
            bounds = self.bound(sigmas=sigmas, confidence=confidence)

            offsets = np.array(
                [
                    [-1, -1, -1],  # Vertex 0
                    [1, -1, -1],  # Vertex 1
                    [1, 1, -1],  # Vertex 2
                    [-1, 1, -1],  # Vertex 3
                    [-1, -1, 1],  # Vertex 4
                    [1, -1, 1],  # Vertex 5
                    [1, 1, 1],  # Vertex 6
                    [-1, 1, 1],  # Vertex 7
                ]
            )

            # Expand dimensions for broadcasting
            offsets = offsets[np.newaxis, np.newaxis, :, :]

            # Generate all vertices by scaling offsets with bounds and adding center
            vertices = pred[:, :, np.newaxis, :] + bounds[:, :, np.newaxis, :] * offsets

            # Define the face connectivity in terms of vertex indices
            face_indices = np.array(
                [
                    [0, 1, 5, 4],  # Bottom face
                    [2, 3, 7, 6],  # Top face
                    [0, 4, 7, 3],  # Left face
                    [1, 5, 6, 2],  # Right face
                    [0, 1, 2, 3],  # Front face
                    [4, 5, 6, 7],  # Back face
                ]
            )
            return vertices, face_indices
        else:
            raise ValueError("Unsupported spatial dimension")

    def in_region(self, pred, sigmas, target, confidence=0.95):
        return np.all(np.abs(target - pred) <= self.bound(sigmas, confidence), axis=-1)

    def evaluate(self, pred, sigmas, target, spacing=None, confidence=0.95, return_summary=True):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().numpy()
        if isinstance(sigmas, torch.Tensor):
            sigmas = sigmas.detach().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().numpy()
        self.check_input(pred)
        assert pred.shape == target.shape
        error = euclidean_distance(
            pixel_to_unit_numpy(pred, pixel_spacing=spacing),
            pixel_to_unit_numpy(target, pixel_spacing=spacing),
        )
        area = self.area_prediction_region(sigmas, spacing=spacing, confidence=confidence)
        in_region = self.in_region(pred, sigmas, target, confidence=confidence)
        coverage = in_region.mean()
        efficiency_mean = area.mean()
        efficiency_median = np.median(area)
        efficiency_q1 = np.quantile(area, 0.25)
        efficiency_q3 = np.quantile(area, 0.75)
        adaptivity = spearmanr(error.flatten(), area.flatten()).statistic
        print(f"Coverage: {coverage}")
        print(f"Efficiency (mean): {efficiency_mean}")
        print(f"Efficiency (median): {efficiency_median}")
        print(f"Efficiency (Q1): {efficiency_q1}")
        print(f"Efficiency (Q3): {efficiency_q3}")
        print(f"Adeptivity: {adaptivity}")
        if not return_summary:
            return (in_region, area, error)
        return {
            "coverage": coverage,
            "efficiency_mean": efficiency_mean,
            "efficiency_median": efficiency_median,
            "efficiency_q1": efficiency_q1,
            "efficiency_q3": efficiency_q3,
            "adaptivity": adaptivity,
        }


class MR2CCP:

    def __init__(
        self,
        spatial_dims=2,
        nb_landmarks=1,
        heatmap_size=(512, 512),
    ):
        self.spatial_dims = spatial_dims
        self.nb_landmarks = nb_landmarks
        self.heatmap_size = heatmap_size
        self.alphas = None

    def check_input(self, heatmaps):
        if heatmaps.ndim != 4 and self.spatial_dims == 2:
            raise ValueError("pred must have shape (N, C, H, W)")
        elif heatmaps.ndim != 5 and self.spatial_dims == 3:
            raise ValueError("pred must have shape (N, C, D, H, W)")
        elif heatmaps.shape[1] != self.nb_landmarks:
            raise ValueError("pred must have the same second dimension as nb_landmarks")
        elif heatmaps.shape[2:] != self.heatmap_size:
            raise ValueError(
                f"pred {heatmaps.shape[2:]} must have the same dimensions as heatmap_size {self.heatmap_size}"
            )

    def fit(self, heatmaps, target, original_dims=None, padding=None):
        if isinstance(heatmaps, torch.Tensor):
            heatmaps = heatmaps.detach().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().numpy()
        self.check_input(heatmaps)
        alphas = -self.conditional_probability(
            heatmaps=heatmaps, target=target, original_dims=original_dims, padding=padding
        )
        self.alphas = np.sort(alphas, axis=0)[::-1]

    def conditional_probability(self, heatmaps, target, original_dims=None, padding=None):
        resized_target = resize_landmarks(
            target, dim=np.array(self.heatmap_size), dim_orig=original_dims, padding=padding
        )
        # clip the target to the heatmap size
        resized_target = np.clip(resized_target, 0, np.array(self.heatmap_size) - 1)
        if self.spatial_dims == 2:
            y_min = np.floor(resized_target[..., 0]).astype(int)
            x_min = np.floor(resized_target[..., 1]).astype(int)
            y_max = np.ceil(resized_target[..., 0]).astype(int)
            x_max = np.ceil(resized_target[..., 1]).astype(int)

            prob_y_min_x_min = heatmaps[
                np.arange(heatmaps.shape[0])[:, None],
                np.arange(heatmaps.shape[1]),
                y_min.astype(int),
                x_min.astype(int),
            ]
            prob_y_min_x_max = heatmaps[
                np.arange(heatmaps.shape[0])[:, None],
                np.arange(heatmaps.shape[1]),
                y_min.astype(int),
                x_max.astype(int),
            ]
            prob_y_max_x_min = heatmaps[
                np.arange(heatmaps.shape[0])[:, None],
                np.arange(heatmaps.shape[1]),
                y_max.astype(int),
                x_min.astype(int),
            ]
            prob_y_max_x_max = heatmaps[
                np.arange(heatmaps.shape[0])[:, None],
                np.arange(heatmaps.shape[1]),
                y_max.astype(int),
                x_max.astype(int),
            ]

            diff_y_max = y_max - resized_target[..., 0]
            diff_x_max = x_max - resized_target[..., 1]
            diff_y_min = resized_target[..., 0] - y_min
            diff_x_min = resized_target[..., 1] - x_min

            # assign at indicecs y_max and y_min are the same diff_y_max and diff_y_min 0.5 to avoid division by zero
            idx_y_max_y_min = np.where(y_max == y_min)
            diff_y_max[idx_y_max_y_min] = 0.5
            diff_y_min[idx_y_max_y_min] = 0.5

            idx_x_max_x_min = np.where(x_max == x_min)
            diff_x_max[idx_x_max_x_min] = 0.5
            diff_x_min[idx_x_max_x_min] = 0.5

            weights_y_min_x_min = diff_y_max * diff_x_max
            weights_y_min_x_max = diff_y_max * diff_x_min
            weights_y_max_x_min = diff_y_min * diff_x_max
            weights_y_max_x_max = diff_y_min * diff_x_min

            prob = (
                prob_y_min_x_min * weights_y_min_x_min
                + prob_y_min_x_max * weights_y_min_x_max
                + prob_y_max_x_min * weights_y_max_x_min
                + prob_y_max_x_max * weights_y_max_x_max
            )
        elif self.spatial_dims == 3:
            z_min = np.floor(resized_target[..., 0])
            y_min = np.floor(resized_target[..., 1])
            x_min = np.floor(resized_target[..., 2])
            z_max = np.ceil(resized_target[..., 0])
            y_max = np.ceil(resized_target[..., 1])
            x_max = np.ceil(resized_target[..., 2])

            prob_z_min_y_min_x_min = heatmaps[
                np.arange(heatmaps.shape[0])[:, None],
                np.arange(heatmaps.shape[1]),
                z_min.astype(int),
                y_min.astype(int),
                x_min.astype(int),
            ]
            prob_z_min_y_min_x_max = heatmaps[
                np.arange(heatmaps.shape[0])[:, None],
                np.arange(heatmaps.shape[1]),
                z_min.astype(int),
                y_min.astype(int),
                x_max.astype(int),
            ]
            prob_z_min_y_max_x_min = heatmaps[
                np.arange(heatmaps.shape[0])[:, None],
                np.arange(heatmaps.shape[1]),
                z_min.astype(int),
                y_max.astype(int),
                x_min.astype(int),
            ]
            prob_z_min_y_max_x_max = heatmaps[
                np.arange(heatmaps.shape[0])[:, None],
                np.arange(heatmaps.shape[1]),
                z_min.astype(int),
                y_max.astype(int),
                x_max.astype(int),
            ]
            prob_z_max_y_min_x_min = heatmaps[
                np.arange(heatmaps.shape[0])[:, None],
                np.arange(heatmaps.shape[1]),
                z_max.astype(int),
                y_min.astype(int),
                x_min.astype(int),
            ]
            prob_z_max_y_min_x_max = heatmaps[
                np.arange(heatmaps.shape[0])[:, None],
                np.arange(heatmaps.shape[1]),
                z_max.astype(int),
                y_min.astype(int),
                x_max.astype(int),
            ]
            prob_z_max_y_max_x_min = heatmaps[
                np.arange(heatmaps.shape[0])[:, None],
                np.arange(heatmaps.shape[1]),
                z_max.astype(int),
                y_max.astype(int),
                x_min.astype(int),
            ]
            prob_z_max_y_max_x_max = heatmaps[
                np.arange(heatmaps.shape[0])[:, None],
                np.arange(heatmaps.shape[1]),
                z_max.astype(int),
                y_max.astype(int),
                x_max.astype(int),
            ]

            diff_z_max = z_max - resized_target[..., 0]
            diff_y_max = y_max - resized_target[..., 1]
            diff_x_max = x_max - resized_target[..., 2]
            diff_z_min = resized_target[..., 0] - z_min
            diff_y_min = resized_target[..., 1] - y_min
            diff_x_min = resized_target[..., 2] - x_min

            # assign at indicecs z_max and z_min are the same diff_z_max and diff_z_min 0.5 to avoid division by zero
            idx_z_max_z_min = np.where(z_max == z_min)
            diff_z_max[idx_z_max_z_min] = 0.5
            diff_z_min[idx_z_max_z_min] = 0.5

            idx_y_max_y_min = np.where(y_max == y_min)
            diff_y_max[idx_y_max_y_min] = 0.5
            diff_y_min[idx_y_max_y_min] = 0.5

            idx_x_max_x_min = np.where(x_max == x_min)
            diff_x_max[idx_x_max_x_min] = 0.5
            diff_x_min[idx_x_max_x_min] = 0.5

            weights_z_min_y_min_x_min = diff_z_max * diff_y_max * diff_x_max
            weights_z_min_y_min_x_max = diff_z_max * diff_y_max * diff_x_min
            weights_z_min_y_max_x_min = diff_z_max * diff_y_min * diff_x_max
            weights_z_min_y_max_x_max = diff_z_max * diff_y_min * diff_x_min
            weights_z_max_y_min_x_min = diff_z_min * diff_y_max * diff_x_max
            weights_z_max_y_min_x_max = diff_z_min * diff_y_max * diff_x_min
            weights_z_max_y_max_x_min = diff_z_min * diff_y_min * diff_x_max
            weights_z_max_y_max_x_max = diff_z_min * diff_y_min * diff_x_min
            prob = (
                prob_z_min_y_min_x_min * weights_z_min_y_min_x_min
                + prob_z_min_y_min_x_max * weights_z_min_y_min_x_max
                + prob_z_min_y_max_x_min * weights_z_min_y_max_x_min
                + prob_z_min_y_max_x_max * weights_z_min_y_max_x_max
                + prob_z_max_y_min_x_min * weights_z_max_y_min_x_min
                + prob_z_max_y_min_x_max * weights_z_max_y_min_x_max
                + prob_z_max_y_max_x_min * weights_z_max_y_max_x_min
                + prob_z_max_y_max_x_max * weights_z_max_y_max_x_max
            )
        return prob

    def bound(self, confidence=0.95):
        """Get the bound for the given confidence level."""
        if self.alphas is None:
            raise ValueError("You must call fit before calling predict")
        return self.alphas[int((1 - confidence) * (len(self.alphas) + 1)) - 1]

    def area_prediction_region(
        self, heatmaps, original_dims=None, padding=None, spacing=None, confidence=0.95
    ):
        area = np.zeros((heatmaps.shape[0], heatmaps.shape[1]))
        print("Alert: area caluculation is a lower bound")
        for i in tqdm(range(heatmaps.shape[0])):
            pred_region = self.predict(
                heatmaps[i : i + 1],
                original_dims=original_dims[i : i + 1],
                padding=padding[i : i + 1],
                confidence=confidence,
            )
            if spacing is None:
                if self.spatial_dims == 2:
                    area[i] = np.sum(pred_region, axis=(-1, -2))
                elif self.spatial_dims == 3:
                    area[i] = np.sum(pred_region, axis=(-1, -2, -3))
            if self.spatial_dims == 2:
                area[i] = np.sum(pred_region, axis=(-1, -2)) * np.prod(spacing[i], axis=-1)
            elif self.spatial_dims == 3:
                area[i] = np.sum(pred_region, axis=(-1, -2, -3)) * np.prod(spacing[i], axis=-1)
        return area

    def predict(self, heatmaps, original_dims=None, padding=None, confidence=0.95):
        if isinstance(heatmaps, torch.Tensor):
            heatmaps = heatmaps.detach().numpy()
        # Rescale the prediction region to the original dimensions
        mode = "bilinear" if self.spatial_dims == 2 else "trilinear"
        heatmaps = transform_heatmap_to_original_size_numpy(
            heatmaps, padding=padding, original_dim=original_dims, mode=mode
        )
        if self.spatial_dims == 2:
            return (
                -heatmaps <= self.bound(confidence).reshape((1, heatmaps.shape[1], 1, 1))
            ).astype(int)
        return (
            -heatmaps <= self.bound(confidence).reshape((1, heatmaps.shape[1], 1, 1, 1))
        ).astype(int)

    def predict_contour(self, heatmaps, original_dims=None, padding=None, confidence=0.95):
        if isinstance(heatmaps, torch.Tensor):
            heatmaps = heatmaps.detach().numpy()
        print("Alert: contour caluculation is a lower bound")
        output = []
        for i in tqdm(range(heatmaps.shape[0])):
            if original_dims is None or padding is None:
                pred_region = self.predict(heatmaps[i : i + 1], confidence=confidence)[0]
            else:
                pred_region = self.predict(
                    heatmaps[i : i + 1],
                    original_dims=original_dims[i : i + 1],
                    padding=padding[i : i + 1],
                    confidence=confidence,
                )[0]
            output_sample = []
            for j in range(pred_region.shape[0]):
                pred_region_sample = pred_region[j]
                if self.spatial_dims == 2:
                    contour, _ = cv2.findContours(
                        pred_region_sample.astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    output_sample.append(contour)
                elif self.spatial_dims == 3:
                    verts, faces, _, _ = measure.marching_cubes(
                        pred_region_sample, level=0.5
                    )  # returns verts, faces, normals, values
                    output_sample.append((verts, faces))
            output.append(output_sample)
        return output

    def in_region(
        self, heatmaps, target, original_dims=None, padding=None, confidence=0.95, batch_size=1
    ):
        return -self.conditional_probability(
            heatmaps=heatmaps, target=target, original_dims=original_dims, padding=padding
        ) <= self.bound(confidence)

    def evaluate(
        self,
        heatmaps,
        pred_landmarks,
        target,
        original_dims=None,
        padding=None,
        spacing=None,
        confidence=0.95,
        batch_size=1,
        return_summary=True,
    ):
        if isinstance(heatmaps, torch.Tensor):
            heatmaps = heatmaps.detach().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().numpy()
        self.check_input(heatmaps)
        nb_batch = heatmaps.shape[0] // batch_size
        if heatmaps.shape[0] % batch_size != 0:
            nb_batch += 1
        error = euclidean_distance(
            pixel_to_unit_numpy(pred_landmarks, pixel_spacing=spacing),
            pixel_to_unit_numpy(target, pixel_spacing=spacing),
        )
        area = np.zeros((heatmaps.shape[0], heatmaps.shape[1]))

        for i in tqdm(range(0, heatmaps.shape[0], batch_size)):
            batch_pred = heatmaps[i : i + batch_size]
            batch_original_dims = (
                original_dims[i : i + batch_size] if original_dims is not None else None
            )
            batch_padding = padding[i : i + batch_size] if padding is not None else None
            batch_spacing = spacing[i : i + batch_size] if spacing is not None else None

            batch_pred_region = self.predict(
                batch_pred,
                original_dims=batch_original_dims,
                padding=batch_padding,
                confidence=confidence,
            )
            if self.spatial_dims == 2:
                batch_area = np.sum(batch_pred_region, axis=(-1, -2)) * np.prod(
                    batch_spacing, axis=-1
                )
            elif self.spatial_dims == 3:
                batch_area = np.sum(batch_pred_region, axis=(-1, -2, -3)) * np.prod(
                    batch_spacing, axis=-1
                )

            area[i : i + batch_size] = batch_area
        in_region = self.in_region(heatmaps, target, original_dims, padding, confidence)
        coverage = in_region.mean()
        efficiency_mean = area.mean()
        efficiency_median = np.median(area)
        efficiency_q1 = np.quantile(area, 0.25)
        efficiency_q3 = np.quantile(area, 0.75)
        adaptivity = spearmanr(error.flatten(), area.flatten()).statistic
        print(f"Coverage: {coverage}")
        print(f"Efficiency (mean): {efficiency_mean}")
        print(f"Efficiency (median): {efficiency_median}")
        print(f"Efficiency (Q1): {efficiency_q1}")
        print(f"Efficiency (Q3): {efficiency_q3}")
        print(f"Adeptivity: {adaptivity}")
        if not return_summary:
            return (in_region, area, error)
        return {
            "coverage": coverage,
            "efficiency_mean": efficiency_mean,
            "efficiency_median": efficiency_median,
            "efficiency_q1": efficiency_q1,
            "efficiency_q3": efficiency_q3,
            "adaptivity": adaptivity,
        }


class MR2C2R:

    def __init__(
        self,
        spatial_dims=2,
        nb_landmarks=1,
        heatmap_size=(512, 512),
        aps=False,
    ):
        self.spatial_dims = spatial_dims
        self.nb_landmarks = nb_landmarks
        self.heatmap_size = heatmap_size
        self.aps = aps
        self.alphas = None

    def check_input(self, heatmaps):
        if heatmaps.ndim != 4 and self.spatial_dims == 2:
            raise ValueError("pred must have shape (N, C, H, W)")
        elif heatmaps.ndim != 5 and self.spatial_dims == 3:
            raise ValueError("pred must have shape (N, C, D, H, W)")
        elif heatmaps.shape[1] != self.nb_landmarks:
            raise ValueError("pred must have the same second dimension as nb_landmarks")
        elif heatmaps.shape[2:] != self.heatmap_size:
            raise ValueError(
                f"pred {heatmaps.shape[2:]} must have the same dimensions as heatmap_size {self.heatmap_size}"
            )

    def fit(self, heatmaps, target, original_dims=None, padding=None, confidence=0.95):
        if isinstance(heatmaps, torch.Tensor):
            heatmaps = heatmaps.detach().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().numpy()
        self.check_input(heatmaps)
        if not self.aps:
            self.alphas = -self.conditional_probability(heatmaps, target, original_dims, padding)
            self.alphas = np.sort(self.alphas, axis=0)[::-1]
        else:
            self.fit_aps(heatmaps, target, original_dims, padding, confidence=confidence)

    def fit_aps(self, heatmaps, target, original_dims=None, padding=None, confidence=0.95):
        resized_target = resize_landmarks(
            target, dim=np.array(self.heatmap_size), dim_orig=original_dims, padding=padding
        )
        resized_target = np.floor(resized_target).astype(int)
        heatmaps_flatten = heatmaps.reshape((heatmaps.shape[0], heatmaps.shape[1], -1))
        if self.spatial_dims == 2:
            resized_target_flatten = (
                resized_target[..., 0] * heatmaps.shape[-1] + resized_target[..., 1]
            ).astype(int)
        else:
            resized_target_flatten = (
                resized_target[..., 0] * heatmaps.shape[-2] * heatmaps.shape[-1]
                + resized_target[..., 1] * heatmaps.shape[-1]
                + resized_target[..., 2]
            ).astype(int)
        idxs_sorted_heatmaps = np.argsort(heatmaps_flatten, axis=-1)[:, :, ::-1]
        sorted_heatmaps = np.take_along_axis(heatmaps_flatten, idxs_sorted_heatmaps, axis=-1)
        cum_sum_heatmaps = np.cumsum(sorted_heatmaps, axis=-1)

        if self.alphas is None:
            self.alphas = {}
        elif not isinstance(self.alphas, dict):
            raise ValueError("Method already fitted with non-aps method")
        self.alphas[confidence] = []
        for j in range(heatmaps_flatten.shape[1]):
            # map the target to the index of the sorted heatmaps
            idx_target = np.where(resized_target_flatten[:, j, None] == idxs_sorted_heatmaps[:, j])[
                1
            ]
            cum_sum_target = cum_sum_heatmaps[np.arange(cum_sum_heatmaps.shape[0]), j, idx_target]
            scores = cum_sum_target - confidence
            level_adjusted = confidence * (1 + 1 / len(scores))
            correction = mquantiles(scores, prob=level_adjusted)
            corrected_confidence = np.minimum(confidence + correction, 1.0)
            self.alphas[confidence].append(corrected_confidence)

    def conditional_probability(self, heatmaps, target, original_dims=None, padding=None):
        resized_target = resize_landmarks(
            target, dim=np.array(self.heatmap_size), dim_orig=original_dims, padding=padding
        )
        min_idxs = np.floor(resized_target).astype(int)

        if self.spatial_dims == 2:
            y_min = min_idxs[:, :, 0]
            x_min = min_idxs[:, :, 1]
            prob_y_min_x_min = heatmaps[
                np.arange(heatmaps.shape[0])[:, None], np.arange(heatmaps.shape[1]), y_min, x_min
            ]
            prob = prob_y_min_x_min

        elif self.spatial_dims == 3:
            z_min = min_idxs[:, :, 0]
            y_min = min_idxs[:, :, 1]
            x_min = min_idxs[:, :, 2]
            prob_z_min_y_min_x_min = heatmaps[
                np.arange(heatmaps.shape[0])[:, None],
                np.arange(heatmaps.shape[1]),
                z_min,
                y_min,
                x_min,
            ]
            prob = prob_z_min_y_min_x_min

        else:
            raise ValueError("Unsupported spatial dimension")

        return prob

    def bound(self, confidence=0.95):
        """Get the bound for the given confidence level."""
        if self.alphas is None:
            raise ValueError("You must call fit before calling predict")
        elif not self.aps:
            return self.alphas[int((1 - confidence) * (len(self.alphas) + 1)) - 1]
        elif confidence not in self.alphas.keys():
            raise ValueError("Confidence level not fitted")
        return self.alphas[confidence]

    def area_prediction_region(
        self, heatmaps, original_dims=None, padding=None, spacing=None, confidence=0.95
    ):
        area = np.zeros((heatmaps.shape[0], heatmaps.shape[1]))
        for i in tqdm(range(heatmaps.shape[0])):
            pred_region = self.predict(
                heatmaps[i : i + 1],
                original_dims=original_dims[i : i + 1],
                padding=padding[i : i + 1],
                confidence=confidence,
            )
            if spacing is None:
                if self.spatial_dims == 2:
                    area[i] = np.sum(pred_region, axis=(-1, -2))
                elif self.spatial_dims == 3:
                    area[i] = np.sum(pred_region, axis=(-1, -2, -3))
            if self.spatial_dims == 2:
                area[i] = np.sum(pred_region, axis=(-1, -2)) * np.prod(spacing[i], axis=-1)
            elif self.spatial_dims == 3:
                area[i] = np.sum(pred_region, axis=(-1, -2, -3)) * np.prod(spacing[i], axis=-1)
        return area

    def predict(self, heatmaps, original_dims=None, padding=None, confidence=0.95):
        if self.aps:
            return self.predict_aps(heatmaps, original_dims, padding, confidence)
        return self.predict_non_aps(heatmaps, original_dims, padding, confidence)

    def predict_non_aps(self, heatmaps, original_dims=None, padding=None, confidence=0.95):
        if isinstance(heatmaps, torch.Tensor):
            heatmaps = heatmaps.detach().numpy()
        if original_dims is None:
            if self.spatial_dims == 2:
                return (
                    -heatmaps <= self.bound(confidence).reshape((1, heatmaps.shape[1], 1, 1))
                ).astype(int)
            elif self.spatial_dims == 3:
                return (
                    -heatmaps <= self.bound(confidence).reshape((1, heatmaps.shape[1], 1, 1, 1))
                ).astype(int)
        if padding is None:
            padding = np.zeros((heatmaps.shape[0], self.spatial_dims))
        if self.spatial_dims == 2:
            pred_region = -heatmaps <= self.bound(confidence).reshape((1, heatmaps.shape[1], 1, 1))
        elif self.spatial_dims == 3:
            pred_region = -heatmaps <= self.bound(confidence).reshape(
                (1, heatmaps.shape[1], 1, 1, 1)
            )
        # Rescale the prediction region to the original dimensions
        mode = "nearest-exact"
        pred_region = transform_heatmap_to_original_size_numpy(
            pred_region.astype(float), padding=padding, original_dim=original_dims, mode=mode
        )
        return (pred_region > 0).astype(int)

    def predict_aps(self, heatmaps, original_dims=None, padding=None, confidence=0.95):
        if isinstance(heatmaps, torch.Tensor):
            heatmaps = heatmaps.detach().numpy()
        corrected_confidences = self.bound(confidence)
        # clone heatmaps
        heatmaps = np.copy(heatmaps)
        for i in range(heatmaps.shape[0]):
            for j in range(heatmaps.shape[1]):
                idxs_sorted_heatmaps = np.argsort(heatmaps[i, j].flatten())[::-1]
                cum_sum_sorted_heatmaps = np.cumsum(heatmaps[i, j].flatten()[idxs_sorted_heatmaps])
                if corrected_confidences[j] >= len(cum_sum_sorted_heatmaps) / (
                    len(cum_sum_sorted_heatmaps) + 1
                ):
                    threshold = len(cum_sum_sorted_heatmaps)
                else:
                    threshold = np.argmax(cum_sum_sorted_heatmaps >= corrected_confidences[j])
                heatmaps[i, j] = np.zeros_like(heatmaps[i, j])
                flatten_heatmaps = heatmaps[i, j].flatten()
                flatten_heatmaps[idxs_sorted_heatmaps[:threshold]] = 1
                heatmaps[i, j] = flatten_heatmaps.reshape(heatmaps[i, j].shape)

        if original_dims is None:
            return heatmaps
        # Rescale the prediction region to the original dimensions
        mode = "nearest-exact"
        heatmap = transform_heatmap_to_original_size_numpy(
            heatmaps.astype(float), padding=padding, original_dim=original_dims, mode=mode
        )
        return (heatmap > 0).astype(int)

    def predict_contour(self, heatmaps, original_dims=None, padding=None, confidence=0.95):
        if isinstance(heatmaps, torch.Tensor):
            heatmaps = heatmaps.detach().numpy()
        output = []
        for i in tqdm(range(heatmaps.shape[0])):
            if original_dims is None or padding is None:
                pred_region = self.predict(heatmaps[i : i + 1], confidence=confidence)[0]
            else:
                pred_region = self.predict(
                    heatmaps[i : i + 1],
                    original_dims=original_dims[i : i + 1],
                    padding=padding[i : i + 1],
                    confidence=confidence,
                )[0]
            output_sample = []
            for j in range(pred_region.shape[0]):
                pred_region_sample = pred_region[j]
                if self.spatial_dims == 2:
                    contour, _ = cv2.findContours(
                        pred_region_sample.astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    output_sample.append(contour)
                elif self.spatial_dims == 3:
                    verts, faces, _, _ = measure.marching_cubes(
                        pred_region_sample, level=0.5
                    )  # returns verts, faces, normals, values
                    output_sample.append((verts, faces))
            output.append(output_sample)
        return output

    def in_region(
        self, heatmaps, target, original_dims=None, padding=None, confidence=0.95, batch_size=1
    ):
        output = []
        for i in tqdm(range(0, heatmaps.shape[0], batch_size)):
            pred_region = self.predict(
                heatmaps[i : i + 1],
                original_dims=original_dims[i : i + 1],
                padding=padding[i : i + 1],
                confidence=confidence,
            )
            if self.spatial_dims == 2:
                target_y_min = np.floor(target[i : i + 1, :, 0]).astype(int)
                target_x_min = np.floor(target[i : i + 1, :, 1]).astype(int)
                in_y_min_x_min = pred_region[
                    np.arange(pred_region.shape[0])[:, None],
                    np.arange(pred_region.shape[1]),
                    target_y_min,
                    target_x_min,
                ]
            elif self.spatial_dims == 3:
                target_z_min = np.floor(target[i : i + 1, :, 0]).astype(int)
                target_y_min = np.floor(target[i : i + 1, :, 1]).astype(int)
                target_x_min = np.floor(target[i : i + 1, :, 2]).astype(int)
                in_y_min_x_min = pred_region[
                    np.arange(pred_region.shape[0])[:, None],
                    np.arange(pred_region.shape[1]),
                    target_z_min,
                    target_y_min,
                    target_x_min,
                ]
            output.append(in_y_min_x_min)
        return np.concatenate(output, axis=0)

    def evaluate(
        self,
        heatmaps,
        pred_landmarks,
        target,
        original_dims=None,
        padding=None,
        spacing=None,
        confidence=0.95,
        batch_size=1,
        return_summary=True,
    ):
        if isinstance(heatmaps, torch.Tensor):
            heatmaps = heatmaps.detach().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().numpy()
        self.check_input(heatmaps)
        nb_batch = heatmaps.shape[0] // batch_size
        if heatmaps.shape[0] % batch_size != 0:
            nb_batch += 1
        error = euclidean_distance(
            pixel_to_unit_numpy(pred_landmarks, pixel_spacing=spacing),
            pixel_to_unit_numpy(target, pixel_spacing=spacing),
        )
        area = np.zeros((heatmaps.shape[0], heatmaps.shape[1]))
        in_region = np.zeros((heatmaps.shape[0], heatmaps.shape[1]))

        for i in tqdm(range(0, heatmaps.shape[0], batch_size)):
            batch_pred = heatmaps[i : i + batch_size]
            batch_target = target[i : i + batch_size]
            batch_original_dims = (
                original_dims[i : i + batch_size] if original_dims is not None else None
            )
            batch_padding = padding[i : i + batch_size] if padding is not None else None
            batch_spacing = spacing[i : i + batch_size] if spacing is not None else None

            batch_pred_region = self.predict(
                batch_pred,
                original_dims=batch_original_dims,
                padding=batch_padding,
                confidence=confidence,
            )
            if self.spatial_dims == 2:
                batch_area = np.sum(batch_pred_region, axis=(-1, -2)) * np.prod(
                    batch_spacing, axis=-1
                )
                batch_target_y_min = np.floor(batch_target[:, :, 0]).astype(int)
                batch_target_x_min = np.floor(batch_target[:, :, 1]).astype(int)

                in_y_min_x_min = batch_pred_region[
                    np.arange(batch_pred_region.shape[0])[:, None],
                    np.arange(batch_pred_region.shape[1]),
                    batch_target_y_min,
                    batch_target_x_min,
                ]
            elif self.spatial_dims == 3:
                batch_area = np.sum(batch_pred_region, axis=(-1, -2, -3)) * np.prod(
                    batch_spacing, axis=-1
                )
                batch_target_z_min = np.floor(batch_target[:, :, 0]).astype(int)
                batch_target_y_min = np.floor(batch_target[:, :, 1]).astype(int)
                batch_target_x_min = np.floor(batch_target[:, :, 2]).astype(int)

                in_y_min_x_min = batch_pred_region[
                    np.arange(batch_pred_region.shape[0])[:, None],
                    np.arange(batch_pred_region.shape[1]),
                    batch_target_z_min,
                    batch_target_y_min,
                    batch_target_x_min,
                ]

            batch_coverage = in_y_min_x_min
            area[i : i + batch_size] = batch_area
            in_region[i : i + batch_size] = batch_coverage

        coverage = in_region.mean()
        efficiency_mean = area.mean()
        efficiency_median = np.median(area)
        efficiency_q1 = np.quantile(area, 0.25)
        efficiency_q3 = np.quantile(area, 0.75)
        adaptivity = spearmanr(error.flatten(), area.flatten()).statistic
        print(f"Coverage: {coverage}")
        print(f"Efficiency (mean): {efficiency_mean}")
        print(f"Efficiency (median): {efficiency_median}")
        print(f"Efficiency (Q1): {efficiency_q1}")
        print(f"Efficiency (Q3): {efficiency_q3}")
        print(f"Adeptivity: {adaptivity}")
        if not return_summary:
            return (in_region, area, error)
        return {
            "coverage": coverage,
            "efficiency_mean": efficiency_mean,
            "efficiency_median": efficiency_median,
            "efficiency_q1": efficiency_q1,
            "efficiency_q3": efficiency_q3,
            "adaptivity": adaptivity,
        }


class MultivariateNormalRegressor:
    def __init__(self, spatial_dims=2, nb_landmarks=1):
        self.spatial_dims = spatial_dims
        self.nb_landmarks = nb_landmarks

    def check_input(self, pred, pred_cov):
        if pred.ndim != 3:
            raise ValueError("pred must have shape (N, C, D)")
        if pred_cov.ndim != 4:
            raise ValueError("pred_cov must have shape (N, C, D, D)")
        if pred.shape[2] != pred_cov.shape[2] and self.spatial_dims != pred.shape[2]:
            raise ValueError("pred and pred_cov must have the same second dimension")
        if pred_cov.shape[2] != pred_cov.shape[3]:
            raise ValueError("pred_cov must be square")
        if pred.shape[0] != pred_cov.shape[0]:
            raise ValueError("pred and pred_cov must have the same first dimension")
        if pred.shape[1] != self.nb_landmarks:
            raise ValueError("pred must have the same second dimension as nb_landmarks")
        if pred_cov.shape[1] != self.nb_landmarks:
            raise ValueError("pred_cov must have the same second dimension as nb_landmarks")

    def fit(self, pred, pred_cov, target):
        return

    def bound(self, pred_cov, confidence=0.95):
        """Get the bound for the given confidence level."""
        return pred_cov * norm.ppf(confidence) ** 2

    def area_prediction_region(self, pred_cov, spacing=None, confidence=0.95):
        """
        Calculate the area of the prediction region, accounting for pixel spacing.

        Parameters:
        pred_cov (ndarray): The predicted covariance matrix.
        spacing (ndarray, optional): The spacing between pixels in millimeters,
                         with shape (Batch_size, spatial_size).
                         Defaults to None.
        confidence (float, optional): The confidence level for the prediction region.
                         Defaults to 0.95.

        Returns:
        float: The area of the prediction region.
        """
        pred_cov_scaled = self.bound(pred_cov, confidence)
        if spacing is not None:
            if isinstance(spacing, torch.Tensor):
                spacing = spacing.detach().numpy()
            assert spacing.shape == (pred_cov.shape[0], self.spatial_dims)
            # Make a diagonal matrix with the pixel spacing (Batch_size, spatial_size, spatial_size)
            # e.g. for 2D: [[s1, 0], [0, s2]], for 3D: [[s1, 0, 0], [0, s2, 0], [0, 0, s3]]
            spacing_matrix = np.stack(
                [np.diag(spacing[i]) for i in range(spacing.shape[0])], axis=0
            ).reshape((pred_cov.shape[0], 1, self.spatial_dims, self.spatial_dims))
            # Scale the covariance matrix by the pixel spacing
            pred_cov_scaled = (
                spacing_matrix @ pred_cov_scaled @ np.transpose(spacing_matrix, (0, 1, 3, 2))
            )
        eig = np.linalg.eigvalsh(pred_cov_scaled)
        return np.pi * np.prod(np.sqrt(eig), axis=-1)

    def predict(self, pred, pred_cov, confidence=0.95):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().numpy()
        if isinstance(pred_cov, torch.Tensor):
            pred_cov = pred_cov.detach().numpy()
        self.check_input(pred, pred_cov)
        return pred, self.bound(pred_cov, confidence)

    def predict_contour(self, pred, pred_cov, confidence=0.95, sample_points=200):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().numpy()
        if isinstance(pred_cov, torch.Tensor):
            pred_cov = pred_cov.detach().numpy()
        self.check_input(pred, pred_cov)

        if self.spatial_dims == 2:
            angles = np.linspace(0, 2 * np.pi, sample_points)
            directions = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
            directions = directions.reshape((1, -1, 1, 1, self.spatial_dims))
            error_bound = directions @ np.linalg.cholesky(self.bound(pred_cov, confidence)).reshape(
                (-1, 1, self.nb_landmarks, self.spatial_dims, self.spatial_dims)
            )
            return pred.reshape(
                (-1, 1, self.nb_landmarks, self.spatial_dims)
            ) + error_bound.reshape((-1, directions.shape[1], self.nb_landmarks, self.spatial_dims))
        elif self.spatial_dims == 3:
            eigenvalues, eigenvectors = np.linalg.eigh(self.bound(pred_cov, confidence))

            # Create a grid of points for a unit sphere
            u = np.linspace(0, 2 * np.pi, sample_points)
            v = np.linspace(0, np.pi, sample_points // 2)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones_like(u), np.cos(v))

            # Scale and rotate the unit sphere to match the ellipsoid
            radii = np.sqrt(eigenvalues)
            transform = eigenvectors @ np.stack(
                [
                    np.stack(
                        [radii[..., 0], np.zeros_like(radii[..., 0]), np.zeros_like(radii[..., 0])],
                        axis=-1,
                    ),
                    np.stack(
                        [np.zeros_like(radii[..., 0]), radii[..., 1], np.zeros_like(radii[..., 0])],
                        axis=-1,
                    ),
                    np.stack(
                        [np.zeros_like(radii[..., 0]), np.zeros_like(radii[..., 0]), radii[..., 2]],
                        axis=-1,
                    ),
                ],
                axis=-2,
            )
            ellipsoid = transform @ np.array([z.ravel(), y.ravel(), x.ravel()])
            # Reshape to 3D grid
            x_ellipsoid = ellipsoid[..., 2, :].reshape((-1, self.nb_landmarks, *x.shape))
            y_ellipsoid = ellipsoid[..., 1, :].reshape((-1, self.nb_landmarks, *y.shape))
            z_ellipsoid = ellipsoid[..., 0, :].reshape((-1, self.nb_landmarks, *z.shape))

            return np.stack([z_ellipsoid, y_ellipsoid, x_ellipsoid], axis=-1) + pred.reshape(
                (-1, self.nb_landmarks, 1, 1, self.spatial_dims)
            )
        else:
            raise ValueError("Unsupported spatial dimension")

    def in_region(self, pred, pred_cov, target, confidence=0.95):
        return mahalanobis_distance(pred, self.bound(pred_cov, confidence), target) <= 1

    def evaluate(self, pred, pred_cov, target, spacing=None, confidence=0.95, return_summary=True):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().numpy()
        if isinstance(pred_cov, torch.Tensor):
            pred_cov = pred_cov.detach().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().numpy()
        self.check_input(pred, pred_cov)
        assert pred.shape == target.shape
        error = euclidean_distance(
            pixel_to_unit_numpy(pred, pixel_spacing=spacing),
            pixel_to_unit_numpy(target, pixel_spacing=spacing),
        )
        area = self.area_prediction_region(pred_cov, spacing=spacing, confidence=confidence)
        in_region = self.in_region(pred, pred_cov, target, confidence=confidence)
        coverage = in_region.mean()
        efficiency_mean = area.mean()
        efficiency_median = np.median(area)
        efficiency_q1 = np.quantile(area, 0.25)
        efficiency_q3 = np.quantile(area, 0.75)
        adaptivity = spearmanr(error.flatten(), area.flatten()).statistic
        print(f"Coverage: {coverage}")
        print(f"Efficiency (mean): {efficiency_mean}")
        print(f"Efficiency (median): {efficiency_median}")
        print(f"Efficiency (Q1): {efficiency_q1}")
        print(f"Efficiency (Q3): {efficiency_q3}")
        print(f"Adeptivity: {adaptivity}")
        if not return_summary:
            return (in_region, area, error)
        return {
            "coverage": coverage,
            "efficiency_mean": efficiency_mean,
            "efficiency_median": efficiency_median,
            "efficiency_q1": efficiency_q1,
            "efficiency_q3": efficiency_q3,
            "adaptivity": adaptivity,
        }


class ContourHuggingRegressor:

    def __init__(self, spatial_dims=2, nb_landmarks=1, heatmap_size=(512, 512)):
        self.spatial_dims = spatial_dims
        self.nb_landmarks = nb_landmarks
        self.heatmap_size = heatmap_size

    def check_input(self, heatmaps):
        if heatmaps.ndim != 4 and self.spatial_dims == 2:
            raise ValueError("pred must have shape (N, C, H, W)")
        elif heatmaps.ndim != 5 and self.spatial_dims == 3:
            raise ValueError("pred must have shape (N, C, D, H, W)")
        elif heatmaps.shape[1] != self.nb_landmarks:
            raise ValueError("pred must have the same second dimension as nb_landmarks")
        elif heatmaps.shape[2:] != self.heatmap_size:
            raise ValueError(
                f"pred {heatmaps.shape[2:]} must have the same dimensions as heatmap_size {self.heatmap_size}"
            )

    def fit(self, heatmaps, target, original_dims=None, padding=None):
        return

    def area_prediction_region(
        self, heatmaps, original_dims=None, padding=None, spacing=None, confidence=0.95
    ):
        area = np.zeros((heatmaps.shape[0], heatmaps.shape[1]))
        for i in tqdm(range(heatmaps.shape[0])):
            pred_region = self.predict(
                heatmaps[i : i + 1],
                original_dims=original_dims[i : i + 1],
                padding=padding[i : i + 1],
                confidence=confidence,
            )
            if spacing is None:
                if self.spatial_dims == 2:
                    area[i] = np.sum(pred_region, axis=(-1, -2))
                elif self.spatial_dims == 3:
                    area[i] = np.sum(pred_region, axis=(-1, -2, -3))
            if self.spatial_dims == 2:
                area[i] = np.sum(pred_region, axis=(-1, -2)) * np.prod(spacing[i], axis=-1)
            elif self.spatial_dims == 3:
                area[i] = np.sum(pred_region, axis=(-1, -2, -3)) * np.prod(spacing[i], axis=-1)
        return area

    def predict(self, heatmaps, original_dims=None, padding=None, spacing=None, confidence=0.95):
        if isinstance(heatmaps, torch.Tensor):
            heatmaps = heatmaps.detach().numpy()
        # clone heatmaps
        heatmaps = np.copy(heatmaps)
        for i in range(heatmaps.shape[0]):
            for j in range(heatmaps.shape[1]):
                idxs_sorted_heatmaps = np.argsort(heatmaps[i, j].flatten())[::-1]
                cum_sum_sorted_heatmaps = np.cumsum(heatmaps[i, j].flatten()[idxs_sorted_heatmaps])
                threshold = np.argmax(cum_sum_sorted_heatmaps >= confidence)
                heatmaps[i, j] = np.zeros_like(heatmaps[i, j])
                flatten_heatmaps = heatmaps[i, j].flatten()
                flatten_heatmaps[idxs_sorted_heatmaps[:threshold]] = 1
                heatmaps[i, j] = flatten_heatmaps.reshape(heatmaps[i, j].shape)
        if original_dims is None:
            return heatmaps
        # Rescale the prediction region to the original dimensions
        heatmap = transform_heatmap_to_original_size_numpy(
            heatmaps.astype(float), padding=padding, original_dim=original_dims
        )
        return (heatmap > 0).astype(int)

    def predict_contour(self, heatmaps, original_dims=None, padding=None, confidence=0.95):
        if isinstance(heatmaps, torch.Tensor):
            heatmaps = heatmaps.detach().numpy()

        output = []
        for i in tqdm(range(heatmaps.shape[0])):
            if original_dims is None or padding is None:
                pred_region = self.predict(heatmaps[i : i + 1], confidence=confidence)[0]
            else:
                pred_region = self.predict(
                    heatmaps[i : i + 1],
                    original_dims=original_dims[i : i + 1],
                    padding=padding[i : i + 1],
                    confidence=confidence,
                )[0]
            output_sample = []
            for j in range(pred_region.shape[0]):
                pred_region_sample = pred_region[j]
                if self.spatial_dims == 2:
                    contour, _ = cv2.findContours(
                        pred_region_sample.astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    output_sample.append(contour)
                elif self.spatial_dims == 3:
                    verts, faces, _, _ = measure.marching_cubes(
                        pred_region_sample, level=0.5
                    )  # returns verts, faces, normals, values
                    output_sample.append((verts, faces))
            output.append(output_sample)
        return output

    def in_region(
        self, heatmaps, target, original_dims=None, padding=None, confidence=0.95, batch_size=1
    ):
        output = []
        for i in tqdm(range(0, heatmaps.shape[0], batch_size)):
            pred_region = self.predict(
                heatmaps[i : i + 1],
                original_dims=original_dims[i : i + 1],
                padding=padding[i : i + 1],
                confidence=confidence,
            )
            if self.spatial_dims == 2:
                target_y_min = np.floor(target[i : i + 1, :, 0]).astype(int)
                target_x_min = np.floor(target[i : i + 1, :, 1]).astype(int)
                in_y_min_x_min = pred_region[
                    np.arange(pred_region.shape[0])[:, None],
                    np.arange(pred_region.shape[1]),
                    target_y_min,
                    target_x_min,
                ]
            elif self.spatial_dims == 3:
                target_z_min = np.floor(target[i : i + 1, :, 0]).astype(int)
                target_y_min = np.floor(target[i : i + 1, :, 1]).astype(int)
                target_x_min = np.floor(target[i : i + 1, :, 2]).astype(int)
                in_y_min_x_min = pred_region[
                    np.arange(pred_region.shape[0])[:, None],
                    np.arange(pred_region.shape[1]),
                    target_z_min,
                    target_y_min,
                    target_x_min,
                ]
            output.append(in_y_min_x_min)
        return np.concatenate(output, axis=0)

    def evaluate(
        self,
        heatmaps,
        pred_landmarks,
        target,
        original_dims=None,
        padding=None,
        spacing=None,
        confidence=0.95,
        batch_size=1,
        return_summary=True,
    ):
        if isinstance(heatmaps, torch.Tensor):
            heatmaps = heatmaps.detach().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().numpy()
        self.check_input(heatmaps)
        nb_batch = heatmaps.shape[0] // batch_size
        if heatmaps.shape[0] % batch_size != 0:
            nb_batch += 1
        error = euclidean_distance(
            pixel_to_unit_numpy(pred_landmarks, pixel_spacing=spacing),
            pixel_to_unit_numpy(target, pixel_spacing=spacing),
        )
        area = np.zeros((heatmaps.shape[0], heatmaps.shape[1]))
        in_region = np.zeros((heatmaps.shape[0], heatmaps.shape[1]))

        for i in tqdm(range(0, heatmaps.shape[0], batch_size)):
            batch_pred = heatmaps[i : i + batch_size]
            batch_target = target[i : i + batch_size]
            batch_original_dims = (
                original_dims[i : i + batch_size] if original_dims is not None else None
            )
            batch_padding = padding[i : i + batch_size] if padding is not None else None
            batch_spacing = spacing[i : i + batch_size] if spacing is not None else None

            batch_pred_region = self.predict(
                batch_pred,
                original_dims=batch_original_dims,
                padding=batch_padding,
                confidence=confidence,
            )
            if self.spatial_dims == 2:
                batch_area = np.sum(batch_pred_region, axis=(-1, -2)) * np.prod(
                    batch_spacing, axis=-1
                )
                batch_target_y_min = np.floor(batch_target[:, :, 0]).astype(int)
                batch_target_x_min = np.floor(batch_target[:, :, 1]).astype(int)

                in_y_min_x_min = batch_pred_region[
                    np.arange(batch_pred_region.shape[0])[:, None],
                    np.arange(batch_pred_region.shape[1]),
                    batch_target_y_min,
                    batch_target_x_min,
                ]
            elif self.spatial_dims == 3:
                batch_area = np.sum(batch_pred_region, axis=(-1, -2, -3)) * np.prod(
                    batch_spacing, axis=-1
                )
                batch_target_z_min = np.floor(batch_target[:, :, 0]).astype(int)
                batch_target_y_min = np.floor(batch_target[:, :, 1]).astype(int)
                batch_target_x_min = np.floor(batch_target[:, :, 2]).astype(int)

                in_y_min_x_min = batch_pred_region[
                    np.arange(batch_pred_region.shape[0])[:, None],
                    np.arange(batch_pred_region.shape[1]),
                    batch_target_z_min,
                    batch_target_y_min,
                    batch_target_x_min,
                ]

            batch_coverage = in_y_min_x_min
            area[i : i + batch_size] = batch_area
            in_region[i : i + batch_size] = batch_coverage

        coverage = in_region.mean()
        efficiency_mean = area.mean()
        efficiency_median = np.median(area)
        efficiency_q1 = np.quantile(area, 0.25)
        efficiency_q3 = np.quantile(area, 0.75)
        adaptivity = spearmanr(error.flatten(), area.flatten()).statistic
        print(f"Coverage: {coverage}")
        print(f"Efficiency (mean): {efficiency_mean}")
        print(f"Efficiency (median): {efficiency_median}")
        print(f"Efficiency (Q1): {efficiency_q1}")
        print(f"Efficiency (Q3): {efficiency_q3}")
        print(f"Adeptivity: {adaptivity}")
        if not return_summary:
            return (in_region, area, error)
        return {
            "coverage": coverage,
            "efficiency_mean": efficiency_mean,
            "efficiency_median": efficiency_median,
            "efficiency_q1": efficiency_q1,
            "efficiency_q3": efficiency_q3,
            "adaptivity": adaptivity,
        }


def transform_heatmap_to_original_size_numpy(heatmaps, padding, original_dim, mode="nearest-exact"):
    heatmap = torch.tensor(heatmaps)
    padding = torch.tensor(padding)
    original_dim = torch.tensor(original_dim)
    return (
        transform_heatmap_to_original_size(heatmap, padding, original_dim, mode=mode)
        .detach()
        .numpy()
    )


def transform_heatmap_to_original_size(heatmaps, padding, original_dim, mode="nearest-exact"):
    heatmaps_orig_size = []
    for i in range(heatmaps.shape[0]):
        # resize image to original size
        resize_dim = original_dim[i] + 2 * padding[i]
        resize_dim = tuple(int(dim.item()) for dim in resize_dim)
        heatmap = torch.nn.functional.interpolate(
            heatmaps[i].unsqueeze(0), size=resize_dim, mode=mode
        )
        # crop image to original size
        if heatmaps.ndim == 4:  # 2D case
            heatmaps_orig_size.append(
                heatmap[
                    :,
                    :,
                    int(padding[i, 0]) : int(padding[i, 0] + original_dim[i, 0].item()),
                    int(padding[i, 1]) : int(padding[i, 1] + original_dim[i, 1].item()),
                ]
            )
        elif heatmaps.ndim == 5:  # 3D case
            heatmaps_orig_size.append(
                heatmap[
                    :,
                    :,
                    int(padding[i, 0]) : int(padding[i, 0] + original_dim[i, 0].item()),
                    int(padding[i, 1]) : int(padding[i, 1] + original_dim[i, 1].item()),
                    int(padding[i, 2]) : int(padding[i, 2] + original_dim[i, 2].item()),
                ]
            )
    return torch.cat(heatmaps_orig_size, dim=0)


def resize_landmarks(
    landmarks: np.ndarray,
    dim: Optional[np.ndarray] = None,
    dim_orig: Optional[np.ndarray] = None,
    padding: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert the landmarks from pixel to unit (Numpy version).

    Args:
        landmarks (np.ndarray): landmarks
        pixel_spacing (Optional[np.ndarray], optional): pixel spacing. Defaults to None.
        dim (Optional[tuple[int, ...] | np.ndarray], optional): image size. Defaults to None.
        dim_orig (Optional[np.ndarray], optional): original image size. Defaults to None.
        padding (Optional[np.ndarray], optional): padding. Defaults to None.

    Returns:
        np.ndarray: landmarks in units
    """
    spatial_dim = landmarks.shape[-1]

    if dim is not None:
        assert dim.shape[-1] == spatial_dim, f"dim must have {spatial_dim} elements."
    if dim_orig is not None:
        assert dim_orig.shape[-1] == spatial_dim, f"dim_orig must have {spatial_dim} elements."
    if padding is None:
        padding = np.zeros_like(landmarks)
    t_landmarks = landmarks + padding.reshape((landmarks.shape[0], 1, spatial_dim))
    t_landmarks = t_landmarks * (dim / (dim_orig + 2 * padding)).reshape((-1, 1, spatial_dim))  # type: ignore
    return t_landmarks
