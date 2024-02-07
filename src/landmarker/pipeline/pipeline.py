from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import numpy as np
import torch
from monai.transforms import Compose
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore

from landmarker.data import LandmarkDataset
from landmarker.data.landmark_dataset import HeatmapDataset
from landmarker.heatmap.decoder import (
    coord_argmax,
    coord_soft_argmax,
    coord_soft_argmax_cov,
    coord_weighted_spatial_mean,
    coord_weighted_spatial_mean_cov,
)
from landmarker.heatmap.generator import GaussianHeatmapGenerator
from landmarker.losses.losses import (
    EuclideanDistanceVarianceReg,
    GaussianHeatmapL2Loss,
    MultivariateGaussianNLLLoss,
    StarLoss,
)
from landmarker.models import get_model
from landmarker.train.utils import EarlyStopping, SaveBestModel
from landmarker.utils import pixel_to_unit
from landmarker.visualize import detection_report


class HeatmapRegressionPipeline(ABC):
    """
    HeatmapRegressionPipeline is a pipeline that predicts heatmaps for a given image.
    """

    def __init__(
        self,
        model: str | nn.Module,
        in_channels_img: int,
        nb_landmarks: int,
        train_config: dict[str, Any],
        dim_img: tuple[int, int] = (512, 512),
        heatmap_decoder: Callable = coord_argmax,
        verbose: bool = False,
        device: str = "cpu",
        fitted: bool = False,
    ):
        self.model = model
        self.in_channels_img = in_channels_img
        self.nb_landmarks = nb_landmarks
        self.init_train_config(train_config)
        self.dim_img = dim_img
        self.heatmap_decoder = heatmap_decoder
        self.fitted = fitted
        self.verbose = verbose
        self.device = device
        self.final_activation: nn.Module = nn.Identity()
        # Set EarlyStopping and SaveBestModel
        self.early_stopping = EarlyStopping(
            patience=self.train_config["patience"], verbose=self.verbose
        )
        self.save_best_model = SaveBestModel(verbose=self.verbose)

    def __reduce__(self) -> str | tuple[Any, ...]:
        return (
            self.__class__,
            (
                self.model,
                self.in_channels_img,
                self.nb_landmarks,
                self.train_config,
                self.dim_img,
                self.heatmap_decoder,
                self.verbose,
                self.device,
                self.fitted,
            ),
        )

    def init_train_config(self, train_config: dict[str, Any]) -> None:
        self.train_config = train_config
        if "lr" not in self.train_config:
            self.train_config["lr"] = 1e-3
        if "batch_size" not in self.train_config:
            self.train_config["batch_size"] = 32
        if "epochs" not in self.train_config:
            self.train_config["epochs"] = 1000
        if "patience" not in self.train_config:
            self.train_config["patience"] = 20
        if "min_delta" not in self.train_config:
            self.train_config["min_delta"] = 0.0
        if "weight_decay" not in self.train_config:
            self.train_config["weight_decay"] = 0.0
        if "lr_scheduler" not in self.train_config:
            self.train_config["lr_scheduler"] = False
        if "lr_scheduler_factor" not in self.train_config:
            self.train_config["lr_scheduler_factor"] = 0.1
        if "lr_scheduler_patience" not in self.train_config:
            self.train_config["lr_scheduler_patience"] = 10
        if "lr_scheduler_cooldown" not in self.train_config:
            self.train_config["lr_scheduler_cooldown"] = 5
        if "optimizer" not in self.train_config:
            self.train_config["optimizer"] = "adam"
        if "criterion" not in self.train_config:
            self.train_config["criterion"] = "mse"

    @abstractmethod
    def fit(
        self,
        imgs: torch.Tensor | np.ndarray | list[str],
        landmarks: torch.Tensor | np.ndarray,
        transform: Optional[Compose] = None,
        cache_ds: bool = True,
    ) -> None:
        pass

    @abstractmethod
    def fit_ds(
        self,
        ds: LandmarkDataset | HeatmapDataset,
        transform: Optional[Compose] = None,
    ) -> None:
        pass

    def _predict(
        self,
        imgs: torch.Tensor | np.ndarray | list[str],
        landmarks: Optional[torch.Tensor | np.ndarray] = None,
        decode_heatmaps=True,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | torch.Tensor
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        assert self.fitted, "Model must be fitted first"
        if decode_heatmaps:
            pred_landmarks = torch.zeros((len(imgs), self.nb_landmarks, 2))
            dim_originals = torch.zeros((len(imgs), 2))
            paddings = torch.zeros((len(imgs), 2))
            if landmarks is not None:
                true_landmarks_original = landmarks
                true_landmarks = torch.zeros((len(imgs), self.nb_landmarks, 2))
            else:
                true_landmarks_original = torch.zeros((len(imgs), self.nb_landmarks, 2))
        else:
            heatmaps = torch.zeros((len(imgs), self.nb_landmarks, *self.dim_img))
        ds = LandmarkDataset(
            imgs=imgs,
            landmarks=true_landmarks_original,
            transform=None,
            store_imgs=True,
            dim_img=self.dim_img,
        )
        self.model.eval()  # type: ignore
        with torch.no_grad():
            for i, (
                images,
                true_landmark,
                _,
                _,
                _,
                dim_original,
                _,
                padding,
            ) in enumerate(tqdm(ds)):
                images = images.unsqueeze(0).to(self.device)
                outputs = self.final_activation(self.model(images))  # type: ignore
                outputs = outputs[:, int(self.background) :]  # type: ignore
                if decode_heatmaps:
                    pred_landmarks[i] = self.heatmap_decoder(outputs).cpu()
                    dim_originals[i] = dim_original
                    paddings[i] = padding
                    if landmarks is not None:
                        true_landmarks[i] = true_landmark
                else:
                    heatmaps[i] = outputs.cpu().squeeze(0)
        if decode_heatmaps:
            if landmarks is not None:
                return pred_landmarks, true_landmarks, dim_originals, paddings
            return pred_landmarks, dim_originals, paddings
        return heatmaps

    def predict(self, imgs: torch.Tensor | np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        pred_landmarks, dim_original, padding = self._predict(imgs)  # type: ignore
        return pixel_to_unit(pred_landmarks, dim_orig=dim_original, padding=padding)

    def predict_heatmaps(self, imgs: torch.Tensor | np.ndarray) -> torch.Tensor:
        return self._predict(imgs, decode_heatmaps=False)  # type: ignore

    def evaluate(
        self,
        imgs: torch.Tensor | np.ndarray | list[str],
        landmarks: torch.Tensor | np.ndarray,
        pixel_spacing: Optional[torch.Tensor] = None,
    ) -> Optional[dict]:
        (pred_landmarks, true_landmarks, dim_original, padding) = self._predict(
            imgs, landmarks=landmarks  # type: ignore
        )  # type: ignore
        if pixel_spacing is None:
            pixel_spacing = torch.ones((len(imgs), 2))
        return detection_report(
            true_landmarks=true_landmarks,
            pred_landmarks=pred_landmarks,
            dim_orig=dim_original,
            dim=self.dim_img,
            padding=padding,
            pixel_spacing=pixel_spacing,
            output_dict=True,
        )

    @abstractmethod
    def train_epoch(self, train_loader) -> float:
        pass

    @abstractmethod
    def val_epoch(self, val_loader) -> float:
        pass

    def train(
        self,
        ds: LandmarkDataset,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        transform = ds.transform
        for epoch in range(self.train_config["epochs"]):
            train_loss = self.train_epoch(train_loader)
            ds.transform = None
            val_loss = self.val_epoch(val_loader)
            ds.transform = transform
            if self.verbose:
                print(
                    f"Epoch {epoch+1}/{self.train_config['epochs']} - Train loss: {train_loss:.8f}"
                    f" - Val loss: {val_loss:.8f}"
                )
            self.early_stopping(val_loss)
            self.save_best_model(val_loss, self.model)  # type: ignore
            if self.train_config["lr_scheduler"]:
                self.lr_scheduler.step(val_loss)  # type: ignore
            if self.early_stopping.early_stop:
                if self.verbose:
                    print("Loading best model...")
                self.model.load_state_dict(torch.load(self.save_best_model.path))  # type: ignore
                break


class StaticHeatmapRegressionPipeline(HeatmapRegressionPipeline):
    """
    StaticHeatmapRegressionPipeline is a pipeline that predicts heatmaps for a given image.
    """

    def __init__(
        self, sigmas: float | list[float] | np.ndarray | torch.Tensor, gamma=None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.sigma = sigmas
        self.gamma = gamma
        # Set criterion
        self.background = False
        if isinstance(self.train_config["criterion"], nn.Module):
            self.criterion = self.train_config["criterion"]
        elif self.train_config["criterion"] == "mse":
            self.criterion = nn.MSELoss()
        elif self.train_config["criterion"] == "l1":
            self.criterion = nn.L1Loss()
        elif self.train_config["criterion"] == "bce_with_logits":
            self.criterion = nn.BCEWithLogitsLoss()
            self.final_activation = nn.Sigmoid()
        elif self.train_config["criterion"] == "cross_entropy":
            self.background = True
            self.criterion = nn.CrossEntropyLoss()  # TODO: add background channel
            self.final_activation = nn.Softmax(dim=1)
        else:
            raise ValueError("Criterion not supported")
        if isinstance(self.model, str):
            self.model: nn.Module = get_model(
                self.model,  # type: ignore
                in_channels=self.in_channels_img,
                out_channels=self.nb_landmarks + int(self.background),
            )
        self.model.to(self.device)

        # Set optimizer
        if isinstance(self.train_config["optimizer"], torch.optim.Optimizer):
            self.optimizer = self.train_config["optimizer"](  # type: ignore
                self.model.parameters(),  # type: ignore
                lr=self.train_config["lr"],
                weight_decay=self.train_config["weight_decay"],
            )
        elif self.train_config["optimizer"] == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.train_config["lr"],
                weight_decay=self.train_config["weight_decay"],
            )
        elif self.train_config["optimizer"] == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.train_config["lr"],
                weight_decay=self.train_config["weight_decay"],
            )
        elif self.train_config["optimizer"] == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.train_config["lr"],
                weight_decay=self.train_config["weight_decay"],
            )
        else:
            raise ValueError("Optimizer not supported")
        # Set lr_scheduler
        if self.train_config["lr_scheduler"]:
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.train_config["lr_scheduler_factor"],
                patience=self.train_config["lr_scheduler_patience"],
                verbose=self.verbose,
                cooldown=self.train_config["lr_scheduler_cooldown"],
            )

    def __reduce__(self) -> str | tuple[Any, ...]:
        return (
            self.__class__,
            (
                self.sigma,
                self.model,
                self.in_channels_img,
                self.nb_landmarks,
                self.train_config,
                self.dim_img,
                self.heatmap_decoder,
                self.verbose,
                self.device,
                self.fitted,
            ),
        )

    def fit(
        self,
        imgs: torch.Tensor | np.ndarray | list[str],
        landmarks: torch.Tensor | np.ndarray,
        transform: Optional[Compose] = None,
        cache_ds: bool = True,
    ) -> None:
        assert len(imgs) == len(landmarks), "Number of images and landmarks must be equal"
        ds = HeatmapDataset(
            imgs,
            landmarks,
            transform=transform,
            sigma=self.sigma,
            gamma=self.gamma,
            store_imgs=cache_ds,
            background=self.background,
            dim_img=self.dim_img,
        )
        split_lengths = [0.8, 0.2]
        ds_train, ds_val = torch.utils.data.random_split(ds, split_lengths)
        train_loader = DataLoader(
            ds_train,
            batch_size=self.train_config["batch_size"],
            shuffle=True,
        )
        val_loader = DataLoader(
            ds_val,
            batch_size=self.train_config["batch_size"],
            shuffle=False,
        )
        self.train(ds, train_loader, val_loader)
        self.fitted = True

    def fit_ds(
        self, ds: HeatmapDataset, transform: Optional[Compose] = None  # type: ignore
    ) -> None:
        if transform is not None:
            ds.transform = transform
        split_lengths = [0.8, 0.2]
        ds_train, ds_val = torch.utils.data.random_split(ds, split_lengths)
        train_loader = DataLoader(
            ds_train,
            batch_size=self.train_config["batch_size"],
            shuffle=True,
        )
        val_loader = DataLoader(
            ds_val,
            batch_size=self.train_config["batch_size"],
            shuffle=False,
        )
        self.train(ds, train_loader, val_loader)
        self.fitted = True

    def train_epoch(self, train_loader: DataLoader) -> float:
        running_loss = 0
        self.model.train()
        for i, (images, heatmaps, landmarks, _, _, _, _, _, _) in enumerate(tqdm(train_loader)):
            images = images.to(self.device)
            heatmaps = heatmaps.to(self.device)
            landmarks = landmarks.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, heatmaps)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(train_loader)

    def val_epoch(self, val_loader: DataLoader) -> float:
        eval_loss = 0
        self.model.eval()
        with torch.no_grad():
            for _, (images, heatmaps, landmarks, _, _, _, _, _, _) in enumerate(tqdm(val_loader)):
                images = images.to(self.device)
                heatmaps = heatmaps.to(self.device)
                landmarks = landmarks.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, heatmaps)
                eval_loss += loss.item()
        return eval_loss / len(val_loader)


class AdaptiveHeatmapRegressionPipeline(HeatmapRegressionPipeline):
    """
    AdaptiveHeatmapRegressionPipeline is a pipeline that predicts heatmaps for a given image.
    """

    def __init__(self, sigmas, gamma=None, alpha=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma = sigmas
        self.alpha = alpha
        self.gamma = gamma
        # Set criterion
        self.background = False
        if isinstance(self.train_config["criterion"], nn.Module):
            self.criterion = self.train_config["criterion"]
        elif self.train_config["criterion"] == "l2":
            self.criterion = GaussianHeatmapL2Loss(alpha=self.alpha)
        else:
            raise ValueError("Criterion not supported")
        self.heatmap_generator = GaussianHeatmapGenerator(
            self.nb_landmarks,
            sigmas=self.sigma,
            gamma=self.gamma,
            heatmap_size=self.dim_img,
            learnable=True,
            device=self.device,
            background=self.background,
        )
        if isinstance(self.model, str):
            self.model: nn.Module = get_model(
                self.model,
                in_channels=self.in_channels_img,
                out_channels=self.nb_landmarks + int(self.background),
            )
        self.model.to(self.device)
        # Set optimizer
        if isinstance(self.train_config["optimizer"], torch.optim.Optimizer):
            self.optimizer = self.train_config["optimizer"](
                [
                    {"params": self.model.parameters(), "weight_decay": 1e-3},
                    {"params": self.heatmap_generator.sigmas},
                    {"params": self.heatmap_generator.rotation},
                ],
                lr=self.train_config["lr"],
            )
        elif self.train_config["optimizer"] == "adam":
            self.optimizer = torch.optim.Adam(
                [
                    {"params": self.model.parameters(), "weight_decay": 1e-3},
                    {"params": self.heatmap_generator.sigmas},
                    {"params": self.heatmap_generator.rotation},
                ],
                lr=self.train_config["lr"],
            )
        elif self.train_config["optimizer"] == "adamw":
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": self.model.parameters(), "weight_decay": 1e-3},
                    {"params": self.heatmap_generator.sigmas},
                    {"params": self.heatmap_generator.rotation},
                ],
                lr=self.train_config["lr"],
            )
        elif self.train_config["optimizer"] == "sgd":
            self.optimizer = torch.optim.SGD(
                [
                    {"params": self.model.parameters(), "weight_decay": 1e-3},
                    {"params": self.heatmap_generator.sigmas},
                    {"params": self.heatmap_generator.rotation},
                ],
                lr=self.train_config["lr"],
                momentum=0.99,
                nesterov=True,
            )
        else:
            raise ValueError("Optimizer not supported")
        # Set lr_scheduler
        if self.train_config["lr_scheduler"]:
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.train_config["lr_scheduler_factor"],
                patience=self.train_config["lr_scheduler_patience"],
                verbose=self.verbose,
                cooldown=self.train_config["lr_scheduler_cooldown"],
            )

    def __reduce__(self) -> str | tuple[Any, ...]:
        return (
            self.__class__,
            (
                self.sigma,
                self.alpha,
                self.model,
                self.in_channels_img,
                self.nb_landmarks,
                self.train_config,
                self.dim_img,
                self.heatmap_decoder,
                self.verbose,
                self.device,
                self.fitted,
            ),
        )

    def fit(
        self,
        imgs: torch.Tensor | np.ndarray | list[str],
        landmarks: torch.Tensor | np.ndarray,
        transform: Optional[Compose] = None,
        cache_ds: bool = True,
    ) -> None:
        assert len(imgs) == len(landmarks), "Number of images and landmarks must be equal"
        ds = LandmarkDataset(
            imgs, landmarks, transform=transform, store_imgs=cache_ds, dim_img=self.dim_img
        )
        split_lengths = [0.8, 0.2]
        ds_train, ds_val = torch.utils.data.random_split(ds, split_lengths)
        train_loader = DataLoader(
            ds_train,
            batch_size=self.train_config["batch_size"],
            shuffle=True,
        )
        val_loader = DataLoader(
            ds_val,
            batch_size=self.train_config["batch_size"],
            shuffle=False,
        )
        self.train(ds, train_loader, val_loader)
        self.fitted = True

    def fit_ds(self, ds: LandmarkDataset, transform: Optional[Compose] = None) -> None:
        if transform is not None:
            ds.transform = transform
        split_lengths = [0.8, 0.2]
        ds_train, ds_val = torch.utils.data.random_split(ds, split_lengths)
        train_loader = DataLoader(
            ds_train,
            batch_size=self.train_config["batch_size"],
            shuffle=True,
        )
        val_loader = DataLoader(
            ds_val,
            batch_size=self.train_config["batch_size"],
            shuffle=False,
        )
        self.train(ds, train_loader, val_loader)
        self.fitted = True

    def train_epoch(self, train_loader) -> float:
        running_loss = 0
        self.model.train()
        for i, (images, landmarks, _, _, _, _, _, _) in enumerate(tqdm(train_loader)):
            images = images.to(self.device)
            landmarks = landmarks.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            heatmaps = self.heatmap_generator(landmarks)
            loss = self.criterion(outputs, self.heatmap_generator.sigmas, heatmaps)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(train_loader)

    def val_epoch(self, val_loader) -> float:
        eval_loss = 0
        self.model.eval()
        with torch.no_grad():
            for i, (images, landmarks, _, _, _, _, _, _) in enumerate(tqdm(val_loader)):
                images = images.to(self.device)
                landmarks = landmarks.to(self.device)
                outputs = self.model(images)
                heatmaps = self.heatmap_generator(landmarks)
                loss = self.criterion(outputs, self.heatmap_generator.sigmas, heatmaps)
                eval_loss += loss.item()
        return eval_loss / len(val_loader)


class IndirectHeatmapRegressionPipeline(HeatmapRegressionPipeline):
    """
    IndirectHeatmapRegressionPipeline is a pipeline that predicts heatmaps for a given image.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set criterion
        self.background = False
        if isinstance(self.model, str):
            self.model: nn.Module = get_model(
                self.model,
                in_channels=self.in_channels_img,
                out_channels=self.nb_landmarks + int(self.background),
            )
        self.model.to(self.device)
        # Set optimizer
        if isinstance(self.train_config["optimizer"], torch.optim.Optimizer):
            self.optimizer = self.train_config["optimizer"](
                self.model.parameters(),
                lr=self.train_config["lr"],
                weight_decay=self.train_config["weight_decay"],
            )
        elif self.train_config["optimizer"] == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.train_config["lr"],
                weight_decay=self.train_config["weight_decay"],
            )
        elif self.train_config["optimizer"] == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.train_config["lr"],
                weight_decay=self.train_config["weight_decay"],
            )
        elif self.train_config["optimizer"] == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.train_config["lr"],
                weight_decay=self.train_config["weight_decay"],
            )
        else:
            raise ValueError("Optimizer not supported")
        # Set lr_scheduler
        if self.train_config["lr_scheduler"]:
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.train_config["lr_scheduler_factor"],
                patience=self.train_config["lr_scheduler_patience"],
                verbose=self.verbose,
                cooldown=self.train_config["lr_scheduler_cooldown"],
            )

    def __reduce__(self) -> str | tuple[Any, ...]:
        return (
            self.__class__,
            (
                self.model,
                self.in_channels_img,
                self.nb_landmarks,
                self.train_config,
                self.dim_img,
                self.heatmap_decoder,
                self.verbose,
                self.device,
                self.fitted,
            ),
        )

    def fit(
        self,
        imgs: torch.Tensor | np.ndarray | list[str],
        landmarks: torch.Tensor | np.ndarray,
        transform: Optional[Compose] = None,
        cache_ds: bool = True,
    ) -> None:
        assert len(imgs) == len(landmarks), "Number of images and landmarks must be equal"
        ds = LandmarkDataset(
            imgs, landmarks, transform=transform, store_imgs=cache_ds, dim_img=self.dim_img
        )
        split_lengths = [0.8, 0.2]
        ds_train, ds_val = torch.utils.data.random_split(ds, split_lengths)
        train_loader = DataLoader(
            ds_train,
            batch_size=self.train_config["batch_size"],
            shuffle=True,
        )
        val_loader = DataLoader(
            ds_val,
            batch_size=self.train_config["batch_size"],
            shuffle=False,
        )
        self.train(ds, train_loader, val_loader)
        self.fitted = True

    def fit_ds(self, ds: LandmarkDataset, transform: Optional[Compose] = None) -> None:
        if transform is not None:
            ds.transform = transform
        split_lengths = [0.8, 0.2]
        ds_train, ds_val = torch.utils.data.random_split(ds, split_lengths)
        train_loader = DataLoader(
            ds_train,
            batch_size=self.train_config["batch_size"],
            shuffle=True,
        )
        val_loader = DataLoader(
            ds_val,
            batch_size=self.train_config["batch_size"],
            shuffle=False,
        )
        self.train(ds, train_loader, val_loader)
        self.fitted = True


class IndirectUncertaintyAwareHeatmapRegressionPipeline(IndirectHeatmapRegressionPipeline):
    """
    IndirectUncertaintyAwareHeatmapRegressionPipeline is a pipeline that predicts heatmaps for a
        given image.
    """

    def __init__(self, differentiable_decoder="softargmax", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if differentiable_decoder == "softargmax":
            self.differentiable_decoder = coord_soft_argmax_cov
        elif differentiable_decoder == "weighted_spatial_mean":
            self.differentiable_decoder = coord_weighted_spatial_mean_cov
        else:
            raise ValueError("Differentiable decoder not supported")
        if isinstance(self.train_config["criterion"], nn.Module):
            self.criterion = self.train_config["criterion"]
        elif self.train_config["criterion"] == "euclidean_distance_variance":
            self.criterion = EuclideanDistanceVarianceReg()
        elif self.train_config["criterion"] == "multivariate_gaussian_nllloss":
            self.criterion = MultivariateGaussianNLLLoss()
        elif self.train_config["criterion"] == "star_loss":
            self.criterion = StarLoss()
        else:
            raise ValueError("Criterion not supported")

    def __reduce__(self) -> str | tuple[Any, ...]:
        return (
            self.__class__,
            (
                self.differentiable_decoder,
                self.model,
                self.in_channels_img,
                self.nb_landmarks,
                self.train_config,
                self.dim_img,
                self.heatmap_decoder,
                self.verbose,
                self.device,
                self.fitted,
            ),
        )

    def train_epoch(self, train_loader) -> float:
        running_loss = 0
        self.model.train()
        for i, (images, landmarks, _, _, _, _, _, _) in enumerate(tqdm(train_loader)):
            images = images.to(self.device)
            landmarks = landmarks.to(self.device)
            self.optimizer.zero_grad()
            pred_landmarks, cov = self.differentiable_decoder(self.model(images))
            loss = self.criterion(pred_landmarks, cov, landmarks)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(train_loader)

    def val_epoch(self, val_loader) -> float:
        eval_loss = 0
        self.model.eval()
        with torch.no_grad():
            for i, (images, landmarks, _, _, _, _, _, _) in enumerate(tqdm(val_loader)):
                images = images.to(self.device)
                landmarks = landmarks.to(self.device)
                pred_landmarks, cov = self.differentiable_decoder(self.model(images))
                loss = self.criterion(pred_landmarks, cov, landmarks)
                eval_loss += loss.item()
        return eval_loss / len(val_loader)


class IndirectUncertaintyUnawareHeatmapRegressionPipeline(IndirectHeatmapRegressionPipeline):
    """
    IndirectUncertaintyUnawareHeatmapRegressionPipeline is a pipeline that predicts heatmaps for a
        given image.
    """

    def __init__(self, differentiable_decoder="softargmax", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if differentiable_decoder == "softargmax":
            self.differentiable_decoder = coord_soft_argmax
        elif differentiable_decoder == "weighted_spatial_mean":
            self.differentiable_decoder = coord_weighted_spatial_mean
        else:
            raise ValueError("Differentiable decoder not supported")
        if isinstance(self.train_config["criterion"], nn.Module):
            self.criterion = self.train_config["criterion"]
        elif self.train_config["criterion"] == "mse":
            self.criterion = nn.MSELoss()
        elif self.train_config["criterion"] == "l1":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError("Criterion not supported")

    def __reduce__(self) -> str | tuple[Any, ...]:
        return (
            self.__class__,
            (
                self.differentiable_decoder,
                self.model,
                self.in_channels_img,
                self.nb_landmarks,
                self.train_config,
                self.dim_img,
                self.heatmap_decoder,
                self.verbose,
                self.device,
                self.fitted,
            ),
        )

    def train_epoch(self, train_loader) -> float:
        running_loss = 0
        self.model.train()
        for i, (images, landmarks, _, _, _, _, _, _) in enumerate(tqdm(train_loader)):
            images = images.to(self.device)
            landmarks = landmarks.to(self.device)
            self.optimizer.zero_grad()
            pred_landmarks = self.differentiable_decoder(self.model(images))
            loss = self.criterion(pred_landmarks, landmarks)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(train_loader)

    def val_epoch(self, val_loader) -> float:
        eval_loss = 0
        self.model.eval()
        with torch.no_grad():
            for i, (images, landmarks, _, _, _, _, _, _) in enumerate(tqdm(val_loader)):
                images = images.to(self.device)
                landmarks = landmarks.to(self.device)
                pred_landmarks = self.differentiable_decoder(self.model(images))
                loss = self.criterion(pred_landmarks, landmarks)
                eval_loss += loss.item()
        return eval_loss / len(val_loader)


class CoordinateRegressionPipeline:
    """
    CoordinateRegressionPipeline is a pipeline that predicts landmarks for a given image.
    """

    def __init__(
        self,
        model,
        in_channels_img,
        nb_landmarks,
        train_config,
        dim_img,
        verbose=False,
        device="cpu",
        fitted=False,
    ):
        self.model = model
        self.model.to(device)
        self.in_channels_img = in_channels_img
        self.nb_landmarks = nb_landmarks
        self.init_train_config(train_config)
        self.dim_img = dim_img
        self.fitted = fitted
        self.verbose = verbose
        self.device = device
        # Set EarlyStopping and SaveBestModel
        self.early_stopping = EarlyStopping(
            patience=self.train_config["patience"], verbose=self.verbose
        )
        self.save_best_model = SaveBestModel(verbose=self.verbose)
        if isinstance(self.train_config["criterion"], nn.Module):
            self.criterion = self.train_config["criterion"]
        elif self.train_config["criterion"] == "mse":
            self.criterion = nn.MSELoss()
        elif self.train_config["criterion"] == "l1":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError("Criterion not supported")
        if isinstance(self.model, str):
            self.model: nn.Module = get_model(
                self.model, in_channels=self.in_channels_img, out_channels=self.nb_landmarks * 2
            )
        # Set optimizer
        if isinstance(self.train_config["optimizer"], torch.optim.Optimizer):
            self.optimizer = self.train_config["optimizer"](
                self.model.parameters(),
                lr=self.train_config["lr"],
                weight_decay=self.train_config["weight_decay"],
            )
        elif self.train_config["optimizer"] == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.train_config["lr"],
                weight_decay=self.train_config["weight_decay"],
            )
        elif self.train_config["optimizer"] == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.train_config["lr"],
                weight_decay=self.train_config["weight_decay"],
            )
        elif self.train_config["optimizer"] == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.train_config["lr"],
                weight_decay=self.train_config["weight_decay"],
            )
        else:
            raise ValueError("Optimizer not supported")
        if self.train_config["lr_scheduler"]:
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.train_config["lr_scheduler_factor"],
                patience=self.train_config["lr_scheduler_patience"],
                verbose=self.verbose,
                cooldown=self.train_config["lr_scheduler_cooldown"],
            )

    def init_train_config(self, train_config: dict[str, Any]) -> None:
        self.train_config = train_config
        if "lr" not in self.train_config:
            self.train_config["lr"] = 1e-3
        if "batch_size" not in self.train_config:
            self.train_config["batch_size"] = 32
        if "epochs" not in self.train_config:
            self.train_config["epochs"] = 1000
        if "patience" not in self.train_config:
            self.train_config["patience"] = 20
        if "min_delta" not in self.train_config:
            self.train_config["min_delta"] = 0.0
        if "weight_decay" not in self.train_config:
            self.train_config["weight_decay"] = 0.0
        if "lr_scheduler" not in self.train_config:
            self.train_config["lr_scheduler"] = False
        if "lr_scheduler_factor" not in self.train_config:
            self.train_config["lr_scheduler_factor"] = 0.1
        if "lr_scheduler_patience" not in self.train_config:
            self.train_config["lr_scheduler_patience"] = 10
        if "lr_scheduler_cooldown" not in self.train_config:
            self.train_config["lr_scheduler_cooldown"] = 5
        if "optimizer" not in self.train_config:
            self.train_config["optimizer"] = "adam"
        if "criterion" not in self.train_config:
            self.train_config["criterion"] = "mse"

    def fit(
        self,
        imgs: torch.Tensor | np.ndarray | list[str],
        landmarks: torch.Tensor | np.ndarray,
        transform: Optional[Compose] = None,
        cache_ds: bool = True,
    ) -> None:
        assert len(imgs) == len(landmarks), "Number of images and landmarks must be equal"
        ds = LandmarkDataset(
            imgs, landmarks, transform=transform, store_imgs=cache_ds, dim_img=self.dim_img
        )
        split_lengths = [0.8, 0.2]
        ds_train, ds_val = torch.utils.data.random_split(ds, split_lengths)
        train_loader = DataLoader(
            ds_train,
            batch_size=self.train_config["batch_size"],
            shuffle=True,
        )
        val_loader = DataLoader(
            ds_val,
            batch_size=self.train_config["batch_size"],
            shuffle=False,
        )
        self.train(ds, train_loader, val_loader)
        self.fitted = True

    def fit_ds(self, ds: LandmarkDataset, transform: Optional[Compose] = None) -> None:
        if transform is not None:
            ds.transform = transform
        split_lengths = [0.8, 0.2]
        ds_train, ds_val = torch.utils.data.random_split(ds, split_lengths)
        train_loader = DataLoader(
            ds_train,
            batch_size=self.train_config["batch_size"],
            shuffle=True,
        )
        val_loader = DataLoader(
            ds_val,
            batch_size=self.train_config["batch_size"],
            shuffle=False,
        )
        self.train(ds, train_loader, val_loader)
        self.fitted = True

    def train_epoch(self, train_loader) -> float:
        running_loss = 0
        self.model.train()
        for i, (images, landmarks, _, _, _, _, _, _) in enumerate(tqdm(train_loader)):
            images = images.to(self.device)
            landmarks = landmarks.to(self.device)
            self.optimizer.zero_grad()
            pred_landmarks = self.model(images).view(-1, self.nb_landmarks, 2)
            loss = self.criterion(pred_landmarks, landmarks)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(train_loader)

    def val_epoch(self, val_loader) -> float:
        eval_loss = 0
        self.model.eval()
        with torch.no_grad():
            for i, (images, landmarks, _, _, _, _, _, _) in enumerate(tqdm(val_loader)):
                images = images.to(self.device)
                landmarks = landmarks.to(self.device)
                pred_landmarks = self.model(images).view(-1, self.nb_landmarks, 2)
                loss = self.criterion(pred_landmarks, landmarks)
                eval_loss += loss.item()
        return eval_loss / len(val_loader)

    def train(
        self,
        ds: LandmarkDataset,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        transform = ds.transform
        for epoch in range(self.train_config["epochs"]):
            train_loss = self.train_epoch(train_loader)
            ds.transform = None
            val_loss = self.val_epoch(val_loader)
            ds.transform = transform
            if self.verbose:
                print(
                    f"Epoch {epoch+1}/{self.train_config['epochs']} - Train loss: {train_loss:.8f}"
                    f" - Val loss: {val_loss:.8f}"
                )
            self.early_stopping(val_loss)
            self.save_best_model(val_loss, self.model)  # type: ignore
            if self.train_config["lr_scheduler"]:
                self.lr_scheduler.step(val_loss)  # type: ignore
            if self.early_stopping.early_stop:
                if self.verbose:
                    print("Loading best model...")
                self.model.load_state_dict(torch.load(self.save_best_model.path))  # type: ignore
                break

    def _predict(
        self,
        imgs: torch.Tensor | np.ndarray | list[str],
        landmarks: Optional[torch.Tensor | np.ndarray] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.fitted, "Model must be fitted first"
        pred_landmarks = torch.zeros((len(imgs), self.nb_landmarks, 2))
        dim_originals = torch.zeros((len(imgs), 2))
        paddings = torch.zeros((len(imgs), 2))
        if landmarks is not None:
            true_landmarks = torch.zeros((len(imgs), self.nb_landmarks, 2))
        ds = LandmarkDataset(imgs=imgs, landmarks=true_landmarks, transform=None, store_imgs=True)
        self.model.eval()
        with torch.no_grad():
            for i, (
                images,
                true_landmark,
                _,
                _,
                _,
                dim_original,
                _,
                padding,
            ) in enumerate(tqdm(ds)):
                images = images.unsqueeze(0).to(self.device)
                outputs = self.model(images).view(-1, self.nb_landmarks, 2)
                pred_landmarks[i] = outputs.cpu()
                paddings[i] = padding
                dim_originals[i] = dim_original
                if landmarks is not None:
                    true_landmarks[i] = true_landmark
        if landmarks is not None:
            return pred_landmarks, true_landmarks, dim_originals, paddings
        return pred_landmarks

    def predict(self, imgs: torch.Tensor | np.ndarray) -> torch.Tensor:
        pred_landmarks, dim_original, padding = self._predict(imgs)  # type: ignore
        return pixel_to_unit(pred_landmarks, dim_orig=dim_original, padding=padding)

    def evaluate(
        self,
        imgs: torch.Tensor | np.ndarray | list[str],
        landmarks: torch.Tensor | np.ndarray,
        pixel_spacing: Optional[torch.Tensor] = None,
    ) -> Any:
        assert self.fitted, "Model must be fitted first"
        pred_landmarks, true_landmarks, dim_original, padding = self._predict(imgs, landmarks)
        if pixel_spacing is None:
            pixel_spacing = torch.ones((len(imgs), 2))
        return detection_report(
            true_landmarks=true_landmarks,
            pred_landmarks=pred_landmarks,
            dim_orig=dim_original,
            dim=self.dim_img,
            padding=padding,
            pixel_spacing=pixel_spacing,
            output_dict=True,
        )
