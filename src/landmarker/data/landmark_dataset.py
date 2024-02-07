"""Module containing the ``LandmarkDataset`` class and its subclasses ``MaskDataset``
and ``HeatmapDataset``. All three classe allow to create a dataset of images and landmarks.
The three classes are subclasses of ``torch.utils.data.Dataset``.
"""

from typing import Callable, Optional

import cv2
import monai
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T  # type: ignore
from monai.transforms import Compose, Flip  # type: ignore
from torch.utils.data import Dataset
from tqdm import tqdm  # type: ignore

from landmarker.heatmap.generator import (
    GaussianHeatmapGenerator,
    HeatmapGenerator,
    LaplacianHeatmapGenerator,
)
from landmarker.transforms.heatmaps import flip_heatmaps
from landmarker.transforms.images import resize_with_pad
from landmarker.transforms.landmarks import (
    affine_landmarks,
    flip_landmarks,
    resize_landmarks,
)


class LandmarkDataset(Dataset):
    """
    ``LandmarkDataset`` is a subclass of ``torch.utils.data.Dataset``. It represents a dataset of
    images and landmarks. The images can be provided as a list of paths to the images or as a list
    of numpy arrays or as a numpy array/torch.Tensor.

    Args:
        imgs (list[str] | list[np.array] | np.ndarray | torch.Tensor): list of paths to the images
            or list of numpy arrays or numpy array/torch.Tensor.
        landmarks (np.ndarray | torch.Tensor): landmarks of the images.
        pixel_spacing (Optional[np.ndarray | torch.Tensor]): pixel spacing of the images.
            (defaults: None)
        class_names (Optional[list]): names of the landmarks. (defaults: None)
        transform (Optional[Callable]): transformation to apply to the images and landmarks.
            (defaults: None)
        store_imgs (bool): whether to store the images in memory or not. (defaults: True)
        dim_img (Optional[tuple[int, int]]): dimension of the images. (defaults: None)
        img_paths (Optional[list[str]]): list of paths to the images. (defaults: None)
        grayscale (bool): whether the images are grayscale or not. (defaults: True)
        resize_pad (bool): whether to resize the images and landmarks or not. (defaults: True)
        normalize_intensity (bool): whether to normalize the intensity of the images or not.
            (defaults: True)
    """

    def __init__(
        self,
        imgs: list[str] | np.ndarray | torch.Tensor | list[np.ndarray] | list[torch.Tensor],
        landmarks: np.ndarray | torch.Tensor,
        pixel_spacing: Optional[list[int] | tuple[int, int] | np.ndarray | torch.Tensor] = None,
        class_names: Optional[list] = None,
        transform: Optional[Callable] = None,
        store_imgs: bool = True,
        dim_img: Optional[tuple[int, int]] = None,
        img_paths: Optional[list[str]] = None,
        grayscale: bool = True,
        resize_pad: bool = True,
        normalize_intensity: bool = True,
        flip_aug_h: bool = False,
        flip_aug_v: bool = False,
        flip_indices_h: Optional[list[int]] = None,
        flip_indices_v: Optional[list[int]] = None,
    ) -> None:
        if len(landmarks.shape) == 2:
            landmarks = landmarks.reshape(
                (landmarks.shape[0], 1, landmarks.shape[1])
            )  # (N, D) => (N, 1, D) only one class
        self.nb_landmarks = landmarks.shape[1]
        self.grayscale = grayscale
        self.resize_pad = resize_pad
        self.normalize_intensity = normalize_intensity
        self.flip_aug_h = flip_aug_h
        self.flip_aug_v = flip_aug_v
        self.ds_size_factor = 2 ** (int(self.flip_aug_h) + int(self.flip_aug_v))
        if flip_aug_h:
            if flip_indices_h is None:
                flip_indices_h = list(range(self.nb_landmarks))
            assert len(flip_indices_h) == self.nb_landmarks
            self.flip_indices_h = flip_indices_h
        if flip_aug_v:
            if flip_indices_v is None:
                flip_indices_v = list(range(self.nb_landmarks))
            assert len(flip_indices_v) == self.nb_landmarks
            self.flip_indices_v = flip_indices_v
        if isinstance(imgs, list):
            if isinstance(imgs[0], str):
                self.img_paths = imgs
            elif isinstance(imgs[0], np.ndarray):
                self.imgs = [torch.Tensor(imgs[i]) for i in range(len(imgs))]
                assert len(self.imgs[0].shape) == 3
            elif isinstance(imgs[0], torch.Tensor):
                self.imgs = imgs  # type: ignore
                assert len(self.imgs[0].shape) == 3
            else:
                raise TypeError("imgs type not supported")
        else:
            self.imgs = [torch.Tensor(imgs[i]) for i in range(len(imgs))]
            assert len(self.imgs[0].shape) == 3  # (C, H, W)
            self.img_paths = img_paths if img_paths is not None else []
        if isinstance(landmarks, np.ndarray):
            self.landmarks_original = torch.Tensor(landmarks.copy())
        elif isinstance(landmarks, torch.Tensor):
            self.landmarks_original = landmarks.clone().float()
        else:
            if landmarks is None:
                raise ValueError("landmarks must be provided")
            raise TypeError("landmarks type not supported")
        if pixel_spacing is not None:
            if isinstance(pixel_spacing, (tuple, list)) or len(pixel_spacing.shape) == 1:
                self.pixel_spacings = (
                    torch.Tensor(pixel_spacing).unsqueeze(0).repeat(len(self.landmarks_original), 1)
                )
            elif pixel_spacing.shape == (len(self.landmarks_original), 2):
                self.pixel_spacings = torch.Tensor(pixel_spacing)
            else:
                raise ValueError("pixel_spacing must be of shape (N, 2) or (2,)")
        else:
            self.pixel_spacings = torch.ones(len(self.landmarks_original), 2)

        if class_names is None:
            self.class_names = [f"landmark_{i}" for i in range(self.landmarks_original.shape[1])]
        else:
            if len(class_names) != self.landmarks_original.shape[1]:
                raise ValueError("class_names must be of length landmarks.shape[1] or None")
            self.class_names = class_names
        self.transform = transform
        self.store_imgs = store_imgs
        self.dim_img = dim_img
        # images need to be stored in memory and images are provided as paths
        if store_imgs and not hasattr(self, "imgs"):
            assert self.img_paths is not None
            if self.dim_img is not None:
                (
                    self.imgs,
                    self.landmarks,
                    self.dim_original,
                    self.paddings,
                ) = self._read_norm__resize_all_imgs_landmarks(
                    self.img_paths, self.landmarks_original, self.dim_img  # type: ignore
                )
            else:
                self.imgs, self.dim_original = self._read_norm_all_imgs(
                    self.img_paths  # type: ignore
                )
                self.landmarks = self.landmarks_original.clone()
                self.paddings = torch.zeros((self.landmarks.shape[0], 2))
        # Images need to be stored in memory and images are provided as numpy arrays or torch.Tensor
        # and need to be resized.
        elif dim_img is not None and hasattr(self, "imgs"):
            (
                self.imgs,
                self.landmarks,
                self.dim_original,
                self.paddings,
            ) = self._resize_all_imgs_landmarks(self.imgs, self.landmarks_original, dim_img)
        # Images need to be stored in memory and images are provided as numpy arrays or torch.Tensor
        elif hasattr(self, "imgs"):
            self.landmarks = self.landmarks_original.clone()
            self.dim_original = torch.zeros((self.landmarks.shape[0], 2))
            self.paddings = torch.zeros((self.landmarks.shape[0], 2))
            for i in range(self.landmarks.shape[0]):
                self.dim_original[i] = torch.tensor(self.imgs[i].shape[-2:]).float()

    def __len__(self) -> int:
        return len(self.landmarks_original) * self.ds_size_factor

    def __getitem__(self, idx: int) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        flip_code = idx % self.ds_size_factor
        idx = idx // self.ds_size_factor
        if not self.store_imgs:
            img = self._read_norm_img(self.img_paths[idx])  # type: ignore
            dim_original = torch.tensor(img.shape[-2:]).float()
            if self.dim_img is not None:
                if self.resize_pad:
                    dim_original = torch.tensor(img.shape[-2:]).float()
                    img, (hp, wp) = resize_with_pad(img, self.dim_img)
                    padding = torch.Tensor([hp, wp])
                    landmark = resize_landmarks(
                        self.landmarks_original[idx], dim_original, self.dim_img, padding
                    )
                else:
                    img = T.Resize(self.dim_img)(img)
                    landmark = resize_landmarks(
                        self.landmarks_original[idx], dim_original, self.dim_img
                    )
                    padding = torch.zeros(2)
            else:
                landmark = self.landmarks_original[idx]
                padding = torch.zeros(2)
        else:
            img = self.imgs[idx]
            landmark = self.landmarks[idx]
            padding = self.paddings[idx]
            dim_original = self.dim_original[idx]
        if self.ds_size_factor > 1:
            if self.ds_size_factor == 2 and flip_code:
                if self.flip_aug_h:
                    img = Flip(0)(img)
                else:
                    img = Flip(1)(img)
                    flip_code += 1
            elif self.ds_size_factor == 4:
                if flip_code == 1 or flip_code == 3:
                    img = Flip(0)(img)
                if flip_code == 2 or flip_code == 3:
                    img = Flip(1)(img)
            landmark = flip_landmarks(
                landmark,
                flip_code,
                flip_indices_v=self.flip_indices_v,
                flip_indices_h=self.flip_indices_h,
            )
        if self.transform is not None:
            img, landmark_t, affine_matrix = self._transform_img_landmark(
                img, landmark, self.transform  # type: ignore
            )
        else:
            landmark_t = landmark
            affine_matrix = torch.eye(4)
        if self.normalize_intensity:
            img = monai.transforms.NormalizeIntensity()(img)  # type: ignore
        return (
            img,
            landmark_t,
            affine_matrix,
            landmark,
            self.landmarks_original[idx],
            dim_original,
            self.pixel_spacings[idx],
            padding,
        )

    def _transform_img_landmark(
        self, img: torch.Tensor, landmark: torch.Tensor, transform: Compose
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transform the image and the landmark using the transformation arg transform.

        Args:
            img (torch.Tensor): image to transform.
            landmark (torch.Tensor): landmark to transform.
            transform (Callable): transformation to apply to the image and the landmark.

        Returns:
            img_t (torch.Tensor): transformed image.
            landmark_t (torch.Tensor): transformed landmark.
            affine_matrix_push (torch.Tensor): push affine matrix of the transformation.
        """
        img_t = transform({"image": img})["image"]  # type: ignore
        affine_matrix_pull = img_t.meta.get("affine", None).float()  # type: ignore
        affine_matrix_push = torch.linalg.inv(affine_matrix_pull)
        landmark_t = affine_landmarks(landmark, affine_matrix_push)
        return img_t, landmark_t, affine_matrix_push

    def _read_norm_img(self, path: str) -> torch.Tensor:
        """
        Read and normalize an image. Only uint8 and uint16 images are supported. Image gets read as
        grayscale if ``self.grayscale`` is True and gets normalized to [0, 1].

        Args:
            path (str): path to the image.

        Returns:
            img (torch.Tensor): normalized image. (C, H, W).
        """
        if path.endswith(".npy"):
            return torch.Tensor(np.load(path))
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if self.grayscale and len(img.shape) == 3:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image {path} not found")
        if len(img.shape) == 3:
            # put channels first
            img = img.transpose(2, 0, 1)
        if img.dtype == np.uint16:
            return torch.Tensor(img.astype(np.float32) / 65535.0).view(
                -1, img.shape[-2], img.shape[-1]
            )
        if img.dtype == np.uint8:
            return torch.Tensor(img.astype(np.float32) / 255.0).view(
                -1, img.shape[-2], img.shape[-1]
            )
        raise TypeError("img type not supported")

    def _read_norm_all_imgs(self, paths: list[str]) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Read and normalize all images in ``paths``.

        Args:
            paths (list[str]): list of paths to the images.

        Returns:
            imgs (list[torch.Tensor]): list of normalized images.
            dim_original (torch.Tensor): original dimensions of the images.
        """
        print(f"Reading and normalizing {len(paths)} images...")
        imgs = []
        dim_original = torch.zeros((len(paths), 2))
        for i, path in enumerate(tqdm(paths)):
            img = self._read_norm_img(path)
            dim_original[i] = torch.tensor(img.shape[-2:]).float()
            imgs.append(img)
        return imgs, dim_original

    def _read_norm__resize_all_imgs_landmarks(
        self, paths: list[str], landmarks: torch.Tensor, dim_img: tuple[int, int]
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Read, normalize and resize all images in ``paths`` and resize all landmarks in
            ``landmarks``.

        Args:
            paths (list[str]): list of paths to the images.
            landmarks (torch.Tensor): landmarks of the images.
            dim_img (tuple[int, int]): dimension of the images.

        Returns:
            imgs (list[torch.Tensor]): list of normalized and resized images.
            landmarks (torch.Tensor): resized landmarks.
            dim_original (torch.Tensor): original dimensions of the images.
            paddings (torch.Tensor): paddings applied to the images.
        """
        print(f"Reading, normalizing and resizing {len(paths)} images and landmarks...")
        imgs = []
        landmarks = landmarks.clone()
        dim_original = torch.zeros((len(paths), 2))
        paddings = torch.zeros((len(paths), 2))
        for i, path in enumerate(tqdm(paths)):
            img = self._read_norm_img(path)
            dim_original[i] = torch.tensor(img.shape[-2:]).float()
            if self.resize_pad:
                img, padding = resize_with_pad(img, dim_img)
                paddings[i] = torch.Tensor([padding[0], padding[1]])
            else:
                img = T.Resize(dim_img)(img)
                paddings[i] = torch.zeros(2)
            landmarks[i] = resize_landmarks(landmarks[i], dim_original[i], dim_img, paddings[i])
            imgs.append(img)
        return imgs, landmarks, dim_original, paddings

    def _resize_all_imgs_landmarks(
        self, imgs: list[torch.Tensor], landmarks: torch.Tensor, dim_img: tuple[int, int]
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Resize all images in ``imgs`` and resize all landmarks in ``landmarks``.

        Args:
            imgs (torch.Tensor): images.
            landmarks (torch.Tensor): landmarks of the images.
            dim_img (tuple[int, int]): dimension of the images.

        Returns:
            imgs (list[torch.Tensor]): list of resized images.
            landmarks (torch.Tensor): resized landmarks.
            dim_original (torch.Tensor): original dimensions of the images.
            paddings (torch.Tensor): paddings applied to the images.
        """
        print(f"Resizing {len(imgs)} images and landmarks...")
        landmarks = landmarks.clone()
        dim_original = torch.zeros((len(imgs), 2))
        paddings = torch.zeros((len(imgs), 2))
        for i in tqdm(range(len(imgs))):
            dim_original[i] = torch.tensor(imgs[i].shape[-2:]).float()
            if self.resize_pad:
                imgs[i], padding = resize_with_pad(imgs[i], dim_img)
                paddings[i] = torch.Tensor([padding[0], padding[1]])
            else:
                imgs[i] = T.Resize(dim_img)(imgs[i])
                paddings[i] = torch.zeros(2)
            landmarks[i] = resize_landmarks(landmarks[i], dim_original[i], dim_img, paddings[i])
        return imgs, landmarks, dim_original, paddings


class MaskDataset(LandmarkDataset):
    """
    ``MaskDataset`` is a subclass of ``LandmarkDataset``. It represents a dataset of images
    and landmarks. The images can be provided as a list of paths to the images or as a list of
    numpy arrays or as a numpy array/torch.Tensor. The landmarks can be provided as a list of paths
    to the masks or as a list of numpy arrays or as a numpy array/torch.Tensor. The masks must be
    grayscale images with the landmarks represented by different values. The landmarks can be
    extracted from the masks and stored as a torch.Tensor or they can be genererated from provided
    landmarks. Note that if mask paths are provide this can be only be grayscale images and thus
    the dataset can only contain one class of landmark per image.

    Args:
        imgs (list[str] | list[np.array] | np.ndarray | torch.Tensor): list of paths to the images
            or list of numpy arrays or numpy array/torch.Tensor.
        landmarks (np.ndarray | torch.Tensor): landmarks of the images.
        pixel_spacing (Optional[np.ndarray | torch.Tensor]): pixel spacing of the images.
            (defaults: None)
        class_names (Optional[list]): names of the landmarks. (defaults: None)
        transform (Optional[Callable]): transformation to apply to the images and landmarks.
            (defaults: None)
        store_masks_imgs (bool): whether to store the imgs and masks in memory or not.
            (defaults: True)
        dim_img (Optional[tuple[int, int]]): dimension of the images. (defaults: None)
        mask_paths (Optional[list[str]]): list of paths to the masks. (defaults: None)
        nb_landmarks (int): number of landmarks in the masks. (defaults: 1)
        img_paths (Optional[list[str]]): list of paths to the images. (defaults: None)
        grayscale (bool): whether the images are grayscale or not. (defaults: True)
        resize_pad (bool): whether to resize the images and landmarks or not. (defaults: True)
        normalize_intensity (bool): whether to normalize the intensity of the images or not.
            (defaults: True)
    """

    def __init__(
        self,
        imgs: list[str] | list[np.ndarray] | np.ndarray | torch.Tensor,
        landmarks: Optional[np.ndarray | torch.Tensor] = None,
        pixel_spacing: Optional[torch.Tensor] = None,
        class_names: Optional[list] = None,
        transform: Optional[Callable] = None,
        store_masks_imgs: bool = True,
        dim_img: Optional[tuple[int, int]] = None,
        mask_paths: Optional[list[str]] = None,
        nb_landmarks: int = 1,
        img_paths: Optional[list[str]] = None,
        grayscale: bool = True,
        resize_pad: bool = True,
        normalize_intensity: bool = True,
        flip_aug_h: bool = False,
        flip_aug_v: bool = False,
        flip_indices_h: Optional[list[int]] = None,
        flip_indices_v: Optional[list[int]] = None,
    ) -> None:
        self.nb_landmarks = nb_landmarks
        self.mask_paths = mask_paths
        self.dim_img = dim_img
        self.resize_pad = resize_pad
        self.store_masks_imgs = store_masks_imgs
        store_imgs = store_masks_imgs
        if self.mask_paths is None and landmarks is not None:
            super().__init__(
                imgs,
                landmarks,
                pixel_spacing,
                class_names,
                transform,
                store_imgs,
                dim_img,
                img_paths=img_paths,
                grayscale=grayscale,
                resize_pad=resize_pad,
                normalize_intensity=normalize_intensity,
                flip_aug_h=flip_aug_h,
                flip_aug_v=flip_aug_v,
                flip_indices_h=flip_indices_h,
                flip_indices_v=flip_indices_v,
            )
            if self.dim_img is not None:
                self.masks = self._create_masks(self.landmarks, self.dim_img)
            else:
                self.masks = self._create_masks(self.landmarks, self.dim_original)
        elif self.mask_paths is not None and landmarks is None:
            if self.dim_img is not None:
                self.masks, landmarks = self._read_extract_resize_masks_landmarks(
                    self.mask_paths, self.dim_img
                )
            else:
                self.masks, landmarks = self._read_extract_masks_landmarks(self.mask_paths)
            super().__init__(
                imgs,
                landmarks,
                pixel_spacing,
                class_names,
                transform,
                store_imgs,
                dim_img,
                img_paths=img_paths,
                grayscale=grayscale,
                resize_pad=resize_pad,
                normalize_intensity=normalize_intensity,
                flip_aug_h=flip_aug_h,
                flip_aug_v=flip_aug_v,
                flip_indices_h=flip_indices_h,
                flip_indices_v=flip_indices_v,
            )
        else:
            raise ValueError("Either mask_paths or landmarks must be provided")

    def __getitem__(self, idx: int) -> tuple[  # type: ignore
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        flip_code = idx % self.ds_size_factor
        idx = idx // self.ds_size_factor
        if not self.store_masks_imgs:
            img = self._read_norm_img(self.img_paths[idx])  # type: ignore
            dim_original = torch.tensor(img.shape[-2:]).float()
            if self.dim_img is not None:
                if self.resize_pad:
                    dim_original = torch.tensor(img.shape[-2:]).float()
                    img, (hp, wp) = resize_with_pad(img, self.dim_img)
                    padding = torch.Tensor([hp, wp])
                    landmark = resize_landmarks(
                        self.landmarks_original[idx], dim_original, self.dim_img, padding
                    )
                else:
                    img = T.Resize(self.dim_img)(img)
                    landmark = resize_landmarks(
                        self.landmarks_original[idx], dim_original, self.dim_img
                    )
                    padding = torch.zeros(2)
                mask = self._create_mask(landmark, self.dim_img)
            else:
                landmark = self.landmarks_original[idx]
                padding = torch.zeros(2)
                mask = self._create_mask(
                    landmark, (int(dim_original[0].item()), int(dim_original[1].item()))
                )
        else:
            img = self.imgs[idx]
            mask = self.masks[idx]
            landmark = self.landmarks[idx]
            padding = self.paddings[idx]
        if self.ds_size_factor > 1:
            if self.ds_size_factor == 2 and flip_code:
                if self.flip_aug_h:
                    img = Flip(0)(img)
                    mask = Flip(0)(mask)
                else:
                    img = Flip(1)(img)
                    mask = Flip(1)(mask)
                    flip_code += 1
            elif self.ds_size_factor == 4:
                if flip_code == 1 or flip_code == 3:
                    img = Flip(0)(img)
                    mask = Flip(0)(mask)
                if flip_code == 2 or flip_code == 3:
                    img = Flip(1)(img)
                    mask = Flip(1)(mask)
            landmark = flip_landmarks(
                landmark,
                flip_code,
                flip_indices_v=self.flip_indices_v,
                flip_indices_h=self.flip_indices_h,
            )
            mask = flip_heatmaps(
                mask,
                flip_code,
                flip_indices_v=self.flip_indices_v,
                flip_indices_h=self.flip_indices_h,
            )
        if self.transform is not None:
            img, mask, landmark_t, affine_matrix = self._transform_img_mask_landmark(
                img, mask, landmark, self.transform
            )
        else:
            landmark_t = landmark
            affine_matrix = torch.eye(4)
        if self.normalize_intensity:
            img = monai.transforms.NormalizeIntensity()(img)  # type: ignore[assignment]
        return (
            img,
            mask,
            landmark_t,
            affine_matrix,
            landmark,
            self.landmarks_original[idx],
            self.dim_original[idx],
            self.pixel_spacings[idx],
            padding,
        )

    def _transform_img_mask_landmark(
        self, img: torch.Tensor, mask: torch.Tensor, landmark: torch.Tensor, transform: Callable
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transform the image, the mask and the landmark using the transformation ``self.transform``.

        Args:
            img (torch.Tensor): image to transform.
            mask (torch.Tensor): mask to transform.
            landmark (torch.Tensor): landmark to transform.
            transform (Callable): transformation to apply to the image, the mask and the landmark.

        Returns:
            img_t (torch.Tensor): transformed image.
            mask_t (torch.Tensor): transformed mask.
            landmark_t (torch.Tensor): transformed landmark.
            affine_matrix_push (torch.Tensor): push affine matrix of the transformation.
        """
        out = transform({"image": img, "seg": mask.squeeze(0)})
        img_t = out["image"]
        mask_t = out["seg"]
        affine_matrix_pull = img_t.meta.get("affine", None).float()
        affine_matrix_push = torch.linalg.inv(affine_matrix_pull)
        landmark_t = affine_landmarks(landmark, affine_matrix_push)
        return img_t, mask_t, landmark_t, affine_matrix_push

    def _create_masks(
        self, landmarks: torch.Tensor, dim: tuple[int, int] | torch.Tensor
    ) -> list[torch.Tensor]:
        """
        Create masks from landmarks.

        Args:
            landmarks (torch.Tensor): landmarks of the images.
            dim (tuple[int, int] | torch.Tensor): dimension of the images.

        Returns:
            masks (list[torch.Tensor]): list of masks.
        """
        print(f"Creating masks for {len(landmarks)} images...")
        masks = []
        for i in tqdm(range(landmarks.shape[0])):
            if isinstance(dim, tuple):
                masks.append(self._create_mask(landmarks[i], dim))
            else:
                if dim[i].shape == (2,):
                    masks.append(
                        self._create_mask(
                            landmarks[i], (int(dim[i, 0].item()), int(dim[i, 0].item()))
                        )
                    )
                else:
                    raise ValueError("dim (tensor) must be of shape (N, 2)")
        return masks

    def _create_mask(self, landmark: torch.Tensor, dim: tuple[int, int]) -> torch.Tensor:
        """
        Create a mask from a landmark.

        Args:
            landmark (torch.Tensor): landmark of the image.
            dim (tuple[int, int]): dimension of the image.

        Returns:
            mask (torch.Tensor): mask.
        """
        mask = torch.zeros((landmark.shape[0], dim[0], dim[1]))
        for i in range(landmark.shape[0]):
            if len(landmark.shape) == 3:
                for j in range(landmark.shape[1]):
                    mask[i, int(landmark[i, j, 0]), int(landmark[i, j, 1])] = 1
            else:
                mask[i, int(landmark[i, 0]), int(landmark[i, 1])] = 1
        return mask

    def _read_extract_masks_landmarks(
        self, paths: list[str]
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Read masks and extract landmarks from these masks.

        Args:
            paths (list[str]): list of paths to the masks.

        Returns:
            masks (list[torch.Tensor]): list of masks.
            landmarks (torch.Tensor): landmarks.
        """
        print(f"Reading and extracting masks from {len(paths)} images...")
        masks = []
        landmarks = []
        for path in tqdm(paths):
            mask, landmark = self._read_extract_mask_landmark(path)
            if self.store_masks_imgs:
                masks.append(mask)
            landmarks.append(landmark)
        max_len = 1
        for landmark in landmarks:
            max_len = max(landmark.shape[1], max_len)
        if max_len != 1:
            landmarks = [
                F.pad(landmark, (0, 0, 0, max_len - landmark.shape[1]), value=torch.nan)
                for landmark in landmarks
            ]
        else:
            # Remove sample channel if every class has only one landmark
            landmarks = [landmark.squeeze(0) for landmark in landmarks]
        return masks, torch.stack(landmarks)

    def _read_extract_mask_landmark(self, path: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Read a mask and extract the landmark from this mask.

        Args:
            path (str): path to the mask.

        Returns:
            mask (torch.Tensor): mask.
            landmarks (torch.Tensor): landmark.
        """
        if path.endswith(".npy"):
            mask = torch.Tensor(np.load(path))
        else:
            mask = torch.Tensor(cv2.imread(path, cv2.IMREAD_UNCHANGED))
        mask = mask.view(1, mask.shape[0], mask.shape[1])
        landmarks = self._extract_landmark_from_mask(mask)
        return mask, landmarks

    def _extract_landmark_from_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Extract the landmark from a mask.

        Args:
            mask (torch.Tensor): mask.

        Returns:
            landmarks (torch.Tensor): landmark.
        """
        landmarks_list = []
        max_len = 0
        if self.nb_landmarks == 1:
            landmarks_list.append(mask[0].nonzero().float())
            max_len = max(len(landmarks_list[0].shape), max_len)
        else:
            for i in range(self.nb_landmarks):
                landmarks_list.append((mask[0] == i + 1).nonzero().float())
                max_len = max(len(landmarks_list[i].shape), 0)
        if max_len == 0:
            raise ValueError("No landmark found in mask")
        if max_len == 1 or self.nb_landmarks == 1:
            landmarks = torch.stack(landmarks_list)
        else:
            landmarks_list = [landmark.view(self.nb_landmarks, -1, 2) for landmark in landmarks]
            landmarks_list = [
                F.pad(landmark, (0, 0, 0, max_len - landmark.shape[1]), value=torch.nan)
                for landmark in landmarks_list
            ]
            landmarks = torch.cat(landmarks_list)
        return landmarks

    def _read_extract_resize_masks_landmarks(
        self, paths: list[str], dim: tuple[int, int]
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Read masks, extract landmarks from these masks and resize the masks and the landmarks.

        Args:
            paths (list[str]): list of paths to the masks.
            dim (tuple[int, int]): dimension of the masks.

        Returns:
            masks (list[torch.Tensor]): list of resized masks.
            landmarks (torch.Tensor): resized landmarks.
        """
        masks = []
        landmarks = []
        for path in tqdm(paths):
            mask, landmark = self._read_extract_resize_mask_landmark(path, dim)
            if self.store_masks_imgs:
                masks.append(mask)
            landmarks.append(landmark)
        max_len = 1
        for landmark in landmarks:
            max_len = max(landmark.shape[1], max_len)
        if max_len != 1:
            landmarks = [
                F.pad(landmark, (0, 0, 0, max_len - landmark.shape[1]), value=torch.nan)
                for landmark in landmarks
            ]
        else:
            # Remove sample channel if every class has only one landmark
            landmarks = [landmark.squeeze(0) for landmark in landmarks]
        return masks, torch.stack(landmarks)

    def _read_extract_resize_mask_landmark(
        self, path: str, dim: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Read a mask, extract the landmark from this mask and resize the mask.

        Args:
            path (str): path to the mask.
            dim (tuple[int, int]): dimension of the mask.

        Returns:
            mask (torch.Tensor): resized mask.
            landmarks (torch.Tensor): landmark.
        """
        mask, landmark = self._read_extract_mask_landmark(path)
        if self.resize_pad:
            mask, _ = resize_with_pad(mask, dim)
        else:
            mask = T.Resize(dim)(mask)
        return mask, landmark


class HeatmapDataset(LandmarkDataset):
    """
    ``HeatmapDataset`` is a subclass of ``LandmarkDataset``. It represents a dataset of images and
    landmarks. The images can be provided as a list of paths to the images or as a list of numpy
    arrays or as a numpy array/torch.Tensor. Th landmarks must be provided. The heatmaps are created
    from the landmarks using a Gaussian or Laplacian function. When using the ``HeatmapDataset``,
    images are always stored in memory.

    Args:
        imgs (list[str] | list[np.array] | np.ndarray | torch.Tensor): list of paths to the images
            or list of numpy arrays or numpy array/torch.Tensor.
        landmarks (np.ndarray | torch.Tensor): landmarks of the images.
        pixel_spacing (Optional[np.ndarray | torch.Tensor]): pixel spacing of the images.
            (defaults: None)
        class_names (Optional[list]): names of the landmarks. (defaults: None)
        transform (Optional[Callable]): transformation to apply to the images and landmarks.
            (defaults: None)
        sigma (float | list[float]): sigma of the Gaussian or Laplacian function. (defaults: 5)
        dim_img (Optional[tuple[int, int]]): dimension of the images. (defaults: None)
        batch_size (int): batch size to use when creating the heatmaps. (defaults: 32)
        full_map (bool): whether to create a full map or not. (defaults: True)
        background (bool): whether to add a background channel or not. (defaults: False)
        all_points (bool): whether to add a channel for all points or not. (defaults: False)
        gamma (Optional[float]): gamma of the Gaussian or Laplacian function. (defaults: None)
        heatmap_fun (str): function to use to create the heatmaps. (defaults: "gaussian")
        heatmap_size (Optional[tuple[int, int]]): size of the heatmaps. (defaults: None)
        grayscale (bool): whether the images are grayscale or not. (defaults: True)
        img_paths (Optional[list[str]]): list of paths to the images. (defaults: None)
        resize_pad (bool): whether to resize the images and landmarks or not. (defaults: True)
        normalize_intensity (bool): whether to normalize the intensity of the images or not.
            (defaults: True)
    """

    def __init__(
        self,
        imgs: list[str] | list[np.ndarray] | np.ndarray | torch.Tensor,
        landmarks: np.ndarray | torch.Tensor,
        pixel_spacing: Optional[np.ndarray | torch.Tensor] = None,
        class_names: Optional[list] = None,
        transform: Optional[Callable] = None,
        store_imgs: bool = True,
        sigma: float | list[float] | np.ndarray | torch.Tensor = 5,
        dim_img: Optional[tuple[int, int]] = None,
        batch_size: int = 32,
        full_map: bool = True,
        background: bool = False,
        all_points: bool = False,
        gamma: Optional[float] = None,
        heatmap_fun: str = "gaussian",
        heatmap_size: Optional[tuple[int, int]] = None,
        img_paths: Optional[list[str]] = None,
        grayscale: bool = True,
        resize_pad: bool = True,
        normalize_intensity: bool = True,
        flip_aug_h: bool = False,
        flip_aug_v: bool = False,
        flip_indices_h: Optional[list[int]] = None,
        flip_indices_v: Optional[list[int]] = None,
    ) -> None:
        super().__init__(
            imgs,
            landmarks,
            pixel_spacing,
            class_names,
            transform,
            store_imgs=store_imgs,
            dim_img=dim_img,
            img_paths=img_paths,
            grayscale=grayscale,
            resize_pad=resize_pad,
            normalize_intensity=normalize_intensity,
            flip_aug_h=flip_aug_h,
            flip_aug_v=flip_aug_v,
            flip_indices_h=flip_indices_h,
            flip_indices_v=flip_indices_v,
        )
        if heatmap_size is None:
            if self.dim_img is not None:
                heatmap_size = self.dim_img
            else:
                assert (
                    self.store_imgs
                ), "Heatmap size must be provided if imgs are not stored, and dim_img is None"
                # heatmap size becomes the size of the first original image
                heatmap_size = (
                    int(self.dim_original[0, 0].item()),
                    int(self.dim_original[0, 1].item()),
                )
        self.background = background
        self.all_points = all_points
        self.heatmap_generator: HeatmapGenerator
        if flip_aug_h:
            self.flip_indices_h = [
                i + int(self.background or self.all_points) for i in self.flip_indices_h
            ]
        if flip_aug_v:
            self.flip_indices_v = [
                i + int(self.background or self.all_points) for i in self.flip_indices_v
            ]
        if heatmap_fun == "gaussian":
            self.heatmap_generator = GaussianHeatmapGenerator(
                nb_landmarks=len(self.class_names),
                sigmas=sigma,
                gamma=gamma,
                full_map=full_map,
                heatmap_size=heatmap_size,
                learnable=False,
                device="cpu",
                background=background,
                all_points=all_points,
            )
        elif heatmap_fun == "laplacian":
            self.heatmap_generator = LaplacianHeatmapGenerator(
                nb_landmarks=len(self.class_names),
                sigmas=sigma,
                gamma=gamma,
                full_map=full_map,
                heatmap_size=heatmap_size,
                learnable=False,
                device="cpu",
                background=background,
                all_points=all_points,
            )
        else:
            raise ValueError("Heatmap function not supported")
        if batch_size > 1:
            self.batch_size = batch_size
        else:
            self.batch_size = 2
        if self.store_imgs:
            self.heatmaps = self._create_heatmaps_batch(self.landmarks)

    def __getitem__(self, idx: int) -> tuple[  # type: ignore[override]
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        flip_code = idx % self.ds_size_factor
        idx = idx // self.ds_size_factor
        if self.store_imgs:
            img = self.imgs[idx]
            heatmap = self.heatmaps[idx]
            landmark = self.landmarks[idx]
            padding = self.paddings[idx]
            dim_original = self.dim_original[idx]
        else:
            img = self._read_norm_img(self.img_paths[idx])  # type: ignore
            dim_original = torch.tensor(img.shape[-2:]).float()
            if self.dim_img is not None:
                if self.resize_pad:
                    dim_original = torch.tensor(img.shape[-2:]).float()
                    img, (hp, wp) = resize_with_pad(img, self.dim_img)
                    padding = torch.Tensor([hp, wp])
                    landmark = resize_landmarks(
                        self.landmarks_original[idx], dim_original, self.dim_img, padding
                    )
                else:
                    img = T.Resize(self.dim_img)(img)
                    landmark = resize_landmarks(
                        self.landmarks_original[idx], dim_original, self.dim_img
                    )
                    padding = torch.zeros(2)
            heatmap = self._create_heatmap(landmark).squeeze(0)
        if self.ds_size_factor > 1:
            if self.ds_size_factor == 2 and flip_code:
                if self.flip_aug_h:
                    img = Flip(0)(img)
                    heatmap = Flip(0)(heatmap)
                else:
                    img = Flip(1)(img)
                    heatmap = Flip(1)(heatmap)
                    flip_code += 1
            elif self.ds_size_factor == 4:
                if flip_code == 1 or flip_code == 3:
                    img = Flip(0)(img)
                    heatmap = Flip(0)(heatmap)
                if flip_code == 2 or flip_code == 3:
                    img = Flip(1)(img)
                    heatmap = Flip(1)(heatmap)
            landmark = flip_landmarks(
                landmark,
                flip_code,
                flip_indices_v=self.flip_indices_v,
                flip_indices_h=self.flip_indices_h,
            )
            heatmap = flip_heatmaps(
                heatmap,
                flip_code,
                flip_indices_v=self.flip_indices_v,
                flip_indices_h=self.flip_indices_h,
            )
        if self.transform is not None:
            (
                img,
                heatmap,
                landmark_t,
                affine_matrix,
            ) = self._transform_img_heatmap_landmark(img, heatmap, landmark, self.transform)
        else:
            landmark_t = landmark
            affine_matrix = torch.eye(4)
        if self.normalize_intensity:
            img = monai.transforms.NormalizeIntensity()(img)  # type: ignore[assignment]
        return (
            img,
            heatmap,
            landmark_t,
            affine_matrix,
            landmark,
            self.landmarks_original[idx],
            dim_original,
            self.pixel_spacings[idx],
            padding,
        )

    def _transform_img_heatmap_landmark(
        self, img: torch.Tensor, heatmap: torch.Tensor, landmark: torch.Tensor, transform: Callable
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transform the image, the heatmap and the landmark using the transformation
        ``self.transform``.

        Args:
            img (torch.Tensor): image to transform.
            heatmap (torch.Tensor): heatmap to transform.
            landmark (torch.Tensor): landmark to transform.

        Returns:
            img_t (torch.Tensor): transformed image.
            heatmap_t (torch.Tensor): transformed heatmap.
            landmark_t (torch.Tensor): transformed landmark.
            affine_matrix_push (torch.Tensor): push affine matrix of the transformation.
        """
        out = transform({"image": img, "seg": heatmap.squeeze(0)})
        img_t = out["image"]
        heatmap_t = out["seg"]
        affine_matrix_pull = img_t.meta.get("affine", None).float()
        affine_matrix_push = torch.linalg.inv(affine_matrix_pull)
        landmark_t = affine_landmarks(landmark, affine_matrix_push)
        return img_t, heatmap_t, landmark_t, affine_matrix_push

    def _create_heatmaps(self, landmarks: torch.Tensor) -> list[torch.Tensor]:
        """
        Create heatmaps from landmarks.

        Args:
            landmarks (torch.Tensor): landmarks of the images.

        Returns:
            heatmaps (list[torch.Tensor]): list of heatmaps.
        """
        print(f"Creating heatmaps for {len(landmarks)} images...")
        heatmaps = []
        for i in tqdm(range(landmarks.shape[0])):
            heatmaps.append(self._create_heatmap(landmarks[i]))
        return heatmaps

    def _create_heatmaps_batch(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Create heatmaps from landmarks using batches.

        Args:
            landmarks (torch.Tensor): landmarks of the images.

        Returns:
            heatmaps (torch.Tensor): heatmaps.
        """
        print(f"Creating heatmaps for {len(landmarks)} images... (batch size: {self.batch_size})")
        heatmaps = torch.zeros(
            (
                len(landmarks),
                len(self.class_names) + int(self.all_points or self.background),
                self.heatmap_generator.heatmap_size[0],
                self.heatmap_generator.heatmap_size[1],
            )
        )
        i = 0
        for i in tqdm(range(landmarks.shape[0] // self.batch_size)):
            heatmaps[i * self.batch_size : (i + 1) * self.batch_size] = self.heatmap_generator(
                landmarks[i * self.batch_size : (i + 1) * self.batch_size]
            )
        if landmarks.shape[0] % self.batch_size != 0:
            try:
                heatmaps[(i + 1) * self.batch_size :] = self.heatmap_generator(
                    landmarks[(i + 1) * self.batch_size :]
                )
            except RuntimeError:
                heatmaps = self.heatmap_generator(landmarks)
        return heatmaps

    def _create_heatmap(self, landmark: torch.Tensor) -> torch.Tensor:
        """
        Create a heatmap from a landmark.

        Args:
            landmark (torch.Tensor): landmark of the image.

        Returns:
            heatmap (torch.Tensor): heatmap.
        """
        return self.heatmap_generator(landmark.unsqueeze(0))
