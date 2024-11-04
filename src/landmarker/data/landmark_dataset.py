"""Module containing the ``LandmarkDataset`` class and its subclasses ``MaskDataset``
and ``HeatmapDataset``. All three classe allow to create a dataset of images and landmarks.
The three classes are subclasses of ``torch.utils.data.Dataset``.
"""

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import Compose, LoadImage, Resize, Transpose  # type: ignore
from torch.utils.data import Dataset
from tqdm import tqdm  # type: ignore

from landmarker.heatmap.generator import (
    GaussianHeatmapGenerator,
    HeatmapGenerator,
    LaplacianHeatmapGenerator,
)
from landmarker.transforms.images import resize_with_pad
from landmarker.transforms.landmarks import affine_landmarks, resize_landmarks
from landmarker.utils import extract_roi


class LandmarkDataset(Dataset):
    """
    ``LandmarkDataset`` is a subclass of ``torch.utils.data.Dataset``. It represents a dataset of
    images and landmarks. The images can be provided as a list of paths to the images or as a list
    of numpy arrays or as a numpy array/torch.Tensor.

    Args:
        imgs (list[str] | list[np.array] | np.ndarray | torch.Tensor): list of paths to the images
            or list of numpy arrays or numpy array/torch.Tensor.
        landmarks (np.ndarray | torch.Tensor): landmarks of the images. If the landmarks are 2D: y, x
            or 3D: z, y, x.
        spatial_dims (int): number of spatial dimensions of the images. (defaults: 2)
        pixel_spacing (Optional[np.ndarray | torch.Tensor]): pixel spacing of the images.
            (defaults: None)
        class_names (Optional[list]): names of the landmarks. (defaults: None)
        transform (Optional[Callable]): transformation to apply to the images and landmarks.
            (defaults: None)
        store_imgs (bool): whether to store the images in memory or not. (defaults: True)
        dim_img (Optional[tuple[int, int]]): dimension of the images. (defaults: None)
        img_paths (Optional[list[str]]): list of paths to the images. (defaults: None)
        resize_pad (bool): whether to resize the images and landmarks or not. (defaults: True)
    """

    def __init__(
        self,
        imgs: list[str] | np.ndarray | torch.Tensor | list[np.ndarray] | list[torch.Tensor],
        landmarks: np.ndarray | torch.Tensor,
        spatial_dims: int = 2,
        pixel_spacing: Optional[list[int] | tuple[int, ...] | np.ndarray | torch.Tensor] = None,
        class_names: Optional[list] = None,
        transform: Optional[Callable] = None,
        store_imgs: bool = True,
        dim_img: Optional[tuple[int, ...]] = None,
        img_paths: Optional[list[str]] = None,
        resize_pad: bool = True,
    ) -> None:
        if spatial_dims not in [2, 3]:
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")
        self.spatial_dims = spatial_dims
        self.resize_pad = resize_pad
        self.transform = transform
        self.store_imgs = store_imgs
        self.dim_img = dim_img
        if isinstance(imgs, list) and isinstance(imgs[0], str):
            if imgs[0].endswith(".npy"):
                self.image_loader = Compose(
                    [
                        LoadImage(image_only=True, ensure_channel_first=True),
                    ]
                )
            elif self.spatial_dims == 2:
                self.image_loader = Compose(
                    [
                        LoadImage(image_only=True, ensure_channel_first=True),
                        Transpose(indices=[0, 2, 1]),
                    ]
                )
            else:
                self.image_loader = Compose(
                    [
                        LoadImage(image_only=True, ensure_channel_first=True),
                        Transpose(indices=[0, 3, 2, 1]),
                    ]
                )
        self._init_landmarks(landmarks, class_names)
        self._init_imgs(imgs, img_paths)
        self._init_pixel_spacing(pixel_spacing)

    def _init_landmarks(
        self,
        landmarks: np.ndarray | torch.Tensor,
        class_names: Optional[list] = None,
    ) -> None:
        if len(landmarks.shape) == 2:
            landmarks = landmarks.reshape(
                (landmarks.shape[0], 1, landmarks.shape[1])
            )  # (N, D) => (N, 1, D) only one class
        self.nb_landmarks = landmarks.shape[1]
        assert self.spatial_dims == landmarks.shape[-1]
        if isinstance(landmarks, np.ndarray):
            self.landmarks_original = torch.Tensor(landmarks.copy())
        elif isinstance(landmarks, torch.Tensor):
            self.landmarks_original = landmarks.clone().float()
        else:
            if landmarks is None:
                raise ValueError("landmarks must be provided")
            raise TypeError("landmarks type not supported")

        if class_names is None:
            self.class_names = [f"landmark_{i}" for i in range(self.landmarks_original.shape[1])]
        else:
            if len(class_names) != self.landmarks_original.shape[1]:
                raise ValueError("class_names must be of length landmarks.shape[1] or None")
            self.class_names = class_names

    def _init_pixel_spacing(
        self, pixel_spacing: Optional[list[int] | tuple[int, ...] | np.ndarray | torch.Tensor]
    ) -> None:
        if pixel_spacing is not None:
            if isinstance(pixel_spacing, (tuple, list)):
                assert (
                    len(pixel_spacing) == self.spatial_dims
                ), f"pixel_spacing must be of shape {self.spatial_dims}"
                self.pixel_spacings = (
                    torch.Tensor(pixel_spacing).unsqueeze(0).repeat(len(self.landmarks_original), 1)
                )
            elif pixel_spacing.shape == (len(self.landmarks_original), self.spatial_dims):
                self.pixel_spacings = torch.Tensor(pixel_spacing)
            elif pixel_spacing.shape == (self.spatial_dims,):
                self.pixel_spacings = (
                    torch.Tensor(pixel_spacing).unsqueeze(0).repeat(len(self.landmarks_original), 1)
                )
            elif pixel_spacing.shape == (len(self.landmarks_original),):
                self.pixel_spacings = (
                    torch.Tensor(pixel_spacing).unsqueeze(1).repeat(1, self.spatial_dims)
                )
            else:
                raise ValueError(
                    f"Pixel_spacing must be of shape (N, {self.spatial_dims}) or "
                    "({self.spatial_dims},) since spatial_dims is {self.spatial_dims}."
                )
        else:
            self.pixel_spacings = torch.ones(len(self.landmarks_original), self.spatial_dims)

    def _init_imgs(
        self,
        imgs: list[str] | np.ndarray | torch.Tensor | list[np.ndarray] | list[torch.Tensor],
        img_paths: Optional[list[str]],
    ) -> None:
        if isinstance(imgs, list):
            if isinstance(imgs[0], str):
                self.img_paths = imgs
            elif isinstance(imgs[0], np.ndarray):
                self.imgs = [torch.Tensor(imgs[i]) for i in range(len(imgs))]
            elif isinstance(imgs[0], torch.Tensor):
                self.imgs = imgs  # type: ignore
            else:
                raise TypeError(f"imgs type not supported, got {type(imgs[0])}")
        else:
            self.imgs = [torch.Tensor(imgs[i]) for i in range(len(imgs))]
            self.img_paths = img_paths if img_paths is not None else []
        if not isinstance(imgs[0], str):
            if self.spatial_dims == 2:
                assert len(self.imgs[0].shape) == 3
            else:
                assert len(self.imgs[0].shape) == 4

        # images need to be stored in memory and images are provided as paths
        if self.store_imgs and not hasattr(self, "imgs"):
            assert self.img_paths is not None
            self.imgs, self.dim_original = self._read_all_imgs(self.img_paths)  # type: ignore
        if self.dim_img is not None and self.store_imgs:
            (
                self.imgs,
                self.landmarks,
                self.dim_original,
                self.paddings,
            ) = self._resize_all_imgs_landmarks(self.imgs, self.landmarks_original, self.dim_img)
        else:
            self.landmarks = self.landmarks_original.clone()
            self.paddings = torch.zeros((self.landmarks.shape[0], self.spatial_dims))
            if not hasattr(self, "dim_original") and self.store_imgs:
                self.dim_original = torch.zeros((self.landmarks.shape[0], self.spatial_dims))
                for i in range(self.landmarks.shape[0]):
                    self.dim_original[i] = torch.tensor(
                        self.imgs[i].shape[-self.spatial_dims :]
                    ).float()

    def __len__(self) -> int:
        return len(self.landmarks_original)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if not self.store_imgs:
            img = self.image_loader(self.img_paths[idx])  # type: ignore
            dim_original = torch.tensor(img.shape[-self.spatial_dims :]).float()
            if self.dim_img is not None:
                if self.resize_pad:
                    dim_original = torch.tensor(img.shape[-self.spatial_dims :]).float()
                    img, pad = resize_with_pad(img, self.dim_img)
                    padding = torch.Tensor(pad)
                    landmark = resize_landmarks(
                        self.landmarks_original[idx], dim_original, self.dim_img, padding
                    )
                else:
                    img = Resize(spatial_size=self.dim_img)(img)
                    landmark = resize_landmarks(
                        self.landmarks_original[idx], dim_original, self.dim_img
                    )
                    padding = torch.zeros(self.spatial_dims)
            else:
                landmark = self.landmarks_original[idx]
                padding = torch.zeros(self.spatial_dims)
        else:
            img = self.imgs[idx]
            landmark = self.landmarks[idx]
            padding = self.paddings[idx]
            dim_original = self.dim_original[idx]
        if self.transform is not None:
            img, landmark_t, affine_matrix = self._transform_img_landmark(
                img, landmark, self.transform  # type: ignore
            )
        else:
            landmark_t = landmark
            affine_matrix = torch.eye(4)

        return {
            "image": img,
            "landmark": landmark_t,
            "affine": affine_matrix,
            "dim_original": dim_original,
            "spacing": self.pixel_spacings[idx],
            "padding": padding,
        }

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

    def _read_all_imgs(self, paths: list[str]) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Read all images in ``paths``.

        Args:
            paths (list[str]): list of paths to the images.

        Returns:
            imgs (list[torch.Tensor]): list of images.
            dim_original (torch.Tensor): original dimensions of the images.
        """
        print(f"Reading {len(paths)} images...")
        imgs = []
        dim_original = torch.zeros((len(paths), self.spatial_dims))
        for i, path in enumerate(tqdm(paths)):
            img = self.image_loader(path)
            dim_original[i] = torch.tensor(img.shape[-self.spatial_dims :]).float()
            imgs.append(img)
        return imgs, dim_original

    def _resize_all_imgs_landmarks(
        self, imgs: list[torch.Tensor], landmarks: torch.Tensor, dim_img: tuple[int, ...]
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
        dim_original = torch.zeros((len(imgs), self.spatial_dims))
        paddings = torch.zeros((len(imgs), self.spatial_dims))
        for i in tqdm(range(len(imgs))):
            dim_original[i] = torch.tensor(imgs[i].shape[-self.spatial_dims :]).float()
            if self.resize_pad:
                imgs[i], padding = resize_with_pad(imgs[i], dim_img)
                paddings[i] = torch.Tensor(padding)
            else:
                imgs[i] = Resize(dim_img)(imgs[i])
                paddings[i] = torch.zeros(self.spatial_dims)
            landmarks[i] = resize_landmarks(landmarks[i], dim_original[i], dim_img, paddings[i])
        return imgs, landmarks, dim_original, paddings


class PatchDataset(LandmarkDataset):
    """
    ``PatchDataset`` is a subclass of ``LandmarkDataset``. It represents a dataset of images and
    landmarks. The images can be provided as a list of paths to the images or as a list of numpy
    arrays or as a numpy array/torch.Tensor. The landmarks can be provided as a numpy array or
    torch.Tensor. The landmarks can be extracted from the images and stored as a torch.Tensor or
    they can be genererated from provided landmarks. The images are cropped around the landmarks
    and stored as torch.Tensor. The landmarks are the landmarks of the cropped images. The offset
    is randomly generated in a range of [-range_aug_patch, range_aug_patch] and added to the
    landmarks to create the cropped images. If the cropped images are not of the same size, they
    are resized to the same size.

    Args:
        imgs (list[str] | list[np.array] | np.ndarray | torch.Tensor): list of paths to the images
            or list of numpy arrays or numpy array/torch.Tensor.
        landmarks (np.ndarray | torch.Tensor): landmarks of the images. If the landmarks are 2D: y, x
            or 3D: z, y, x.
        index_landmark (int): index of the landmark to use to extract the patch. (defaults: 0)
        spatial_dims (int): number of spatial dimensions of the images. (defaults: 2)
        pixel_spacing (Optional[np.ndarray | torch.Tensor]): pixel spacing of the images.
            (defaults: None)
        class_names (Optional[list]): names of the landmarks. (defaults: None)
        transform (Optional[Callable]): transformation to apply to the images and landmarks.
            (defaults: None)
        store_imgs (bool): whether to store the images in memory or not. (defaults: True)
        dim_patch (tuple[int, ...]): dimension of the patches. (defaults: (256, 256))
        range_aug_patch (int): range of the random offset to apply to the landmarks to extract the
            patches. (defaults: 64)

    """

    def __init__(
        self,
        imgs: list[str] | list[np.ndarray] | np.ndarray | torch.Tensor,
        landmarks: np.ndarray | torch.Tensor,
        index_landmark=0,
        spatial_dims: int = 2,
        pixel_spacing: Optional[torch.Tensor] = None,
        class_names: Optional[list] = None,
        transform: Optional[Callable] = None,
        store_imgs: bool = False,
        dim_patch: tuple[int, ...] = (256, 256),
        range_aug_patch: int = 64,
    ) -> None:
        self.range_aug_patch = range_aug_patch
        self.dim_patch = dim_patch
        self.index_landmark = index_landmark
        super().__init__(
            imgs=imgs,
            landmarks=landmarks,
            spatial_dims=spatial_dims,
            pixel_spacing=pixel_spacing,
            class_names=class_names,
            transform=transform,
            store_imgs=store_imgs,
            dim_img=None,
            resize_pad=False,
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        results_landmark_ds = super().__getitem__(idx)
        img = results_landmark_ds["image"]
        landmark = results_landmark_ds["landmark"][self.index_landmark]
        if self.range_aug_patch is not None:
            offset = torch.randint(-self.range_aug_patch, self.range_aug_patch, landmark.shape)
            roi_middle = landmark + offset
        else:
            offset = torch.zeros(landmark.shape)
            roi_middle = landmark
        patch, landmark_patch, roi_corner_point = extract_roi(
            img, roi_middle, landmark, self.dim_patch, self.spatial_dims, ensure_dim=True
        )
        return {
            "image": patch,
            "landmark": landmark_patch,
            "offset": offset,
            "landmark_original": landmark,
            "roi_corner_point": roi_corner_point,
            "affine": results_landmark_ds["affine"],
            "dim_original": torch.tensor(self.dim_patch),
            "dim_original_full_image": results_landmark_ds["dim_original"],
            "spacing": results_landmark_ds["spacing"],
            "padding": results_landmark_ds["padding"],
        }


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
        landmarks (np.ndarray | torch.Tensor): landmarks of the images. If the landmarks are 2D: y, x
            or 3D: z, y, x.
        spatial_dims (int): number of spatial dimensions of the images. (defaults: 2)
        pixel_spacing (Optional[np.ndarray | torch.Tensor]): pixel spacing of the images.
            (defaults: None)
        class_names (Optional[list]): names of the landmarks. (defaults: None)
        transform (Optional[Callable]): transformation to apply to the images and landmarks.
            (defaults: None)
        store_imgs (bool): whether to store the images in memory or not. (defaults: True)
        store_masks (bool): whether to store the masks in memory or not.
            (defaults: False)
        dim_img (Optional[tuple[int, int]]): dimension of the images. (defaults: None)
        mask_paths (Optional[list[str]]): list of paths to the masks. (defaults: None)
        nb_landmarks (int): number of landmarks in the masks. (defaults: 1)
        img_paths (Optional[list[str]]): list of paths to the images. (defaults: None)
        resize_pad (bool): whether to resize the images and landmarks or not. (defaults: True)
    """

    def __init__(
        self,
        imgs: list[str] | list[np.ndarray] | np.ndarray | torch.Tensor,
        landmarks: Optional[np.ndarray | torch.Tensor] = None,
        spatial_dims: int = 2,
        pixel_spacing: Optional[torch.Tensor] = None,
        class_names: Optional[list] = None,
        transform: Optional[Callable] = None,
        store_imgs: bool = True,
        dim_img: Optional[tuple[int, ...]] = None,
        mask_paths: Optional[list[str]] = None,
        nb_landmarks: int = 1,
        img_paths: Optional[list[str]] = None,
        resize_pad: bool = True,
    ) -> None:
        self.nb_landmarks = nb_landmarks
        self.mask_paths = mask_paths
        self.dim_img = dim_img
        self.resize_pad = resize_pad
        self.spatial_dims = spatial_dims
        if mask_paths is not None:
            if mask_paths[0].endswith(".npy"):
                self.mask_loader = Compose(
                    [
                        LoadImage(image_only=True, ensure_channel_first=True),
                    ]
                )
            elif self.spatial_dims == 2:
                self.mask_loader = Compose(
                    [
                        LoadImage(image_only=True, ensure_channel_first=True),
                        Transpose(indices=[0, 2, 1]),
                    ]
                )
            else:
                self.mask_loader = Compose(
                    [
                        LoadImage(image_only=True, ensure_channel_first=True),
                        Transpose(indices=[0, 3, 2, 1]),
                    ]
                )
        if landmarks is not None:
            super().__init__(
                imgs=imgs,
                landmarks=landmarks,
                spatial_dims=spatial_dims,
                pixel_spacing=pixel_spacing,
                class_names=class_names,
                transform=transform,
                store_imgs=store_imgs,
                dim_img=dim_img,
                img_paths=img_paths,
                resize_pad=resize_pad,
            )
        elif self.mask_paths is not None:
            landmarks = self._read_extract_landmarks(self.mask_paths)
            super().__init__(
                imgs=imgs,
                landmarks=landmarks,
                spatial_dims=spatial_dims,
                pixel_spacing=pixel_spacing,
                class_names=class_names,
                transform=transform,
                store_imgs=store_imgs,
                dim_img=dim_img,
                img_paths=img_paths,
                resize_pad=resize_pad,
            )
        else:
            raise ValueError("Either mask_paths or landmarks must be provided")

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if not self.store_imgs:
            img = self.image_loader(self.img_paths[idx])  # type: ignore
            dim_original = torch.tensor(img.shape[-self.spatial_dims :]).float()
            if self.dim_img is not None:
                if self.resize_pad:
                    dim_original = torch.tensor(img.shape[-self.spatial_dims :]).float()
                    img, pad = resize_with_pad(img, self.dim_img)
                    padding = torch.Tensor(pad)
                    landmark = resize_landmarks(
                        self.landmarks_original[idx], dim_original, self.dim_img, padding
                    )
                else:
                    img = Resize(self.dim_img)(img)
                    landmark = resize_landmarks(
                        self.landmarks_original[idx], dim_original, self.dim_img
                    )
                    padding = torch.zeros(self.spatial_dims)
                dim_img = self.dim_img
            else:
                landmark = self.landmarks_original[idx]
                padding = torch.zeros(self.spatial_dims)
                if self.spatial_dims == 2:
                    dim_img = (int(dim_original[0].item()), int(dim_original[1].item()))
                else:
                    dim_img = (
                        int(dim_original[0].item()),
                        int(dim_original[1].item()),
                        int(dim_original[2].item()),
                    )
        else:
            img = self.imgs[idx]
            landmark = self.landmarks[idx]
            padding = self.paddings[idx]
            dim_original = self.dim_original[idx]
            if self.dim_img is not None:
                dim_img = self.dim_img
            else:
                if self.spatial_dims == 2:
                    dim_img = (int(dim_original[0].item()), int(dim_original[1].item()))
                else:
                    dim_img = (
                        int(dim_original[0].item()),
                        int(dim_original[1].item()),
                        int(dim_original[2].item()),
                    )
        if self.transform is not None:
            img, landmark_t, affine_matrix = self._transform_img_landmark(
                img, landmark, self.transform  # type: ignore
            )
        else:
            landmark_t = landmark
            affine_matrix = torch.eye(4)
        mask = self._create_mask(landmark_t, dim_img)
        return {
            "image": img,
            "mask": mask,
            "landmark": landmark_t,
            "affine": affine_matrix,
            "dim_original": dim_original,
            "spacing": self.pixel_spacings[idx],
            "padding": padding,
        }

    def _create_mask(self, landmark: torch.Tensor, dim: tuple[int, ...]) -> torch.Tensor:
        """
        Create a mask from a landmark.

        Args:
            landmark (torch.Tensor): landmark of the image.
            dim (tuple[int, ...]): dimension of the image.

        Returns:
            mask (torch.Tensor): mask.
        """
        mask = torch.zeros((landmark.shape[0], *dim))
        for i in range(landmark.shape[0]):
            if len(landmark.shape) == 3:
                for j in range(landmark.shape[1]):
                    if self.spatial_dims == 2:
                        if 0 <= landmark[i, j, 0] < dim[0] and 0 <= landmark[i, j, 1] < dim[1]:
                            mask[i, int(landmark[i, j, 0]), int(landmark[i, j, 1])] = 1
                    else:
                        if (
                            0 <= landmark[i, j, 0] < dim[0]
                            and 0 <= landmark[i, j, 1] < dim[1]
                            and 0 <= landmark[i, j, 2] < dim[2]
                        ):
                            mask[
                                i,
                                int(landmark[i, j, 0]),
                                int(landmark[i, j, 1]),
                                int(landmark[i, j, 2]),
                            ] = 1
            else:
                if self.spatial_dims == 2:
                    if 0 <= landmark[i, 0] < dim[0] and 0 <= landmark[i, 1] < dim[1]:
                        mask[i, int(landmark[i, 0]), int(landmark[i, 1])] = 1
                else:
                    if (
                        0 <= landmark[i, 0] < dim[0]
                        and 0 <= landmark[i, 1] < dim[1]
                        and 0 <= landmark[i, 2] < dim[2]
                    ):
                        mask[i, int(landmark[i, 0]), int(landmark[i, 1]), int(landmark[i, 2])] = 1
        return mask

    def _read_extract_landmarks(self, paths: list[str]) -> torch.Tensor:
        """
        Read masks and extract landmarks from these masks.

        Args:
            paths (list[str]): list of paths to the masks.

        Returns:
            landmarks (torch.Tensor): landmarks.
        """
        print(f"Reading and extracting masks from {len(paths)} images...")
        landmarks = []
        for path in tqdm(paths):
            landmark = self._read_extract_landmark(path)
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
        return torch.stack(landmarks)

    def _read_extract_landmark(self, path: str) -> torch.Tensor:
        """
        Read a mask and extract the landmark from this mask.

        Args:
            path (str): path to the mask.

        Returns:
            landmarks (torch.Tensor): landmark.
        """
        mask = self.mask_loader(path)
        landmarks = self._extract_landmark_from_mask(mask)
        return landmarks

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
            landmarks_list = [
                landmark.view(self.nb_landmarks, -1, self.spatial_dims) for landmark in landmarks
            ]
            landmarks_list = [
                F.pad(landmark, (0, 0, 0, max_len - landmark.shape[1]), value=torch.nan)
                for landmark in landmarks_list
            ]
            landmarks = torch.cat(landmarks_list)
        return landmarks


class PatchMaskDataset(PatchDataset):
    """
    ``PatchMaskDataset`` is a subclass of ``PatchDataset``. It represents a dataset of images and
    landmarks. The images can be provided as a list of paths to the images or as a list of numpy
    arrays or as a numpy array/torch.Tensor. The landmarks can be provided as a list of paths to
    the masks or as a list of numpy arrays or as a numpy array/torch.Tensor. The masks must be
    grayscale images with the landmarks represented by different values. The landmarks can be
    extracted from the masks and stored as a torch.Tensor or they can be genererated from provided
    landmarks. The images are cropped around the landmarks and stored as torch.Tensor. The landmarks
    are the landmarks of the cropped images. The offset is randomly generated in a range of
    [-range_aug_patch, range_aug_patch] and added to the landmarks to create the cropped images.
    If the cropped images are not of the same size, they are resized to the same size.
    """

    def __init__(
        self,
        imgs: list[str] | list[np.ndarray] | np.ndarray | torch.Tensor,
        landmarks: np.ndarray | torch.Tensor,
        index_landmark=0,
        spatial_dims: int = 2,
        pixel_spacing: Optional[torch.Tensor] = None,
        class_names: Optional[list] = None,
        transform: Optional[Callable] = None,
        store_imgs: bool = False,
        dim_patch: tuple[int, ...] = (256, 256),
        range_aug_patch: int = 64,
    ) -> None:
        super().__init__(
            imgs=imgs,
            landmarks=landmarks,
            index_landmark=index_landmark,
            spatial_dims=spatial_dims,
            pixel_spacing=pixel_spacing,
            class_names=class_names,
            transform=transform,
            store_imgs=store_imgs,
            dim_patch=dim_patch,
            range_aug_patch=range_aug_patch,
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        batch = super().__getitem__(idx)
        mask = self._create_mask(
            batch["landmark"].unsqueeze(0), batch["image"].shape[-self.spatial_dims :]
        )
        return {**batch, "mask": mask}

    def _create_mask(self, landmark: torch.Tensor, dim: tuple[int, ...]) -> torch.Tensor:
        """
        Create a mask from a landmark.

        Args:
            landmark (torch.Tensor): landmark of the image.
            dim (tuple[int, ...]): dimension of the image.

        Returns:
            mask (torch.Tensor): mask.
        """
        mask = torch.zeros((landmark.shape[0], *dim))
        for i in range(landmark.shape[0]):
            if len(landmark.shape) == 3:
                for j in range(landmark.shape[1]):
                    if self.spatial_dims == 2:
                        if 0 <= landmark[i, j, 0] < dim[0] and 0 <= landmark[i, j, 1] < dim[1]:
                            mask[i, int(landmark[i, j, 0]), int(landmark[i, j, 1])] = 1
                    else:
                        if (
                            landmark[i, j, 0] < dim[0]
                            and landmark[i, j, 1] < dim[1]
                            and landmark[i, j, 2] < dim[2]
                        ):
                            mask[
                                i,
                                int(landmark[i, j, 0]),
                                int(landmark[i, j, 1]),
                                int(landmark[i, j, 2]),
                            ] = 1
            else:
                if self.spatial_dims == 2:
                    if 0 <= landmark[i, 0] < dim[0] and 0 <= landmark[i, 1] < dim[1]:
                        mask[i, int(landmark[i, 0]), int(landmark[i, 1])] = 1
                else:
                    if (
                        landmark[i, 0] < dim[0]
                        and landmark[i, 1] < dim[1]
                        and landmark[i, 2] < dim[2]
                    ):
                        mask[i, int(landmark[i, 0]), int(landmark[i, 1]), int(landmark[i, 2])] = 1
        return mask


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
        landmarks (np.ndarray | torch.Tensor): landmarks of the images. If the landmarks are 2D: y, x
            or 3D: z, y, x.
        spatial_dims (int): number of spatial dimensions of the images. (defaults: 2)
        pixel_spacing (Optional[np.ndarray | torch.Tensor]): pixel spacing of the images.
            (defaults: None)
        class_names (Optional[list]): names of the landmarks. (defaults: None)
        transform (Optional[Callable]): transformation to apply to the images and landmarks.
            (defaults: None)
        sigma (float | list[float]): sigma of the Gaussian or Laplacian function. (defaults: 5)
        dim_img (Optional[tuple[int, int]]): dimension of the images. (defaults: None)
        batch_size (int): batch size to use when creating the heatmaps. (defaults: 32)
        background (bool): whether to add a background channel or not. (defaults: False)
        all_points (bool): whether to add a channel for all points or not. (defaults: False)
        gamma (Optional[float]): gamma of the Gaussian or Laplacian function. (defaults: None)
        heatmap_fun (str): function to use to create the heatmaps. (defaults: "gaussian")
        heatmap_size (Optional[tuple[int, int]]): size of the heatmaps. (defaults: None)
        img_paths (Optional[list[str]]): list of paths to the images. (defaults: None)
        resize_pad (bool): whether to resize the images and landmarks or not. (defaults: True)
    """

    def __init__(
        self,
        imgs: list[str] | list[np.ndarray] | np.ndarray | torch.Tensor,
        landmarks: np.ndarray | torch.Tensor,
        spatial_dims: int = 2,
        pixel_spacing: Optional[np.ndarray | torch.Tensor] = None,
        class_names: Optional[list] = None,
        transform: Optional[Callable] = None,
        store_imgs: bool = True,
        sigma: float | list[float] | np.ndarray | torch.Tensor = 5,
        dim_img: Optional[tuple[int, ...]] = None,
        batch_size: int = 32,
        background: bool = False,
        all_points: bool = False,
        gamma: Optional[float] = None,
        heatmap_fun: str = "gaussian",
        heatmap_size: Optional[tuple[int, ...]] = None,
        img_paths: Optional[list[str]] = None,
        resize_pad: bool = True,
    ) -> None:
        super().__init__(
            imgs=imgs,
            landmarks=landmarks,
            spatial_dims=spatial_dims,
            pixel_spacing=pixel_spacing,
            class_names=class_names,
            transform=transform,
            store_imgs=store_imgs,
            dim_img=dim_img,
            img_paths=img_paths,
            resize_pad=resize_pad,
        )
        if heatmap_size is None:
            if self.dim_img is not None:
                heatmap_size = self.dim_img
            else:
                assert (
                    self.store_imgs
                ), "Heatmap size must be provided if imgs are not stored, and dim_img is None"
                # heatmap size becomes the size of the first original image
                if self.spatial_dims == 2:
                    heatmap_size = (
                        int(self.dim_original[0, 0].item()),
                        int(self.dim_original[0, 1].item()),
                    )
                else:
                    heatmap_size = (
                        int(self.dim_original[0, 0].item()),
                        int(self.dim_original[0, 1].item()),
                        int(self.dim_original[0, 2].item()),
                    )
        self.background = background
        self.all_points = all_points
        self.heatmap_generator: HeatmapGenerator
        if heatmap_fun == "gaussian":
            self.heatmap_generator = GaussianHeatmapGenerator(
                nb_landmarks=len(self.class_names),
                sigmas=sigma,
                gamma=gamma,
                heatmap_size=heatmap_size,
                learnable=False,
                background=background,
                all_points=all_points,
            )
        elif heatmap_fun == "laplacian":
            self.heatmap_generator = LaplacianHeatmapGenerator(
                nb_landmarks=len(self.class_names),
                sigmas=sigma,
                gamma=gamma,
                heatmap_size=heatmap_size,
                learnable=False,
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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.store_imgs:
            img = self.imgs[idx]
            heatmap = self.heatmaps[idx]
            landmark = self.landmarks[idx]
            padding = self.paddings[idx]
            dim_original = self.dim_original[idx]
        else:
            img = self.image_loader(self.img_paths[idx])  # type: ignore
            dim_original = torch.tensor(img.shape[-self.spatial_dims :]).float()
            if self.dim_img is not None:
                if self.resize_pad:
                    dim_original = torch.tensor(img.shape[-self.spatial_dims :]).float()
                    img, pad = resize_with_pad(img, self.dim_img)
                    padding = torch.Tensor(pad)
                    landmark = resize_landmarks(
                        self.landmarks_original[idx], dim_original, self.dim_img, padding
                    )
                else:
                    img = Resize(self.dim_img)(img)
                    landmark = resize_landmarks(
                        self.landmarks_original[idx], dim_original, self.dim_img
                    )
                    padding = torch.zeros(2)
            heatmap = self._create_heatmap(landmark).squeeze(0)
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
        return {
            "image": img,
            "mask": heatmap,
            "landmark": landmark_t,
            "affine": affine_matrix,
            "dim_original": dim_original,
            "spacing": self.pixel_spacings[idx],
            "padding": padding,
        }

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
        out = transform({"image": img, "mask": heatmap.squeeze(0)})
        img_t = out["image"]
        heatmap_t = out["mask"]
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
