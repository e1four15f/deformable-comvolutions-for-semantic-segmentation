from pathlib import Path
from typing import *

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.core.composition import TransformType
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class ADE20KDataset(Dataset):
    def __init__(self, images_path: Path, annotation_path: Path, transforms: Optional[TransformType] = None):
        self.images_files = sorted(images_path.iterdir())
        self.annotation_files = sorted(annotation_path.iterdir())
        self.transform = transforms
        self._check_dataset_correctness()

    def _check_dataset_correctness(self) -> None:
        assert len(self.images_files) == len(self.annotation_files)
        for img_file, ann_file in zip(self.images_files, self.annotation_files):
            assert (
                img_file.stem[-len("00000001") :] == ann_file.stem[-len("00000001") :]
            )

    def __len__(self) -> int:
        return len(self.images_files)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        # read images do BGR to RGB
        img = cv2.imread(str(self.images_files[index]))[:, :, ::-1]
        ann = cv2.imread(str(self.annotation_files[index]), cv2.IMREAD_GRAYSCALE)

        if self.transform is None:
            return img, ann

        # apply transformations and augmentations
        transformed = self.transform(image=img, mask=ann)
        res_img = transformed["image"]
        # shift class labels, so non-labelled class is -1 and other classes goes [0, 149]
        res_ann = transformed["mask"].type(torch.long) - 1
        return res_img, res_ann


def get_transforms(
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    fixed_img_size: Optional[int] = None,
    use_augmentations: bool = False,
) -> TransformType:
    transformations = []
    if use_augmentations:
        transformations += [
            A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            A.GaussNoise(),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(rotate_limit=30, p=1, border_mode=0, value=0, mask_value=0),  # 0 is non label class
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        ]
    if fixed_img_size is not None:
        transformations += [
            A.SmallestMaxSize(fixed_img_size),
            A.CenterCrop(fixed_img_size, fixed_img_size),
        ]
    transformations += [
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
    return A.Compose(transformations)
