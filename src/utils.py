from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm


def compute_images_mean_and_std(images_path: Path) -> None:
    channels_sum = np.zeros(3)
    channels_sum_sq = np.zeros(3)
    total_pixels = 0

    for path in tqdm(list(images_path.iterdir())):
        img = cv2.imread(str(path))[:, :, ::-1]
        img = cv2.normalize(
            img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
        channels_sum += np.sum(img, axis=(0, 1))
        channels_sum_sq += np.sum(img**2, axis=(0, 1))
        total_pixels += img.shape[0] * img.shape[1]

    mean = channels_sum / total_pixels
    variance = (channels_sum_sq / total_pixels) - (mean**2)
    std = variance ** (1 / 2)

    print("mean: " + str(mean))
    print("std:  " + str(std))
