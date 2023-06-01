import random
from pathlib import Path
from typing import *

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def show_random_examples_from_path(
    examples_path: Path,
    annotations_path: Path,
    scene_categories: Dict[str, str],
    objects_info: Dict[int, str],
    k: int = 5,
) -> None:
    for _ in range(k):
        example_path = random.choice(list(examples_path.iterdir()))
        annotation_path = annotations_path / f"{example_path.stem}.png"
        show_example_from_path(example_path, annotation_path, scene_categories, objects_info)


def show_example_from_path(
    example_path: Path,
    annotation_path: Path,
    scene_categories: Dict[str, str],
    objects_info: Dict[int, str],
) -> None:
    print(example_path)
    print(annotation_path)

    example = cv2.imread(str(example_path))[:, :, ::-1]  # BGR2RGB
    annotation = cv2.imread(str(annotation_path))
    segment_classes = np.unique(annotation)
    segment_classes = segment_classes[segment_classes != 0]

    show_example(example, annotation, titles=["Image", f"Annotation: {scene_categories[example_path.stem]}"])

    segment_classes = {index: objects_info[index] for index in segment_classes}
    print(str(segment_classes))


def show_example(example: np.ndarray, annotation: np.ndarray, predictions: List[np.ndarray] = [], titles: List[str] = ["Image", "Annotation"]) -> None:
    if example.min() < 0:
        example -= example.min()
        example /= example.max()

    fig, ax = plt.subplots(nrows=1, ncols=2 + len(predictions), figsize=(20, 10))

    ax[0].imshow(example)
    if titles:
        ax[0].set_title(titles[0])
    ax[0].axis("off")

    colors = _get_color_palette()
    colored_annotation = np.apply_along_axis(lambda x: colors[x[0]], axis=2, arr=annotation)
    ax[1].imshow(colored_annotation)
    if titles:
        ax[1].set_title(titles[1])
    ax[1].axis("off")

    for index, prediction in enumerate(predictions, 2):
        colored_prediction = np.apply_along_axis(lambda x: colors[x[0]], axis=2, arr=prediction)
        ax[index].imshow(colored_prediction)
        if titles:
            ax[index].set_title(titles[index])
        ax[index].axis("off")

    plt.tight_layout()
    plt.show()


def _get_color_palette():
    colors = sns.color_palette("turbo", n_colors=150)
    random.Random(42).shuffle(colors)
    colors.insert(0, (1., 1., 1.))
    return colors
