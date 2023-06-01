from typing import *
from pathlib import Path


def read_scene_categories(path: Path) -> Dict[str, str]:
    return {
        i.split()[0]: i.split()[1]
        for i in path.read_text().strip().split("\n")
    }


def read_objects_info(path: Path) -> Dict[int, str]:
    return {
        int(i.split("\t")[0]): i.split("\t")[-1]
        for i in path.read_text().strip().split("\n")[1:]
    }
