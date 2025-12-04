"""
This file is used to load the FER2013 dataset.
It consists of 48x48 pixel grayscale images of faces
with 7 emotions - angry, disgust, fear, happy, sad, surprise, and neutral.
"""

import pathlib
from typing import Any, Callable, Optional, Tuple
import torch
import torchvision
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import cv2
import os
import numpy as np
from utils import (
    to_numpy,
    to_torch,
    add_img_text,
    get_transforms,
)
import json

EMOTIONS_MAP = {
    0: "Enojo",
    1: "Disgusto",
    2: "Miedo",
    3: "Alegria",
    4: "Tristeza",
    5: "Sorpresa",
    6: "Neutral",
}
file_path = pathlib.Path(__file__).parent.absolute()


def get_loader(split, batch_size, shuffle=True, num_workers=0):
    dataset = FER2013(root=file_path, split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dataset, dataloader


class FER2013(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.img_size = 48
        self.target_transform = target_transform
        self.split = split
        self.root = root
        self.unnormalize = None
        self.transform, self.unnormalize = get_transforms(
            split=self.split, img_size=self.img_size
        )

        df = self._read_data()
        _str_to_array = [
            np.fromstring(val, dtype=np.uint8, sep=" ") for val in df["pixels"].values
        ]

        self._samples = np.array(_str_to_array)
        if split == "test":
            self._labels = np.empty(shape=len(self._samples))
        else:
            self._labels = df["emotion"].values

    def _read_data(self):
        base_folder = pathlib.Path(self.root) / "data"

        _split = "train" if self.split in ["train", "val"] else "test"
        file_name = f"{_split}.csv"
        data_file = base_folder / file_name

        if not os.path.isfile(data_file.as_posix()):
            raise RuntimeError(
                f"{file_name} not found in {base_folder} or corrupted."
            )

        df = pd.read_csv(data_file)

        if self.split != "test":
            train_val_split = json.load(open(base_folder / "split.json", "r"))
            split_samples = train_val_split[self.split]
            df = df.iloc[split_samples]

        return df

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):

        _vector_img = self._samples[idx]

        sample_image_np = _vector_img.reshape(self.img_size, self.img_size)
        sample_image_pil = to_pil_image(sample_image_np)

        if self.transform is not None:
            image = self.transform(sample_image_pil)
        else:
            image = torch.from_numpy(sample_image_np)

        target = int(self._labels[idx])
        emotion = EMOTIONS_MAP[target]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {
            "transformed": image,
            "label": target,
            "original": torch.from_numpy(sample_image_np).unsqueeze(0).float() / 255.0,
            "emotion": emotion,
        }