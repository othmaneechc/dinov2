# dinov2/data/datasets/custom_satellite.py
import os
from torchvision.datasets import ImageFolder
from .extended import ExtendedVisionDataset

class CustomSatellite(ExtendedVisionDataset):
    def __init__(self, root: str, split: str = "train", **kwargs):
        # harcoded
        split_map = {
            "TRAIN": "train",
            "VAL": "val",
            "TEST": "test"
        }
        actual_split = split_map.get(split, split.lower())
        root = os.path.join(root, actual_split)
        
        super().__init__(root, **kwargs)
        # We use ImageFolder just to parse the directory structure
        self.dataset = ImageFolder(root)

    def get_image_data(self, index: int):
        # just read bytes
        image_path, _ = self.dataset.samples[index]
        with open(image_path, mode="rb") as f:
            return f.read()

    def get_target(self, index: int):
        return self.dataset[index][1]

    def __len__(self):
        return len(self.dataset)