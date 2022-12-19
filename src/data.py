from typing import Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CIFAR_10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_10_STD = (0.2023, 0.1994, 0.2010)

class DeepMindCIFAR10(Dataset[Tuple[Image.Image, torch.Tensor]]):
    def __init__(self, root_dir: str, transform: Optional[Callable] = None) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        with np.load(root_dir) as data:
            self.images = data["image"]
            self.labels = torch.from_numpy(data["label"])
            self.length = data["label"].shape[0]

    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, index) -> Tuple[Image.Image, torch.Tensor]:
        image = Image.fromarray(self.images[index])
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[index]
        return image, label
                
