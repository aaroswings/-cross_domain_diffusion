import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from sortedcollections import OrderedSet
import numpy as np
import os
from PIL import Image
from typing import Tuple

from train.util import get_image_file_names

class PairedDataset(Dataset):
    """
    Return concatenated batch of A/B pairs of spatially aligned images
    from two aligned directories, and a mask over domain A images.
    The mask is used in the concat-guided-sampling process.
    With the mask, it isn't necessary to keep track of which channels are from A or B images.
    """
    def __init__(
        self, 
        roots: Tuple[str, str], 
        size: int = 256, 
        channels: Tuple[int, int] = (3, 3),
        p_hflip: float = 0.5
    ):
        super().__init__()
        self.root_a, self.root_b = roots
        self.channels_a, self.channels_b = channels
        self.files: OrderedSet = get_image_file_names(self.root_a) | get_image_file_names(self.root_b)
        self.size = size
        self.p_hflip = p_hflip

    def __len__(self): 
        return len(self.files)

    def load_image(self, path) -> Image:
        img = Image.open(path).convert('RGB')
        img = TF.resize(img, self.size)
        return img

    @torch.no_grad()
    def __getitem__(self, i) -> torch.Tensor:
        name = self.files[i]
        img_a = self.load_image(os.path.join(self.root_a, name))
        img_b = self.load_image(os.path.join(self.root_b, name))

        # Augmentation
        if random.random() > self.p_hflip:
            img_a, img_b = TF.hflip(img_a), TF.hflip(img_b)

        a, b = TF.to_tensor(img_a), TF.to_tensor(img_b)
        a, b = a[:self.channels_a], b[:self.channels_b]
        # center on 0
        x = torch.cat([a, b], dim=0) * 2.0 - 1.0
        channel_mask = torch.cat([torch.zeros(a.size(0), 1, 1), torch.ones(b.size(0), 1, 1)], dim=0)

        return x, channel_mask