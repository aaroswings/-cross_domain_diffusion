import torch
from pathlib import Path
import os
import numpy as np
from PIL import Image
from typing import List
from sortedcollections import OrderedSet
import torchvision.transforms.functional as TF

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

def resize_reformat_files(root_in: str, root_out: str, size: int) -> None:
    # Files to convert to resized JPEGs
    image_names = get_image_file_names(root_in)

    os.makedirs(root_out, exist_ok=True)
    for name in image_names:
        path_in = os.path.join(root_in, name)
        img = Image.open(path_in).convert('RGB')
        img = TF.resize(img, size)
        name_out = name.replace('.png', '.jpeg')
        path_out = os.path.join(root_out, name_out)
        img.save(path_out)

def is_image_file(filename: str) -> bool:
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_image_file_names(dir) -> OrderedSet[str]:
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                # path = os.path.join(root, fname)
                images.append(fname)
    return OrderedSet(images)

@torch.no_grad()
def tensor_to_image(y: torch.Tensor) -> torch.Tensor:
    y = torch.clamp(y, -1, 1).cpu()
    y = (y / 2.0 + 0.5) * 255.
    arr = y.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    if arr.shape[2] == 3:
        im = Image.fromarray(arr)
    else:
        im = Image.fromarray(arr.squeeze(2), 'L')
    return im

@torch.no_grad()
def cat_images(sequence_of_images: List[object]) -> Image:
    # https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
    widths, heights = zip(*(i.size for i in sequence_of_images))
    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in sequence_of_images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    return new_im

@torch.no_grad()
def chunk_and_cat_pair(x: torch.Tensor, channels=3) -> Image:
    assert x.size(0) > channels
    assert x.size(0) - channels == 1 or x.size(0) - channels == 3
    return cat_images([tensor_to_image(x[:channels]), tensor_to_image(x[channels:])])

@torch.no_grad()
def save_images(
    images: List[torch.Tensor], 
    dir: Path,
    channel_split: int) -> None:

    os.makedirs(dir, exist_ok=True)
    for step_i, sample_batch in enumerate(images):
        for sample_i, sample in enumerate(sample_batch):
            sample = sample.cpu()
            if channel_split is not None:
                out_image = chunk_and_cat_pair(sample, channel_split)
            out_image.save(Path(dir) / f"batch_idx_{sample_i}_step_{step_i}.jpeg")

