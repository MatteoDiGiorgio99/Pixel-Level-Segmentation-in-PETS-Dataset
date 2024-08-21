
import torch
from torchvision import transforms as T
import torchvision

class PetsDataset(torchvision.datasets.OxfordIIITPet):
    def __init__(
        self,
        root: str, # The root directory where the dataset is stored.
        split: str, # The split of the dataset to load. One of "train", "test", "trainval", or "val".
        target_types="segmentation", # The type of target to return. One of "segmentation" or "class".
        download=False, # Whether to download the dataset if it is not found in the root directory.
        pre_transform=None, # The transformation to apply to the input image before any other transformations.
        post_transform=None, # The transformation to apply to the input image after all other transformations.
        pre_target_transform=None, # The transformation to apply to the target before any other transformations.
        post_target_transform=None, # The transformation to apply to the target after all other transformations.
        common_transform=None, # The transformation to apply to both the input and the target before any other transformations.
    ):
        super().__init__(
            root=root,
            split=split,
            target_types=target_types,
            download=download,
            transform=pre_transform,
            target_transform=pre_target_transform,
        )
        self.post_transform = post_transform
        self.post_target_transform = post_target_transform
        self.common_transform = common_transform

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        (input, target) = super().__getitem__(idx)
        
        # Standard transformations are applied to both the inputs and the labels by forming a 4-channel image and applying the transformations to the entire image.
        # Afterward, the segmentation mask (the 4th channel) is extracted separately.

        if self.common_transform is not None:
            both = torch.cat([input, target], dim=0)
            both = self.common_transform(both)
            (input, target) = torch.split(both, 3, dim=0)
        # end if
        
        if self.post_transform is not None:
            input = self.post_transform(input)
        if self.post_target_transform is not None:
            target = self.post_target_transform(target)

        return (input, target)