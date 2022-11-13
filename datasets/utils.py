from torch.utils.data import Dataset
import torch
from torchvision import transforms
from abc import abstractmethod


class FullDatasetBase:
    mean: tuple
    std: tuple
    img_shape: tuple
    num_classes: int
    name: str

    def __init__(self, **kwargs):
        pass

    def gen_base_transforms(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ]), None

    def gen_test_transforms(self):
        base, _ = self.gen_base_transforms()
        return base, _

    @abstractmethod
    def gen_train_transforms(self):
        return transforms.Compose([]), None

    @abstractmethod
    def gen_train_datasets(self, transform=None, target_transform=None) -> Dataset:
        pass

    @abstractmethod
    def gen_val_datasets(self, transform=None, target_transform=None) -> Dataset:
        pass

    @abstractmethod
    def gen_test_datasets(self, transform=None, target_transform=None) -> Dataset:
        pass

    def sample_imgs(self) -> torch.Tensor:
        return torch.stack([torch.zeros(self.img_shape)] * 2)

    @staticmethod
    @abstractmethod
    def is_dataset_name(name: str):
        return name == "my_dataset"

