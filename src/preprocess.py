import random
from typing import Tuple, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from omegaconf import DictConfig

CACHE_DIR = ".cache/"
CIFAR_MEAN_STD = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}


class DualAugVisionDataset(Dataset):
    """Returns (clean_img, aug_img, visibility, label)."""

    def __init__(self, base_dataset: datasets.VisionDataset, normalize: bool = True):
        self.base = base_dataset
        self.normalize = normalize
        mean, std = CIFAR_MEAN_STD[base_dataset.__class__.__name__.lower()]
        self.transform_clean = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std) if normalize else transforms.Lambda(lambda x: x),
            ]
        )
        self.rand_aug = transforms.RandAugment(num_ops=2, magnitude=9)
        self.transform_aug = transforms.Compose(
            [
                self.rand_aug,
                transforms.ToTensor(),
                transforms.Normalize(mean, std) if normalize else transforms.Lambda(lambda x: x),
            ]
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        img_clean = self.transform_clean(img)
        img_aug = self.transform_aug(img)
        visibility = torch.tensor(random.uniform(0.3, 1.0), dtype=torch.float32)
        return img_clean, img_aug, visibility, torch.tensor(label, dtype=torch.long)


class SingleTransformVisionDataset(Dataset):
    def __init__(self, base_dataset: datasets.VisionDataset, normalize: bool = True):
        self.base = base_dataset
        mean, std = CIFAR_MEAN_STD[base_dataset.__class__.__name__.lower()]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std) if normalize else transforms.Lambda(lambda x: x),
            ]
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return self.transform(img), torch.tensor(label, dtype=torch.long)


def _get_base_dataset(name: str, train: bool):
    root = Path(CACHE_DIR).expanduser()
    if name.lower() == "cifar10":
        return datasets.CIFAR10(root=root, train=train, download=True)
    elif name.lower() == "cifar100":
        return datasets.CIFAR100(root=root, train=train, download=True)
    else:
        raise ValueError(f"Unsupported dataset: {name}")


def get_datasets(cfg: DictConfig, subset_ratio: Optional[float] = None) -> Tuple[Dataset, Dataset, Dataset]:
    name = cfg.dataset.name.lower()
    num_classes = 10 if name == "cifar10" else 100
    cfg.dataset.num_classes = num_classes

    full_train_base = _get_base_dataset(name, train=True)
    test_base = _get_base_dataset(name, train=False)

    total_size = len(full_train_base)
    train_size = int(cfg.dataset.train_split)
    val_size = int(cfg.dataset.val_split)
    assert train_size + val_size <= total_size, "Train/val split exceeds size"

    indices = list(range(total_size))
    random.seed(cfg.training.seed)
    random.shuffle(indices)

    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]

    dual_aug_ds = DualAugVisionDataset(full_train_base, normalize=cfg.dataset.preprocessing.normalize)
    single_ds = SingleTransformVisionDataset(full_train_base, normalize=cfg.dataset.preprocessing.normalize)
    test_ds = SingleTransformVisionDataset(test_base, normalize=cfg.dataset.preprocessing.normalize)

    train_ds: Dataset = Subset(dual_aug_ds, train_idx)
    val_ds: Dataset = Subset(single_ds, val_idx)

    if subset_ratio is not None and 0 < subset_ratio < 1.0:
        sub_train_len = int(len(train_ds) * subset_ratio)
        sub_val_len = int(len(val_ds) * subset_ratio)
        train_ds = Subset(train_ds, list(range(sub_train_len)))
        val_ds = Subset(val_ds, list(range(sub_val_len)))

    return train_ds, val_ds, test_ds
