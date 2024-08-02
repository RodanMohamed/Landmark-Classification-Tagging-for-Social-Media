import math
import torch
import torch.utils.data
from pathlib import Path
from torchvision import datasets, transforms
import multiprocessing
import matplotlib.pyplot as plt
from .helpers import compute_mean_and_std, get_data_location


def get_data_loaders(batch_size: int = 32, valid_size: float = 0.2, num_workers: int = 1, limit: int = -1):
    """
    Create and return the train, validation, and test data loaders.

    :param batch_size: Size of the mini-batches
    :param valid_size: Fraction of the dataset to use for validation (e.g., 0.2 means 20% for validation)
    :param num_workers: Number of workers to use in the data loaders. Default is 1.
    :param limit: Maximum number of data points to consider. Default is -1 (no limit).
    :return: A dictionary with keys 'train', 'valid', and 'test' containing the respective data loaders
    """

    data_loaders = {"train": None, "valid": None, "test": None}
    base_path = Path(get_data_location())

    mean, std = compute_mean_and_std()
    print(f"Dataset mean: {mean}, std: {std}")

    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandAugment(num_ops=2, magnitude=15, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        "valid": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    train_data = datasets.ImageFolder(base_path / "train", transform=data_transforms["train"])
    valid_data = datasets.ImageFolder(base_path / "train", transform=data_transforms["valid"])

    total_samples = len(train_data)
    indices = torch.randperm(total_samples)

    if limit > 0:
        indices = indices[:limit]
        total_samples = limit

    split_idx = int(math.ceil(valid_size * total_samples))
    train_indices, valid_indices = indices[split_idx:], indices[:split_idx]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_indices)

    data_loaders["train"] = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    data_loaders["valid"] = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)

    test_data = datasets.ImageFolder(base_path / "test", transform=data_transforms["test"])

    if limit > 0:
        test_sampler = torch.utils.data.SubsetRandomSampler(torch.arange(limit))
    else:
        test_sampler = None

    data_loaders["test"] = torch.utils.data.DataLoader(test_data, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers, shuffle=False)

    return data_loaders


def visualize_one_batch(data_loaders, max_n: int = 5):
    """
    Visualize one batch of data.

    :param data_loaders: Dictionary containing data loaders
    :param max_n: Maximum number of images to show
    :return: None
    """

    data_iter = iter(data_loaders["train"])
    images, labels = next(data_iter)

    mean, std = compute_mean_and_std()
    inv_transforms = transforms.Compose([
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0 / s for s in std]),
        transforms.Normalize(mean=[-m for m in mean], std=[1.0, 1.0, 1.0])
    ])

    images = inv_transforms(images)

    class_names = data_loaders["train"].dataset.classes
    images = images.permute(0, 2, 3, 1).clip(0, 1)

    fig = plt.figure(figsize=(25, 4))
    for idx in range(max_n):
        ax = fig.add_subplot(1, max_n, idx + 1, xticks=[], yticks=[])
        ax.imshow(images[idx])
        ax.set_title(class_names[labels[idx].item()])
    plt.show()

######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    return get_data_loaders(batch_size=2, num_workers=0)


def test_data_loaders_keys(data_loaders):
    assert set(data_loaders.keys()) == {"train", "valid", "test"}, "The keys of the data_loaders dictionary should be train, valid and test"

def test_data_loaders_output_type(data_loaders):
    # Test the data loaders
    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.__next__()

    assert isinstance(images, torch.Tensor), "images should be a Tensor"
    assert isinstance(labels, torch.Tensor), "labels should be a Tensor"
    assert images.shape[-1] == 224, "The tensors returned by your dataloaders should be 224x224. Did you " \
                                     "forget to resize and/or crop?"

def test_data_loaders_output_shape(data_loaders):
    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.__next__()

    assert len(images) == 2, f"Expected a batch of size 2, got size {len(images)}"
    assert len(labels) == 2, f"Expected a labels tensor of size 2, got size {len(labels)}"

def test_visualize_one_batch(data_loaders):
    visualize_one_batch(data_loaders, max_n=2)
