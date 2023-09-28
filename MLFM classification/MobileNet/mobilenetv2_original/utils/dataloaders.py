import os
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def get_pytorch_train_transform(input_size = 256):
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_pytorch_val_transform(input_size = 256):
    return transforms.Compose([
        transforms.Resize(int(input_size/0.875)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_pytorch_train_loader(data_path, batch_size, input_size =256, workers=5, _worker_init_fn=None):
    traindir = os.path.join(data_path, 'train')
    train_dataset = datasets.ImageFolder(
        traindir,
        get_pytorch_train_transform(input_size),
    )

    if torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True, sampler=train_sampler
    )

    return train_loader, len(train_loader)


def get_pytorch_val_loader(data_path, batch_size,input_size =256, workers=5, _worker_init_fn=None):
    valdir = os.path.join(data_path, 'val')
    val_dataset = datasets.ImageFolder(
        valdir,
        get_pytorch_val_transform(input_size),
    )

    if torch.distributed.is_initialized():
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        val_sampler = None

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True,
    )

    return val_loader, len(val_loader)
