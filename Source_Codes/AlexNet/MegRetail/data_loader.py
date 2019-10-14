import numpy as np
from torch.utils import data
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from glob import glob
from PIL import Image
import os
import utils
from torch.utils.data import Dataset


def get_dataloader(batch_size, domain):
    root = '/data/dataset/FGDA/4evaluation'
    datas = {
        'sku': root + "/sku/unsupervised/",
        'shelf': root + "/shelf/unsupervised/",
        'web': root + "/Web/unsupervised/listbycls/",
    }
    img_size = (227, 227)

    transform = [
        transforms.Scale(img_size),
        transforms.ToTensor()
    ]

    data_loader = data.DataLoader(
        dataset=datasets.ImageFolder(
            datas[domain],
            transform=transforms.Compose(transform),
        ),
        num_workers=16,
        batch_size=batch_size,
        shuffle=True,
        # shuffle=False,
        drop_last=True
    )

    return data_loader


def get_dataloader_target(batch_size, domain, istrain):
    root = '/data/dataset/FGDA/4evaluation'
    datas = {
        'train': {
            'sku': root + '/sku/semi/train/',
            'shelf': root + '/shelf/semi/train/',
            'web': root + '/Web/semi/listbycls/train/'
        },
        'test': {
            'sku': root + '/sku/semi/test/',
            'shelf': root + '/shelf/semi/test/',
            'web': root + '/Web/semi/listbycls/test/'
        },
    }
    img_size = (227, 227)

    transform = [
        transforms.Scale(img_size),
        transforms.ToTensor()
    ]

    data_loader = data.DataLoader(
        dataset=datasets.ImageFolder(
            datas[istrain][domain],
            transform=transforms.Compose(transform),
        ),
        num_workers=16,
        batch_size=batch_size,
        shuffle=True,
        # shuffle=False,
        drop_last=True
    )

    return data_loader
