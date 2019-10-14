import numpy as np
from torch.utils import data
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from glob import glob
from PIL import Image
import os
import utils
import folder


def get_gsv_dataloader(case, batch_size):
    print('[INFO] Loading datasets: {}'.format(case))

    datas = {
        'GSV': "/data/dataset/FGDA/semi/nocrop/train_full/",
        'Web': "/data/dataset/FGDA/semi/gsv_100k_unwarp/",
        'Webtest': "/data/dataset/FGDA/semi/gsv_100k_unwarp/test/",
    }
    means = {
        'imagenet': [0.485, 0.456, 0.406]
    }
    stds = {
        'imagenet': [0.229, 0.224, 0.225]
    }

    config = {
        'is_semi': is_semi,
        'mode' : mode,
        'is_train' : is_train,
        'case' : case,
        'list' : list
    }

    img_size = (227, 227)

    transform = [
        transforms.Scale(img_size),
        transforms.ToTensor(),
        transforms.Normalize(means['imagenet'], stds['imagenet']),
    ]

    data_loader = data.DataLoader(
        dataset=folder.ImageFolder(
            datas[case],
            transform=transforms.Compose(transform),
        ),
        num_workers=16,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    return data_loader
