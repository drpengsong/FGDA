import numpy as np
from torch.utils import data
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from glob import glob
from PIL import Image
import os
import utils
import folder

# For Office31 datasets data_loader
def get_office31_dataloader(case, mode, batch_size):
    print('[INFO] Loading datasets: {}'.format(case))
    datas = {
    'source':{
        'amazon': '/data/dataset/FGDA/domain_adaptation_images/amazon/images/source/',
        'dslr': '/data/dataset/FGDA/domain_adaptation_images/dslr/images/source/',
        'webcam': '/data/dataset/FGDA/domain_adaptation_images/webcam/images/source/'
    },
    'targettrain':{
        'amazon': '/data/dataset/FGDA/domain_adaptation_images/amazon/images/targettrain/',
        'dslr': '/data/dataset/FGDA/domain_adaptation_images/dslr/images/targettrain/',
        'webcam': '/data/dataset/FGDA/domain_adaptation_images/webcam/images/targettrain/'
    },
    'targettest':{
        'amazon': '/data/dataset/FGDA/domain_adaptation_images/amazon/images/targettest/',
        'dslr': '/data/dataset/FGDA/domain_adaptation_images/dslr/images/targetest/',
        'webcam': '/data/dataset/FGDA/domain_adaptation_images/webcam/images/targettest/'
    }    

    }
    means = {
        'amazon': [0.79235075407833078, 0.78620633471295642, 0.78417965306916637],
        'webcam': [0.61197983011509638, 0.61876474000372972, 0.61729662103473015],
        'dslr': [],
        'imagenet': [0.485, 0.456, 0.406]
    }
    stds = {
        'amazon': [0.27691643643313618, 0.28152348841965347, 0.28287296762830788],
        'webcam': [0.22763857108616978, 0.23339382150450594, 0.23722725519031848],
        'dslr': [],
        'imagenet': [0.229, 0.224, 0.225]
    }

    img_size = (227, 227)

    transform = [
        transforms.Scale(img_size),
        transforms.ToTensor(),
        transforms.Normalize(means[case], stds[case]),
    ]

    data_loader = data.DataLoader(
        dataset=folder.ImageFolder(
            datas[mode[case]],
            transform=transforms.Compose(transform),
        ),
        num_workers=2,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    return data_loader
