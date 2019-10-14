import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models


class classInvariant(nn.Module):
    def __init__(self, delta=1):
        super(classInvariant, self).__init__()
        self.cos = torch.nn.CosineEmbeddingLoss(margin=0, size_average=True)
        self.delta = delta

    def computeC(self, a, b):
        result = np.squeeze(np.ones((len(a), 1)))
        for i in range(0, len(a)):
            if a[i] != b[i]:
                result[i] = -1

        return result

    def to_var(self, x, volatile=False):
        x = x.cuda()
        return Variable(x, volatile=volatile)

    def forward(self, mmd1, mmd2, label1, label2):
        cosine = torch.nn.CosineSimilarity(dim=0)
        result = 0
        num = 0.0
        for i in range(0, len(label1)):
            for j in range(0, len(label2)):
                if label1[i] == label2[j]:
                    num += 1.0
                    result += max(self.delta - cosine(mmd1[i], mmd2[j]), 0)
                else:
                    num += 1.0
                    result += cosine(mmd1[i], mmd2[j])
        if num != 0:
            result /= num
            return result
        else:
            return self.to_var(torch.FloatTensor(1))



class Clssifier(nn.Module):
    def __init__(self, num_classes=1000):
        super(DeepCORAL, self).__init__()
        self.fc1 = nn.Linear(2048, num_classes)
        self.fc1.weight.data.normal_(0, 0.005)

    def forward(self, source):
        tmp_mmd = source
        source = self.fc1(source)

        return source, tmp_mmd


class ResNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        original_model = models.resnet50(True)
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        self.adv = list(original_model.children())[-2]

    def forward(self, x):
        x = self.features(x)
        feaure = x
        x = self.adv(x)
        return x.squeeze(), feaure


class DomainDis(nn.Module):
    def __init__(self):
        super(DomainDis, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=2),
            nn.ReLU(inplace=True),
        )
        self.fc0 = nn.Linear(2048, 1024)
        self.fc1 = nn.Linear(1024, 1)
        self.si = nn.Sigmoid()

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.si(x)
        return x

