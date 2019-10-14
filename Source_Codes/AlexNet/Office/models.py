import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class classInvariant(nn.Module):
    def __init__(self, delta = 1):
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


class Classfier(nn.Module):
    def __init__(self, num_classes=1000):
        super(DeepCORAL, self).__init__()
        self.fc1 = nn.Linear(4096, num_classes)
        self.fc1.weight.data.normal_(0, 0.005)

    def forward(self, source):
        tmp_mmd = source
        source = self.fc1(source)

        return source, tmp_mmd


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        feaure = x
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x, feaure


class DomainDis(nn.Module):
    def __init__(self):
        super(DomainDis, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fc0 = nn.Linear(4096, 1024)
        self.fc1 = nn.Linear(1024, 1)
        self.si = nn.Sigmoid()

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = x.view(x.size(0), 1024 * 2 * 2)
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.si(x)
        return x

