from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo
from torch.autograd import Variable
from torch.autograd import Function, Variable
from torch.autograd import grad
from torchvision.utils import save_image
from torchvision import transforms
import numpy as np
import os
import time
import datetime
from PIL import Image
import utils
from data_loader import *
from tqdm import tqdm
import cv2
import math

CUDA = True if torch.cuda.is_available() else False

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from models import *

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


class mySolver(object):

    def __init__(self, config):
        if config['random'] == False:
            SEED = 0
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)

        self.config = config

        self.CUDA = True
        self.BATCH_SIZE = config['batch_size']
        self.EPOCHS = config['epoch']
        self.is_semi = False

        self.source_loader = get_dataloader(batch_size=self.BATCH_SIZE[0], domain=config['source'])
        print('source : {}_{}'.format(len(self.source_loader), len(self.source_loader.dataset)))
        self.target_loader = get_dataloader_target(batch_size=self.BATCH_SIZE[1], domain=config['target'],
                                                   istrain='train')
        print('target : {}_{}'.format(len(self.target_loader), len(self.target_loader.dataset)))

        self.source_loader_test = self.source_loader
        self.target_loader_test = get_dataloader_target(batch_size=self.BATCH_SIZE[1], domain=config['target'],
                                                        istrain='test')
        print('target_test : {}_{}'.format(len(self.target_loader_test), len(self.target_loader_test.dataset)))

        self.LEARNING_RATE = config['lr']
        self.WEIGHT_DECAY = config['weight_decay']
        self.MOMENTUM = config['momentum']

        self.build_model()
        self.load_model()

    def build_model(self):
        self.featureExactor = AlexNet()
        self.featureExactor.cuda()

        self.featureExactor1 = AlexNet()
        self.featureExactor1.cuda()

        self.classfier = Clssifier(200)
        self.classfier.cuda()

        self.domain = [1, 1]
        for i in range(2):
            self.domain[i] = DomainDis()
            self.domain[i].cuda()

        # i.e. 10 times learning rate for the last two fc layers.
        self.optimizer_F = torch.optim.SGD(self.featureExactor.parameters(),
                                           lr=self.config['lr_f'] * self.LEARNING_RATE,
                                           momentum=self.MOMENTUM)
        self.optimizer_F1 = torch.optim.SGD(self.featureExactor1.parameters(),
                                            lr=self.config['lr_f'] * self.LEARNING_RATE,
                                            momentum=self.MOMENTUM)
        self.optimizer_C = torch.optim.SGD(self.classfier.parameters(), lr=self.config['lr_c'] * self.LEARNING_RATE,
                                           momentum=self.MOMENTUM)
        self.optimizer_D0 = torch.optim.SGD(self.domain[0].parameters(), lr=self.config['lr_d'] * self.LEARNING_RATE,
                                            momentum=self.MOMENTUM)
        self.optimizer_D1 = torch.optim.SGD(self.domain[1].parameters(), lr=self.config['lr_d'] * self.LEARNING_RATE,
                                            momentum=self.MOMENTUM)

        self.optimizer_main = torch.optim.SGD(
            list(self.featureExactor.parameters()) + list(self.classfier.parameters()),
            lr=self.config['lr_f'] * self.LEARNING_RATE,
            momentum=self.MOMENTUM,
            weight_decay=self.WEIGHT_DECAY)

        self.l1 = nn.L1Loss(size_average=True)
        self.cos = classInvariant()

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def grad_reset(self):
        self.optimizer_C.zero_grad()
        self.optimizer_F.zero_grad()
        self.optimizer_F1.zero_grad()
        self.optimizer_D0.zero_grad()
        self.optimizer_D1.zero_grad()
        self.optimizer_main.zero_grad()

    def load_model(self):
        url = '/data/pretrain_model/alexnet-owt-4df8aa71.pth'
        pretrained_dict = torch.load(url)
        model_dict = self.featureExactor.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.featureExactor.load_state_dict(model_dict)
    
        pretrained_dict = torch.load(url)
        model_dict = self.featureExactor1.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.featureExactor1.load_state_dict(model_dict)
    
        pretrained_dict = torch.load(url)
        model_dict = self.classfier.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.classfier.load_state_dict(model_dict)

    def cropBatch(self, batch, featuremap):
        featuremap = featuremap
        featuremap = np.sum(featuremap.data.cpu().numpy(), axis=1)

        newBatch = []

        for i in range(len(batch)):
            tmp_heat = featuremap[i]
            tmp_heat -= np.sum(tmp_heat) / (tmp_heat.shape[0] * 2)
            tmp_heat = np.maximum(tmp_heat, 0)
            tmp_heat /= np.max(tmp_heat)
            tmp_heatmap = np.uint8(255 * tmp_heat)
            _, binary = cv2.threshold(tmp_heatmap, 127, 255, cv2.THRESH_BINARY)

            contours, _2 = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                tmp_max = -1
                tmp_maxi = -1
                for i in range(len(contours)):
                    cnt = contours[i]
                    _, _, w, h = cv2.boundingRect(cnt)
                    if w * h > tmp_max:
                        tmp_max = w * h
                        tmp_maxi = i
                tmpx, tmpy, tmpw, tmph = cv2.boundingRect(contours[tmp_maxi])
                tmpx1, tmpy1, tmpx2, tmpy2 = int(tmpx * 227 / tmp_heat.shape[0]), int(
                    tmpy * 227 / tmp_heat.shape[0]), int(
                    math.ceil((tmpx + tmpw) * 227 / tmp_heat.shape[0])), int(
                    math.ceil((tmpy + tmph) * 227 / tmp_heat.shape[0]))

                tmp_img = batch[i].data.cpu().numpy().transpose(1, 2, 0)
                tmp_img = Image.fromarray(np.uint8(tmp_img))
                tmp_bbox = (tmpx1, tmpy1, tmpx2, tmpw)
                tmp_bbox = tuple(tmp_bbox)
                tmp_img = tmp_img.crop(tmp_bbox).resize((227, 227))
                tmpiimg = np.asarray(tmp_img)
            else:
                tmpiimg = batch[i].data.cpu().numpy().transpose(1, 2, 0)

            newBatch.append(tmpiimg)

        newBatch = np.array(newBatch).transpose(0, 3, 1, 2)
        return self.to_var(torch.from_numpy(newBatch).float())

    def train_perEpoch(self, epoch):
        result = []

        source, target = list(enumerate(self.source_loader)), list(enumerate(self.target_loader))
        target_test = list(enumerate(self.target_loader_test))
        train_steps = len(source)

        for batch_idx in range(train_steps):
            _, (source_data, source_label) = source[batch_idx]
            _, (target_data, target_label) = target[batch_idx % len(target)]
            _, (target_data_test, _) = target_test[batch_idx % len(target_test)]

            source_data = self.to_var(source_data)
            source_label = self.to_var(source_label)
            target_data = self.to_var(target_data)
            target_label = self.to_var(target_label)
            target_data_test = self.to_var(target_data_test)

            if batch_idx % self.config['FC/D'] == 0:
                # train D
                self.grad_reset()

                _, feature1 = self.featureExactor(source_data)
                _, feature2 = self.featureExactor(target_data)
                _, feature3 = self.featureExactor(target_data_test)

                domain11 = self.domain[0](feature1)
                domain12 = self.domain[1](feature1)
                domain21 = self.domain[0](feature2)
                domain22 = self.domain[1](feature2)

                tmp_1a = self.to_var(torch.FloatTensor(np.ones(domain11.shape)))
                tmp_1b = self.to_var(torch.FloatTensor(np.zeros(domain11.shape)))
                tmp_2a = self.to_var(torch.FloatTensor(np.zeros(domain21.shape)))
                tmp_2b = self.to_var(torch.FloatTensor(np.ones(domain21.shape)))

                domaindisloss0 = self.l1(domain11, tmp_1a) + self.l1(domain12, tmp_1b)
                domaindisloss1 = self.l1(domain21, tmp_2a) + self.l1(domain22, tmp_2b)
                domaindisloss_d = domaindisloss0 + domaindisloss1
                domaindisloss_d.backward()

                self.optimizer_D0.step()
                self.optimizer_D1.step()

            # train F
            self.grad_reset()

            out1_re, feature1 = self.featureExactor(source_data)
            out2_re, feature2 = self.featureExactor(target_data)

            out1, mmd_1 = self.classfier(out1_re)
            out2, mmd_2 = self.classfier(out2_re)

            try:
                source_data_spatial = self.cropBatch(source_data, feature1)
                target_data_spatial = self.cropBatch(target_data, feature2)

                # crop
                out1_re_spatial, _ = self.featureExactor1(source_data_spatial)
                out2_re_spatial, _ = self.featureExactor1(target_data_spatial)

                out1_spatial, _ = self.classfier(out1_re_spatial)
                out2_spatial, _ = self.classfier(out2_re_spatial)

                out1 = self.config['ori'] * out1 + (1 - self.config['ori']) * out1_spatial
                out2 = self.config['ori'] * out2 + (1 - self.config['ori']) * out2_spatial
            except IndexError:
                out1 = out1
                out2 = out2

            classification_loss = torch.nn.functional.cross_entropy(out1, source_label) \
                                  + torch.nn.functional.cross_entropy(out2, target_label)
            class_invariant = self.cos(mmd_1, mmd_2, source_label.data, target_label.data)

            domain11 = self.domain[0](feature1)
            domain12 = self.domain[1](feature1)
            domain21 = self.domain[0](feature2)
            domain22 = self.domain[1](feature2)
            loss1 = torch.abs(domain11 - domain12)
            loss1 = loss1.sum() / self.BATCH_SIZE[0]
            loss2 = torch.abs(domain21 - domain22)
            loss2 = loss2.sum() / self.BATCH_SIZE[1]

            domaindisloss = loss1 + loss2
            domaindisloss /= 2

            f_loss = classification_loss + self.config['domain'] * domaindisloss + self.config[
                'class'] * class_invariant
            f_loss.backward()

            self.optimizer_F.step()
            self.optimizer_F1.step()

            c_loss = classification_loss

            self.optimizer_C.step()

            result.append({
                'epoch': epoch,
                'step': batch_idx + 1,
                'total_steps': train_steps,
                'classification_loss': classification_loss.item(),
                'domain_loss': domaindisloss.item(),
                'class_loss': class_invariant.item(),
                'f_loss': f_loss.item(),
                'c_loss': c_loss.item(),
                'd_loss': domaindisloss_d.item()
            })

        return result

    def train(self):
        for key in self.config.keys():
            print(key + '\t' + str(self.config[key]))
        training_statistic = []
        testing_s_statistic = []
        testing_t_statistic = []

        max_acc = 0
        i = 0

        self.best_cls = self.classfier
        self.best_feature = self.featureExactor
        self.best_feature1 = self.featureExactor1
        self.best_domain0 = self.domain[0]
        self.best_domain1 = self.domain[1]

        for e in tqdm(range(0, self.EPOCHS)):
            self._lambda = (e + 1) / self.EPOCHS
            res = self.train_perEpoch(e + 1)
            tqdm_result = '###EPOCH {}: Class: {:.6f}, domain: {:.6f}, cls-inv: {:.6f}, f_Loss: {:.6f}, c_Loss: {:.6f}, d_Loss: {:.6f}'.format(
                e + 1,
                sum(row['classification_loss'] / row['total_steps'] for row in res),
                sum(row['domain_loss'] / row['total_steps'] for row in res),
                sum(row['class_loss'] / row['total_steps'] for row in res),
                sum(row['f_loss'] / row['total_steps'] for row in res),
                sum(row['c_loss'] / row['total_steps'] for row in res),
                sum(row['d_loss'] / row['total_steps'] for row in res),
            )
            tqdm.write(tqdm_result)

            training_statistic.append(res)

            test_source = self.test(self.source_loader_test, e)
            test_target = self.test(self.target_loader, e)
            test_target_test = self.test(self.target_loader_test, e)
            testing_s_statistic.append(test_source)
            testing_t_statistic.append(test_target)

            tqdm_result = '###Test Source: Epoch: {}, avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                e + 1,
                test_source['average_loss'],
                test_source['correct'],
                test_source['total'],
                test_source['accuracy'],
            )
            tqdm.write(tqdm_result)
            tqdm_result = '###Test Target: Epoch: {}, avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                e + 1,
                test_target['average_loss'],
                test_target['correct'],
                test_target['total'],
                test_target['accuracy'],
            )
            tqdm.write(tqdm_result)
            tqdm_result = '###Test Target test: Epoch: {}, avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                e + 1,
                test_target_test['average_loss'],
                test_target_test['correct'],
                test_target_test['total'],
                test_target_test['accuracy'],
            )
            tqdm.write(tqdm_result)

            if test_target_test['accuracy'] > max_acc:
                max_acc = test_target_test['accuracy']
                i = e

                self.best_cls = self.classfier
                self.best_feature = self.featureExactor
                self.best_feature1 = self.featureExactor1
                self.best_domain0 = self.domain[0]
                self.best_domain1 = self.domain[1]

            bestnow = '###Epoch: {},Accuracy: {:.2f}'.format(
                i,
                max_acc,
            )
            tqdm.write(bestnow)

        for key in self.config.keys():
            print(key + '\t' + str(self.config[key]))

        print(bestnow)

        root = '/data/result/FGDA/FGDA/semi-supervised/MegRetail(AlexNet)-' + self.config['source'] + '2' + self.config[
            'target'] + '{:.4f}'.format(max_acc) + '/'
        if not os.path.exists(root):
            os.makedirs(root)

        tqdm.write(utils.save_net(self.best_cls, root + '/classifier_final.tar'))
        tqdm.write(utils.save_net(self.best_feature, root + '/featureExactor_final.tar'))
        tqdm.write(utils.save_net(self.best_feature1, root + '/featureExactor1_final.tar'))
        tqdm.write(utils.save_net(self.domain[0], root + '/domainDiscriminator0_final.tar'))
        tqdm.write(utils.save_net(self.domain[1], root + '/domainDiscriminator1_final.tar'))
        print(utils.save(training_statistic, root + '/training_statistic.pkl'))
        print(utils.save(testing_s_statistic, root + '/testing_s_statistic.pkl'))
        print(utils.save(testing_t_statistic, root + '/testing_t_statistic.pkl'))

    def test(self, dataset_loader, e):
        self.classfier.eval()
        self.featureExactor.eval()
        test_loss = 0
        correct = 0
        for data, target in dataset_loader:
            data, target = data.cuda(), target.cuda()

            data, target = Variable(data, volatile=True), Variable(target)

            tmp_data, feature1 = self.featureExactor(data)
            out, _ = self.classfier(tmp_data)
            try:
                source_data0 = self.cropBatch(data, feature1)
                # crop
                out1_re0, _ = self.featureExactor1(source_data0)

                out10, _ = self.classfier(out1_re0)

                out = self.config['ori'] * out + (1 - self.config['ori']) * out10
            except IndexError:
                out = out

            test_loss += torch.nn.functional.cross_entropy(out, target, size_average=False).item()

            pred = out.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(dataset_loader.dataset)

        return {
            'epoch': e,
            'average_loss': test_loss,
            'correct': correct,
            'total': len(dataset_loader.dataset),
            'accuracy': 100. * float(correct) / float(len(dataset_loader.dataset))
        }
