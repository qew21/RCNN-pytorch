# coding: utf8
import os
import config as cfg
from sklearn import svm
import joblib
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch


class AlexNet(nn.Module):
    """Alexnet model."""
    def __init__(self, num_classes=17):
        """ Init
        Args:
            num_classes (int): The number of output classes
        """
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # [n, 3, 224, 224]
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # [n, 96, 55, 55]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # [n, 96, 27, 27]
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # [n, 256, 27, 27]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # [n, 256, 13, 13]
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # [n, 384, 13, 13]
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # [n, 384, 13, 13]
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # [n, 256, 13, 13]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # [n, 256, 6, 6]
        )
        self.classifier = nn.Sequential(
            # [n, 256 * 6 * 6]
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            # [n, 4096]
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # [n, 4096]
            nn.Linear(4096, num_classes),
            # [n, num_classes]
        )

    def forward(self, x):
        """Pytorch forward function implementation."""
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    # def run(self):
    #     self.load_data()
    #     self.load_model()
    #     accuracy = 0
    #     for epoch in range(1, self.epochs + 1):
    #         self.scheduler.step(epoch)
    #         print("\n===> epoch: %d/200" % epoch)
    #         train_result = self.train()
    #         print(train_result)
    #         test_result = self.test()
    #         accuracy = max(accuracy, test_result[1])
    #         if epoch == self.epochs:
    #             print("===> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))
    #             self.save()


class SVM :
    def __init__(self, data):
        self.data = data
        self.data_save_path = cfg.SVM_and_Reg_save
        self.output = cfg.Out_put

    def train(self):
        import numpy as np
        svms=[]
        data_dirs = os.listdir(self.data_save_path)
        for data_dir in data_dirs:
            images, labels = self.data.get_SVM_data(data_dir)
            image_data = [i[0].cpu().detach().numpy() for i in images]
            print(np.shape(image_data))
            print(np.shape(labels))
            clf = svm.LinearSVC()
            clf.fit(image_data, labels)
            svms.append(clf)
            SVM_model_path = os.path.join(self.output, 'SVM_model')
            if not os.path.exists(SVM_model_path):
                os.makedirs(SVM_model_path)
            joblib.dump(clf, os.path.join(SVM_model_path,  str(data_dir)+ '_svm.pkl'))


class RegNet(nn.Module):
    """RegNet model."""
    def __init__(self, num_classes=5):
        """ Init
        Args:
            num_classes (int): The number of output classes
        """
        super(RegNet, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.Tanh(),
            # [n, 4096]
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        """Pytorch forward function implementation."""
        x = self.features(x)
        return x

    # def run(self):
    #     self.load_data()
    #     self.load_model()
    #     accuracy = 0
    #     for epoch in range(1, self.epochs + 1):
    #         self.scheduler.step(epoch)
    #         print("\n===> epoch: %d/200" % epoch)
    #         train_result = self.train()
    #         print(train_result)
    #         test_result = self.test()
    #         accuracy = max(accuracy, test_result[1])
    #         if epoch == self.epochs:
    #             print("===> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))
    #             self.save()