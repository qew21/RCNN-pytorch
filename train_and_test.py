import os
import sys
import argparse
import numpy as np
import config as cfg
import process_data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
import time
import cv2
import Networks
import torch.optim as optim

#修复最后的判别问题

import joblib
flower = {1: 'pancy', 2: 'Tulip'}
preprocess = transforms.Compose([
    transforms.ToTensor(),
])


def default_loader(img_pil):
    img_tensor = preprocess(img_pil)
    return img_tensor


def reg_loader(img_pil):
    img_tensor = torch.Tensor(img_pil)
    return img_tensor


class trainset(Dataset):
    def __init__(self, images, target, loader=default_loader):
        #定义好 image 的路径
        self.images = images
        self.target = target
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        return img,target

    def __len__(self):
        return len(self.images)


class Solver(object):
    def __init__(self, train_set, net_type):
        self.model = None
        self.class_num = 17
        self.net_type = net_type

        if net_type[0] == "F":
            self.initial_learning_rate = cfg.F_learning_rate
            self.model_out_path = cfg.F_model
            self.epochs = cfg.F_epoch
            self.class_num = 3
        elif net_type == "R":
            self.initial_learning_rate = cfg.R_learning_rate
            self.epochs = cfg.R_epoch
            self.model_out_path = cfg.R_model
        else:
            self.initial_learning_rate = cfg.T_learning_rate
            self.epochs = cfg.T_epoch
            self.model_out_path = cfg.T_model

        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.train_loader = train_set
        self.test_loader = train_set

    def load_model(self):
        if cfg.cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        if self.net_type == "F2":
            self.model = torch.load(cfg.F_model)
            removed = list(self.model.classifier.children())[:-2]
            self.model.classifier = torch.nn.Sequential(*removed)
        elif os.path.exists(self.model_out_path):
            self.model = torch.load(self.model_out_path)
        elif self.net_type == "F1":
            self.model = torch.load(cfg.T_model)
            self.model.classifier._modules['6'] = nn.Linear(4096, 3).to(self.device, torch.float)
        elif self.net_type == "R":
            self.model = Networks.RegNet().to(self.device, torch.float)
        else:
            self.model = Networks.AlexNet(num_classes=self.class_num).to(self.device, torch.float)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.initial_learning_rate)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75, 150], gamma=0.5)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train(self):
        print("train:")
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):
            if self.net_type == "R":
                data, target = data.to(self.device,torch.float), target.to(self.device, dtype=torch.float)
            else:
                data, target = data.to(self.device, torch.float), target.to(self.device, dtype=torch.long)
            self.optimizer.zero_grad()
            output = self.model(data)
            if self.net_type == "R":
                no_object_loss = torch.mean(torch.square((1 - target[:, 0]) * output[:, 0]))
                object_loss = torch.mean(torch.square((target[:, 0]) * (output[:, 0] - 1)))
                loss = (torch.mean(target[:, 0] * (
                    torch.sum(torch.square(target[:, 1:5] - output[:, 1:5]), 1))) + no_object_loss + object_loss)
            else:
                loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            prediction = torch.max(output, 1)
            total += target.size(0)
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

            process_data.progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_num + 1), 100. * train_correct / total, train_correct, total))

        return train_loss, train_correct / total

    def test(self):
        print("test:")
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                if self.net_type == "R":
                    data, target = data.to(self.device, torch.float), target.to(self.device, dtype=torch.float)
                else:
                    data, target = data.to(self.device, torch.float), target.to(self.device, dtype=torch.long)
                output = self.model(data)
                if self.net_type == "R":
                    no_object_loss = torch.mean(torch.square((1 - target[:, 0]) * output[:, 0]))
                    object_loss = torch.mean(torch.square((target[:, 0]) * (output[:, 0] - 1)))
                    loss = (torch.mean(target[:, 0] * (
                        torch.sum(torch.square(target[:, 1:5] - output[:, 1:5]), 1))) + no_object_loss + object_loss)
                else:
                    loss = self.criterion(output, target)
                test_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
                process_data.progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))

        return test_loss, test_correct / total

    def save(self):
        torch.save(self.model, self.model_out_path)
        print("Checkpoint saved to {}".format(self.model_out_path))

    def run(self):
        self.load_model()
        if not os.path.exists(self.model_out_path):
            accuracy = 0
            for epoch in range(1, self.epochs + 1):

                print("\n===> epoch: %d/%d" % (epoch,self.epochs))
                train_result = self.train()
                self.scheduler.step()
                print(train_result)
                test_result = self.test()
                accuracy = max(accuracy, test_result[1])
                if epoch == self.epochs:
                    print("===> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))
                    self.save()

    def predict(self, input_data, reg=False):
        if reg:
            test_data = trainset(input_data, input_data, loader=reg_loader)
        else:
            test_data = trainset(input_data, input_data)
        train_loader1 = torch.utils.data.DataLoader(test_data, batch_size=1)
        res = []
        self.optimizer.zero_grad()
        for batch_num, (data, target) in enumerate(train_loader1):
            data= data.to(self.device,torch.float)
            output = self.model(data)
            res.append(output)
        return res


if __name__ == "__main__":
    TEST = True
    Train_alexnet_data = process_data.Train_Alexnet_Data()
    i,l=Train_alexnet_data.get_data()
    l=[np.argmax(c) for c in l]
    train_dataset = trainset(i,l)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.T_batch_size, shuffle=True)
    AlexNet_solver = Solver(train_loader, net_type="T")
    AlexNet_solver.run()
    if TEST:
        cnt = 0
        for img_file in os.listdir("17flowers/jpg/0/"):
            image_path = os.path.join("17flowers/jpg/0/", img_file)
            img = cv2.imread(image_path)
            img = np.array(img)
            test_img = process_data.resize_image(img, 224, 224)
            test_img = np.asarray(test_img, dtype='float32')
            rr = AlexNet_solver.predict([test_img])
            print(np.argmax(rr[0].cpu().detach().numpy()))
            cnt += 1
            if cnt > 5:
                cnt = 0
                break
        for img_file in os.listdir("17flowers/jpg/3/"):
            image_path = os.path.join("17flowers/jpg/3/", img_file)
            img = cv2.imread(image_path)
            img = np.array(img)
            test_img = process_data.resize_image(img, 224, 224)
            test_img = np.asarray(test_img, dtype='float32')
            rr = AlexNet_solver.predict([test_img])
            print(np.argmax(rr[0].cpu().detach().numpy()))
            cnt += 1
            if cnt > 5:
                break

    Fineturn_data = process_data.FineTun_And_Predict_Data()
    i,l=Fineturn_data.get_data()
    l=[np.argmax(c) for c in l]
    train_dataset = trainset(i,l)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.F_batch_size, shuffle=True)
    Fineturn_solver = Solver(train_loader, net_type="F1")
    Fineturn_solver.run()
    if TEST:
        for image_path in ['./2flowers/jpg/0/image_0561.jpg', './2flowers/jpg/0/image_0562.jpg',
                           './2flowers/jpg/1/image_1286.jpg', './2flowers/jpg/1/image_1282.jpg']:
            img = cv2.imread(image_path)
            img = np.array(img)
            test_img = process_data.resize_image(img, 224, 224)
            test_img = np.asarray(test_img, dtype='float32')
            rr = Fineturn_solver.predict([test_img])
            print(np.argmax(rr[0].cpu().detach().numpy()))

    Features_solver = Solver(None, net_type="F2")
    Features_solver.load_model()
    Features_data = process_data.FineTun_And_Predict_Data(Features_solver, is_svm=True, is_save=True)

    svms = []
    if not os.path.exists(r'./output/SVM_model') :
        os.makedirs(r'./output/SVM_model')
    if len(os.listdir(r'./output/SVM_model')) == 0:
        SVM_net = Networks.SVM(Features_data)
        SVM_net.train()
    for file in os.listdir(r'./output/SVM_model'):
        svms.append(joblib.load(os.path.join('./output/SVM_model', file)))

    i, l = Features_data.get_Reg_batch()
    train_dataset = trainset(i, l, loader=reg_loader)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.R_batch_size, shuffle=True)
    Reg_box_solver = Solver(train_loader, net_type="R")
    Reg_box_solver.run()

    for img_path in ['./2flowers/jpg/0/image_0561.jpg', './2flowers/jpg/0/image_0562.jpg',
                     './2flowers/jpg/1/image_1286.jpg', './2flowers/jpg/1/image_1282.jpg']:
        imgs, verts = process_data.image_proposal(img_path)
        img_float = np.asarray(imgs, dtype="float32")
        features = Features_solver.predict(img_float)

        results = []
        results_old = []
        results_label = []
        count = 0
        for f in features:
            for svm in svms:
                feature_data = f.tolist()
                pred = svm.predict(feature_data)
                if pred[0] != 0:
                    results_old.append(verts[count])
                    tes = Reg_box_solver.predict(feature_data, reg=True)
                    res = tes[0].cpu().detach().numpy()
                    if res[0][0] > 0.5:
                        px, py, pw, ph = verts[count][0], verts[count][1], verts[count][2], verts[count][3]
                        old_center_x, old_center_y = px + pw / 2.0, py + ph / 2.0
                        x_ping, y_ping, w_suo, h_suo = res[0][1], res[0][2], res[0][3], res[0][4]
                        new__center_x = x_ping * pw + old_center_x
                        new__center_y = y_ping * ph + old_center_y
                        new_w = pw * np.exp(w_suo)
                        new_h = ph * np.exp(h_suo)
                        new_verts = [new__center_x, new__center_y, new_w, new_h]
                        results.append(new_verts)
                        results_label.append(pred[0])
            count += 1

        average_center_x, average_center_y, average_w, average_h = 0, 0, 0, 0
        # 给预测出的所有的预测框区一个平均值，代表其预测出的最终位置
        for vert in results:
            average_center_x += vert[0]
            average_center_y += vert[1]
            average_w += vert[2]
            average_h += vert[3]
        if results:
            average_center_x = average_center_x / len(results)
            average_center_y = average_center_y / len(results)
            average_w = average_w / len(results)
            average_h = average_h / len(results)
            average_result = [[average_center_x, average_center_y, average_w, average_h]]
            result_label = max(results_label, key=results_label.count)
            process_data.show_rect(img_path, results_old, ' ')
            process_data.show_rect(img_path, average_result, flower[result_label])
        else:
            print("None.")