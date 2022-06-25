# -*- coding:utf-8 -*-

"""
信号设计课程小组设计

@ by: Leaf
@ date: 2022-05-28
"""

import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
import shutil
import math
from os.path import exists
from os import mkdir
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset


# 旋转函数
def rotate(angle, x, y, point_x, point_y):
    px = (x - point_x) * math.cos(angle) - (y - point_y) * math.sin(angle) + point_x
    py = (x - point_x) * math.sin(angle) + (y - point_y) * math.cos(angle) + point_y
    return px, py


class CNN(nn.Module):
    def __init__(self, m):
        super(CNN, self).__init__()
        self.out_label = []
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3),
        )
        self.med = nn.Linear(32 * 7 * 1, 500)
        self.med2 = nn.Linear(1*21*3, 100)
        self.med3 = nn.Linear(100, 500)
        self.out = nn.Linear(500, m)  # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        x = self.med(x)
        # x = self.med2(x)
        # x = self.med3(x)
        output = self.out(x)
        return output


class HandDetector:
    """
    使用mediapipe库查找手。导出地标像素格式。添加了额外的功能。
    如查找方式，许多手指向上或两个手指之间的距离。而且提供找到的手的边界框信息。
    """

    def __init__(self, mode=False, max_hands=2, detection_con=0.5, min_track_con=0.5):
        """
        :param mode: 在静态模式下，对每个图像进行检测
        :param max_hands: 要检测的最大手数
        :param detection_con: 最小检测置信度
        :param min_track_con: 最小跟踪置信度
        """
        self.results = None
        self.mode = mode
        self.max_hands = max_hands
        self.modelComplex = 1
        self.detection_con = detection_con
        self.min_track_con = min_track_con

        # 初始化手部的识别模型
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.max_hands,
                                        min_detection_confidence=self.detection_con,
                                        min_tracking_confidence=self.min_track_con)
        self.mpDraw = mp.solutions.drawing_utils  # 初始化绘图器
        self.tipIds = [4, 8, 12, 16, 20]  # 指尖列表
        self.fingers = []
        self.lmList = []
        self.re_lmList = []

    def find_hands(self, img, draw=True):
        """
        从图像(BRG)中找到手部。
        :param img: 用于查找手的图像。
        :param draw: 在图像上绘制输出的标志。
        :return: 带或不带图形的图像
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将传入的图像由BGR模式转标准的Opencv模式——RGB模式，
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_no=0, draw=True):
        """
        查找单手的地标并将其放入列表中像素格式。还可以返回手部的周围的边界框。
        :param img: 要查找的主图像
        :param hand_no: 如果检测到多只手，则为手部id
        :param draw: 在图像上绘制输出的标志。(默认绘制矩形框)
        :return: 像素格式的手部关节位置列表；手部边界框
        """
        x_list = []
        y_list = []
        bbox_info = []
        self.lmList = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for i, lm in enumerate(my_hand.landmark):
                px, py = int(lm.x * w), int(lm.y * h)
                x_list.append(px)
                y_list.append(py)
                self.lmList.append([lm.x, lm.y, lm.z])
                if draw:
                    cv2.circle(img, (px, py), 5, (255, 0, 255), cv2.FILLED)
            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)
            box_w, box_h = x_max - x_min, y_max - y_min
            bbox = x_min, y_min, box_w, box_h
            cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)
            bbox_info = {"id": hand_no, "bbox": bbox, "center": (cx, cy), "shape": (h, w)}

            if draw:
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                              (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                              (0, 255, 0), 2)

        self.revolve(img)

        return self.re_lmList, bbox_info

    def revolve(self, img, draw=True):
        """
            旋转手势识别点
            :param img: 要查找的主图像
            :param draw: 在图像上绘制输出的标志。(默认绘制矩形框)
            :return: 像素格式的手部关节位置列表
        """
        h, w, c = img.shape
        if len(self.lmList) >= 21:
            # print(self.lmList)
            self.re_lmList = []
            point_x = self.lmList[0][0]
            point_y = self.lmList[0][1]
            delta_x = self.lmList[13][0] - point_x
            delta_y = self.lmList[13][1] - point_y
            if delta_y == 0:
                if delta_x < 0:
                    theta = math.pi / 2
                else:
                    theta = -math.pi / 2
            else:
                theta = math.atan(delta_x / delta_y)
                if delta_y > 0:
                    theta = theta + math.pi
            # print(theta*180/math.pi)
            for i in self.lmList:
                x, y = rotate(theta, i[0], i[1], point_x, point_y)
                px, py = int(x * w), int(y * h)
                self.re_lmList.append([x, y, i[2]])
                if draw:
                    cv2.circle(img, (px, py), 5, (0, 0, 255), cv2.FILLED)
        else:
            self.re_lmList = self.lmList
        return self.re_lmList

    def hand_type(self):
        """
        检查传入的手部 是左还是右
        :return: 1 或 0
        """
        if self.results.multi_hand_landmarks:
            if self.lmList[17][0] < self.lmList[5][0]:
                return 1
            else:
                return 0


class AI:
    def __init__(self, datasets_dir):
        self.EPOCH = 20
        self.BATCH_SIZE = 2
        self.LR = 10e-5
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.datasets_dir = datasets_dir
        self.train_loader = None
        self.m = 0
        self.out_label = []  # CNN网络输出后数字标签转和字符串标签的映射关系

    def load_datasets(self):
        train_data = []
        train_label = []
        self.m = 0
        for file in Path(self.datasets_dir).rglob("*.npz"):
            data = np.load(str(file))
            train_data.append(data["data"])
            label_number = np.ones(len(data["data"])) * len(self.out_label)
            train_label.append(label_number)
            self.out_label.append(data["label"])
            self.m += 1
        train_data = torch.Tensor(np.concatenate(train_data, axis=0))
        train_data = train_data.unsqueeze(1)
        train_label = torch.tensor(np.concatenate(train_label, axis=0)).long()

        dataset = TensorDataset(train_data, train_label)
        self.train_loader = DataLoader(dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        return self.m

    def train_cnn(self):
        cnn = CNN(self.m).to(self.DEVICE)
        optimizer = torch.optim.Adam(cnn.parameters(), self.LR)  # optimize all cnn parameters
        loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

        for epoch in range(self.EPOCH):
            for step, (data, target) in enumerate(self.train_loader):
                # 分配 batch data, normalize x when iterate train_loader
                data, target = data.to(self.DEVICE), target.to(self.DEVICE)
                output = cnn(data)  # cnn output
                loss = loss_func(output, target)  # cross entropy loss
                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients
                if (step + 1) % 100 == 0:  # 输出结果
                    if (step + 1) % 100 == 0:  # 输出结果
                        print(
                            "\r[Epoch: %d] [%d/%d (%0.f %%)][Loss: %f]"
                            % (
                                epoch,
                                step * len(data),
                                len(self.train_loader.dataset),
                                100. * step / len(self.train_loader),
                                loss.item()
                            ), end="")

        cnn.out_label = self.out_label
        torch.save(cnn, 'CNN.pkl')
        print("训练结束")


class Main:
    def __init__(self):
        self.camera = None
        self.detector = HandDetector()
        self.default_datasets = "Datasets"

    def make_datasets(self, datasets_dir="default", n=100):
        if datasets_dir == "default":
            return
        if exists(datasets_dir):
            shutil.rmtree(datasets_dir)
        mkdir(datasets_dir)
        if self.camera is None:
            self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.camera.set(3, 1280)
            self.camera.set(4, 720)
        label = input("label:")
        while not label == "":
            data = np.zeros([n, 21, 3])
            shape_list = np.zeros([n, 2], dtype=np.int16)
            hand_type = np.zeros(n, dtype=np.int8)

            zero_data = np.zeros([21, 3])
            count = 0
            cv2.startWindowThread()
            while True:
                frame, img = self.camera.read()
                img = self.detector.find_hands(img)
                result = np.zeros((21, 3))

                lm_list, bbox = self.detector.find_position(img)
                for i in range(len(lm_list)):
                    result[i] = np.array(lm_list[i])
                shape = bbox["shape"]

                if result.all() != zero_data.all():  # 假设矩阵不为0，即捕捉到手部时
                    x_1, y_1 = bbox["bbox"][0], bbox["bbox"][1]
                    data[count] = result
                    hand_type[count] = self.detector.hand_type()
                    shape_list[count] = np.array(shape)
                    count += 1
                    cv2.putText(img, str("{}/{}".format(count, n)), (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                                (0, 255, 0), 3)

                cv2.imshow("camera", img)
                key = cv2.waitKey(100)
                if cv2.getWindowProperty('camera', cv2.WND_PROP_VISIBLE) < 1:
                    break
                elif key == 27:
                    break
                elif count == n - 1:
                    break
            cv2.destroyAllWindows()

            open(datasets_dir + "/" + label + ".npz", "w")
            np.savez(datasets_dir + "/" + label + ".npz", label=label, data=data,
                     handtype=hand_type, shape=shape_list)
            label = input("label:")

    def train(self, datasets_dir="default"):
        if datasets_dir == "default":
            datasets_dir = self.default_datasets
        ai = AI(datasets_dir)
        ai.load_datasets()
        ai.train_cnn()

    def gesture_recognition(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.camera.set(3, 1280)
            self.camera.set(4, 720)
        self.detector = HandDetector()
        cnn = torch.load("CNN.pkl")
        out_label = cnn.out_label
        while True:
            frame, img = self.camera.read()
            img = self.detector.find_hands(img)
            lm_list, bbox = self.detector.find_position(img)

            if lm_list:
                x_1, y_1 = bbox["bbox"][0], bbox["bbox"][1]
                data = torch.Tensor(lm_list)
                data = data.unsqueeze(0)
                data = data.unsqueeze(0)

                test_output = cnn(data)
                result = torch.max(test_output, 1)[1].data.cpu().numpy()[0]
                cv2.putText(img, str(out_label[result]), (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                            (0, 0, 255), 3)

            cv2.imshow("camera", img)
            key = cv2.waitKey(1)
            if cv2.getWindowProperty('camera', cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == 27:
                break


if __name__ == '__main__':
    solution = Main()
    my_datasets_dir = "test"
    solution.make_datasets(my_datasets_dir, 50)
    solution.train(my_datasets_dir)
    solution.gesture_recognition()
