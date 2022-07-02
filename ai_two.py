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
import tkinter as tk
import shutil
import math
from scipy import stats
from os.path import exists
from os import mkdir
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset


# 旋转函数
def rotate(angle, x, y, point_x, point_y):
    px = (x - point_x) * math.cos(angle) - (y - point_y) * math.sin(angle) + point_x
    py = (x - point_x) * math.sin(angle) + (y - point_y) * math.cos(angle) + point_y
    return px, py


# 归一化
def normalize(x):
    max_x = np.max(x)
    min_x = np.min(x)
    return (x-min_x)/(max_x-min_x)


class CNNTwo(nn.Module):
    def __init__(self, m):
        super(CNNTwo, self).__init__()
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
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.med = nn.Linear(32 * 11 * 1, 500)
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

        is_two_hand = False
        if self.results.multi_hand_landmarks is not None and len(self.results.multi_hand_landmarks) >= 2:
            is_two_hand = True
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img, is_two_hand

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
                self.lmList.append([lm.x, lm.y, 0])
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
        self.re_lmList = np.array(self.re_lmList)
        if self.re_lmList.any():
            self.re_lmList = np.concatenate((np.zeros((21, 1)), self.re_lmList), axis=1)
            self.re_lmList = np.concatenate((self.re_lmList, np.zeros((1, 4))), axis=0)

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
                px, py = rotate(theta, i[0] * w, i[1] * h, point_x * w, point_y * h)
                self.re_lmList.append([px, py, 0])
                if draw:
                    cv2.circle(img, (int(px), int(py)), 5, (0, 0, 255), cv2.FILLED)
            # 归一化
            x_array = normalize(np.array(self.re_lmList)[:, 0])
            # print(x_array)
            for i in range(len(x_array)):
                self.re_lmList[i][0] = x_array[i]
            y_array = normalize(np.array(self.re_lmList)[:, 1])
            for i in range(len(y_array)):
                self.re_lmList[i][1] = x_array[i]
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
        cnn = CNNTwo(self.m).to(self.DEVICE)
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
                if (step + 1) % 50 == 0:  # 输出结果
                    print(
                        "\r[Epoch: %d] [%d/%d (%0.f %%)][Loss: %f]"
                        % (
                            epoch + 1,
                            (step + 1) * len(data),
                            len(self.train_loader.dataset),
                            100. * (step + 1) / len(self.train_loader),
                            loss.item()
                        ), end="")

        cnn.out_label = self.out_label
        torch.save(cnn, 'CNN_two.pkl')
        print("训练结束")


class Main:
    def __init__(self):
        self.camera = None
        self.detector = HandDetector()
        self.default_datasets = "Datasets"
        self.len_x = 44
        self.len_y = 4
        self.label = ''

        self.result = []
        self.disp = ""

    def change_state(self):
        self.label = self.entry.get()  # 调用get()方法，将Entry中的内容获取出来
        self.top1.quit()
        if self.label == "":
            self.top1.destroy()

    def on_closing(self):
        self.label = ""
        self.top1.destroy()

    def make_datasets(self, camera, datasets_dir="default", n=100):
        if datasets_dir == "default":
            return
        if exists(datasets_dir):
            shutil.rmtree(datasets_dir)
        mkdir(datasets_dir)
        self.camera = camera

        self.top1 = tk.Tk()
        self.top1.geometry('300x50')
        self.top1.title('请输入标签')
        self.top1.protocol("WM_DELETE_WINDOW", self.on_closing)
        tk.Label(self.top1, text='Label:').place(x=27, y=10)
        self.entry = tk.Entry(self.top1, width=15)
        self.entry.place(x=80, y=10)
        tk.Button(self.top1, text='确定', command=self.change_state).place(x=235, y=5)

        self.top1.mainloop()
        while not self.label == "":
            data = np.zeros([n, self.len_x, self.len_y])
            shape_list = np.zeros([n, 2], dtype=np.int16)
            hand_type = np.zeros(n, dtype=np.int8)

            count = 0
            cv2.startWindowThread()
            while True:
                frame, img = self.camera.read()
                img, is_two_hand = self.detector.find_hands(img)
                result = np.zeros((self.len_x, self.len_y))

                if is_two_hand:
                    lm_list1, bbox1 = self.detector.find_position(img, 0)
                    lm_list2, bbox2 = self.detector.find_position(img, 1)
                    for i in range(len(lm_list1)):
                        result[i] = np.array(lm_list1[i])
                    for i in range(len(lm_list1), len(lm_list1)+len(lm_list2)):
                        result[i] = np.array(lm_list2[i-len(lm_list1)])
                    if result.sum() > 0:  # 假设矩阵不为0，即捕捉到手部时

                        shape1 = bbox1["shape"]
                        x_1, y_1 = bbox1["bbox"][0], bbox1["bbox"][1]
                        shape2 = bbox2["shape"]
                        x_2, y_2 = bbox2["bbox"][0], bbox2["bbox"][1]
                        data[count] = result
                        hand_type[count] = self.detector.hand_type()
                        shape_list[count] = np.array(shape1)
                        count += 1
                        cv2.putText(img, str("{}/{}".format(count, n)), (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                                    (0, 255, 0), 3)
                        cv2.putText(img, str("{}/{}".format(count, n)), (x_2, y_2), cv2.FONT_HERSHEY_PLAIN, 3,
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

            open(datasets_dir + "/" + self.label + ".npz", "w")
            np.savez(datasets_dir + "/" + self.label + ".npz", label=self.label, data=data,
                     handtype=hand_type, shape=shape_list)

            self.top1.mainloop()

    def train(self, datasets_dir="default"):
        if datasets_dir == "default":
            datasets_dir = self.default_datasets
        ai = AI(datasets_dir)
        ai.load_datasets()
        ai.train_cnn()

    def gesture_recognition(self, detector, img, cnn):
        self.detector = detector
        out_label = cnn.out_label
        img, is_two_hand = self.detector.find_hands(img)
        if is_two_hand:
            lm_list1, bbox1 = self.detector.find_position(img, 0)
            lm_list2, bbox2 = self.detector.find_position(img, 1)
            if lm_list1.any() and lm_list2.any():
                x_1, y_1 = bbox1["bbox"][0], bbox1["bbox"][1]
                x_2, y_2 = bbox2["bbox"][0], bbox2["bbox"][1]
                lm_list = np.concatenate((lm_list1, lm_list2), axis=0)
                data = torch.Tensor(lm_list)
                data = data.unsqueeze(0)
                data = data.unsqueeze(0)

                test_output = cnn(data)
                self.result.append(torch.max(test_output, 1)[1].data.cpu().numpy()[0])
                if len(self.result) > 4:
                    self.disp = str(out_label[stats.mode(self.result)[0][0]])
                    self.result = []

                cv2.putText(img, self.disp, (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                            (0, 0, 255), 3)
                cv2.putText(img, self.disp, (x_2, y_2), cv2.FONT_HERSHEY_PLAIN, 3,
                            (0, 0, 255), 3)
        else:
            return 1
        return 0


if __name__ == '__main__':
    solution = Main()
    my_datasets_dir = "test-two"
    solution.make_datasets(my_datasets_dir, 100)
    solution.train(my_datasets_dir)
    solution.gesture_recognition()
