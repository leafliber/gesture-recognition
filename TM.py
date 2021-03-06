# -*- coding:utf-8 -*-

import cv2
import mediapipe as mp
import numpy as np


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
        self.modelComplex = False
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
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for _, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                px, py = int(lm.x * w), int(lm.y * h)
                x_list.append(px)
                y_list.append(py)
                self.lmList.append(np.array([px, py]))
                if draw:
                    cv2.circle(img, (px, py), 5, (255, 0, 255), cv2.FILLED)
            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)
            box_w, box_h = x_max - x_min, y_max - y_min
            bbox = x_min, y_min, box_w, box_h
            cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)
            bbox_info = {"id": hand_no, "bbox": bbox, "center": (cx, cy)}

            if draw:
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                              (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                              (0, 255, 0), 2)

        return self.lmList, bbox_info

    def fingers_up(self):
        """
        查找列表中打开并返回的手指数。会分别考虑左手和右手
        :return: 竖起手指的列表
        """
        fingers = []
        if self.results.multi_hand_landmarks:
            my_hand_type = self.hand_type()
            # Thumb
            if my_hand_type == "Right":
                if self.lmList[self.tipIds[0]][0] > self.lmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if self.lmList[self.tipIds[0]][0] < self.lmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            # 4 Fingers
            for i in range(1, 5):
                # if self.lmList[self.tipIds[i]][1] < self.lmList[self.tipIds[i] - 2][1]:
                if np.dot(self.lmList[self.tipIds[i]-2]-self.lmList[self.tipIds[i]-3],
                          self.lmList[self.tipIds[i]-1]-self.lmList[self.tipIds[i]-2]) >= 0:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def hand_type(self):
        """
        检查传入的手部是左还是右
        :return: "Right" 或 "Left"
        """
        if self.results.multi_hand_landmarks:
            if self.lmList[17][0] < self.lmList[5][0]:
                return "Right"
            else:
                return "Left"


class Main:
    def __init__(self):
        self.detector = None
        self.camera = None
        # self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # self.camera.set(3, 1280)
        # self.camera.set(4, 720)

    def gesture_recognition(self, img, detector):
        self.detector = detector
        img = self.detector.find_hands(img)
        lm_list, bbox = self.detector.find_position(img)

        if lm_list:
            x_1, y_1 = bbox["bbox"][0], bbox["bbox"][1]
            x1, x2, x3, x4, x5 = self.detector.fingers_up()
            if (np.linalg.norm(lm_list[4]-lm_list[8]) < 50) and (np.linalg.norm(lm_list[4]-lm_list[12]) < 50):
                cv2.putText(img, "7_SEVEN", (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                            (0, 0, 255), 3)
            elif (x2 == 1 and x3 == 1) and (x4 == 0 and x5 == 0 and x1 == 0):
                cv2.putText(img, "2_TWO", (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                            (0, 0, 255), 3)
            elif (x2 == 1 and x3 == 1 and x4 == 1) and (x1 == 0 and x5 == 0):
                cv2.putText(img, "3_THREE", (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                            (0, 0, 255), 3)
            elif (x2 == 1 and x3 == 1 and x4 == 1 and x5 == 1) and (x1 == 0):
                cv2.putText(img, "4_FOUR", (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                            (0, 0, 255), 3)
            elif x1 == 1 and x2 == 1 and x3 == 1 and x4 == 1 and x5 == 1:
                cv2.putText(img, "5_FIVE", (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                            (0, 0, 255), 3)
            elif (x2 == 1 and x1 == 0) and (x3 == 0 and x4 == 0 and x5 == 0):
                cv2.putText(img, "1_ONE", (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                            (0, 0, 255), 3)
            elif (x1 == 1 and x2 == 1) and (x3 == 0 and x4 == 0 and x5 == 0):
                cv2.putText(img, "8_EIGHT", (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                            (0, 0, 255), 3)
            elif (x1 == 1 and x5 == 1) and (x3 == 0 and x4 == 0 and x2 == 0):
                cv2.putText(img, "6_SIX", (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                            (0, 0, 255), 3)
            elif x1 == 0 and x5 == 0 and x3 == 0 and x4 == 0 and x2 == 0:
                cv2.putText(img, "0_ZERO", (x_1, y_1), cv2.FONT_HERSHEY_PLAIN, 3,
                            (0, 0, 255), 3)
            else:
                return 1
        return 0


if __name__ == '__main__':
    Solution = Main()
    Solution.gesture_recognition()
