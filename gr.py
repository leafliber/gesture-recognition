import TM
import ai
import ai_two
import cv2
import copy
import torch
import torch.nn as nn


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
            nn.MaxPool2d(2),
        )
        self.med = nn.Linear(32 * 11 * 2, 500)
        self.med2 = nn.Linear(1 * 21 * 3, 100)
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


class Main:
    def __init__(self):
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.camera.set(3, 1280)
        self.camera.set(4, 720)

        self.tm_detector = TM.HandDetector()
        self.ai_detector = ai.HandDetector()
        self.at_detector = ai_two.HandDetector()

        self.tm_main = TM.Main()
        self.ai_main = ai.Main()
        self.at_main = ai_two.Main()

    def gr_img(self, filedir, diy):
        print(filedir)
        if diy:
            cnn = torch.load("CNN.pkl")
            cnn_two = torch.load("CNN_two.pkl")
        while True:
            not_match = 0
            img = cv2.imread(filedir)
            img_tm = copy.deepcopy(img)
            is_one_hand = self.at_main.gesture_recognition(self.at_detector, img, cnn_two)
            if is_one_hand:
                not_match = self.ai_main.gesture_recognition_camera(self.ai_detector, img, cnn)
                if not_match:
                    self.tm_main.gesture_recognition(img_tm, self.tm_detector)

            if not_match:
                cv2.imshow("camera", img_tm)
            else:
                cv2.imshow("camera", img)
            key = cv2.waitKey(1)
            if cv2.getWindowProperty('camera', cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == 27:
                break

    def gr_video(self, filedir, diy):
        cap = cv2.VideoCapture(filedir)
        if diy:
            cnn = torch.load("CNN.pkl")
            cnn_two = torch.load("CNN_two.pkl")
        while True:
            ret, img = cap.read()
            not_match = 0
            img_tm = copy.deepcopy(img)
            is_one_hand = self.at_main.gesture_recognition(self.at_detector, img, cnn_two)
            if is_one_hand:
                not_match = self.ai_main.gesture_recognition_camera(self.ai_detector, img, cnn)
                if not_match:
                    self.tm_main.gesture_recognition(img_tm, self.tm_detector)

            if not_match:
                cv2.imshow("camera", img_tm)
            else:
                cv2.imshow("camera", img)
            key = cv2.waitKey(1)
            if cv2.getWindowProperty('camera', cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == 27:
                break
        cap.release()

    def gr_realtime(self, diy):
        if diy:
            cnn = torch.load("CNN.pkl")
            cnn_two = torch.load("CNN_two.pkl")
        while True:
            frame, img = self.camera.read()
            not_match = 0
            img_tm = copy.deepcopy(img)
            is_one_hand = self.at_main.gesture_recognition(self.at_detector, img, cnn_two)
            if is_one_hand:
                not_match = self.ai_main.gesture_recognition_camera(self.ai_detector, img, cnn)
                if not_match:
                    self.tm_main.gesture_recognition(img_tm, self.tm_detector)

            if not_match:
                cv2.imshow("camera", img_tm)
            else:
                cv2.imshow("camera", img)
            key = cv2.waitKey(1)
            if cv2.getWindowProperty('camera', cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == 27:
                break

    def ai_input(self):
        self.ai_main.make_datasets(self.camera, "ai_datasets", 100)
        self.ai_main.train("ai_datasets")
        self.at_main.make_datasets(self.camera, "ai_two_datasets", 100)
        self.at_main.train("ai_two_datasets")


if __name__ == '__main__':
    main = Main()
    main.gr_img("", 0)
