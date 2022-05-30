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
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.detection_con, self.min_track_con)
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
        onedata = np.zeros([21,3])
        zerodata = np.zeros([21,3])
        h, w, c = img.shape
        self.lmList = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for i, lm in enumerate(my_hand.landmark):
                onedata[i] = np.array([lm.x,lm.y,lm.z])     #将三维坐标添加到单次截屏的数据中

                px, py= int(lm.x * w), int(lm.y * h)
                x_list.append(px)
                y_list.append(py)
                self.lmList.append([px, py])
                if draw:
                    cv2.circle(img, (px, py), 5, (255, 0, 255), cv2.FILLED)

        return onedata, (h, w)

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
                if self.lmList[self.tipIds[i]][1] < self.lmList[self.tipIds[i] - 2][1]:
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
                return 1
            else:
                return 0


class Main:
    def __init__(self, label, N = 100):
        self.detector = None
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.camera.set(3, 1280)
        self.camera.set(4, 720)
        self.N = N
        #初始化数据包
        self.label = label
        self.data = np.zeros([N,21,3])
        self.shape = np.zeros([N,2], dtype = np.int16)
        self.handtype = np.zeros(N, dtype = np.int8)

    def gesture_recognition(self):
        self.detector = HandDetector()
        #初始化数据
        
        zerodata = np.zeros([21,3])
        rezult = np.zeros([21,3])
        count = 0

        while True:
            frame, img = self.camera.read()
            img = self.detector.find_hands(img)
            
            rezult,shape = self.detector.find_position(img)
            if rezult.all() != zerodata.all():              #假设矩阵不为0，即捕捉到手部时
                self.data[count] = rezult
                self.handtype[count] = self.detector.hand_type()
                self.shape[count] = np.array(shape)
                count += 1

            cv2.imshow("camera", img)
            key = cv2.waitKey(1)
            if cv2.getWindowProperty('camera', cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == 27:
                break
            elif count == self.N - 1:
                break
        
        np.savez('firstdata', label = self.label, data = self.data, 
                                handtype = self.handtype, shape = self.shape)


if __name__ == '__main__':
    Solution = Main(label = "five")
    Solution.gesture_recognition()
    npzfile = np.load('firstdata.npz')
    
    #print(npzfile['data'][0])
    #print(" ")
    #print(npzfile['handtype'])
    #print(npzfile['label'])
    #print(npzfile['shape'])
