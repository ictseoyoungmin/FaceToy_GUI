import sys
import time

import cv2
import torch
from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel

# temp
import numpy as np

import mediapipe as mp

# Load the face mesh model
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=10,min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(thickness=1,circle_radius=2)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Config Values
        # Frame Handler 
        self.frame_handler = 2

        # Set up the UI
        self.setWindowTitle("Face Landmarks Detection")
        self.setGeometry(100, 100, 640, 480)

        # Set up the video capture
        self.video = cv2.VideoCapture('http://192.168.0.33:4747/video')
        # self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Set up the timer to update the video feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)

        # Set up the buttons
        self.button1 = QPushButton("Button 1", self)
        self.button1.setGeometry(10, 10, 100, 30)
        self.button1.clicked.connect(self.button1_clicked)

        self.button2 = QPushButton("Button 2", self)
        self.button2.setGeometry(120, 10, 100, 30)
        self.button2.clicked.connect(self.button2_clicked)

        self.button3 = QPushButton("FaceMesh", self)
        self.button3.setGeometry(220, 10, 100, 30)
        self.button3.clicked.connect(self.button3_clicked)

        self.label = QLabel(self)
        self.label.setGeometry(QRect(10, 60, 620, 400))
        self.label.setObjectName("label")
    
    # bounding box test
    @staticmethod
    def test_m1(frame):
        height, width = 100,20

        thickness = 5  # 바운딩 박스 두께
        color = (0, 255, 0)  # 바운딩 박스 색상
       
        cv2.rectangle(frame, (100, 180+height - thickness), (width, height), color, thickness)  # 아래쪽 모서리
        return frame
    
    
    # lib test
    @staticmethod
    def test_m2(frame):
        # faces, confidences = cvlib.detect_face(frame)
        # print(np.shape(faces))
        # for face in faces:
        #     # 바운딩 박스 좌표
        #     (startX, startY) = face[0], face[1]
        #     (endX, endY) = face[2], face[3]
            
        #     # 바운딩 박스 그리기
        #     cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
        return frame

    @staticmethod
    def test_m3(frame):
        re = faceMesh.process(frame)
        frame = np.zeros_like(frame)
        if re.multi_face_landmarks:
            for faceLms in re.multi_face_landmarks:
                mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_TESSELATION,
                                      drawSpec,drawSpec)
                # print(faceLms)
        return frame

    def processing(self,frame):
        methods = [self.test_m1,self.test_m2,self.test_m3]
        return methods[self.frame_handler](frame)
    
    @staticmethod
    def plot_fps(frame,pTime,loc=(20,80),font_face=cv2.FONT_HERSHEY_PLAIN,font_scale=1,color=(255,0,0),thinkness=2):
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(frame,f'FPS: {int(fps)}',loc,font_face,font_scale,color,thinkness)
        return frame,pTime
    
    def update_frame(self):
        pTime = 0
        # Read a frame from the video capture
        while True:
            ret, frame = self.video.read()

            # frame = cv2.imread('./images.jpg')
            # ret = 1
            # Read a frame and apply method
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Image Processing
                frame = self.processing(frame)

                # FPS
                frame, pTime = self.plot_fps(frame,pTime)

                image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(image)
                self.label.setPixmap(pixmap)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.video.release()
        cv2.destroyAllWindows()


    def button1_clicked(self):
        print('btn1 clk')
        self.frame_handler = 0

    def button2_clicked(self):
        print('btn2')
        self.frame_handler = 1

    def button3_clicked(self):
        print('btn3')
        self.frame_handler = 2

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
