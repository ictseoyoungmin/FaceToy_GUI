import sys
import time

import cv2
import torch
from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel

# temp
import cvlib

# Load the face landmarks detection model
# model = torch.hub.load('pytorch/vision', 'faster_rcnn_resnet50_fpn', pretrained=True)
# model.eval()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Config Values
        # Frame Handler
        self.frame_handler = 0

        # Set up the UI
        self.setWindowTitle("Face Landmarks Detection")
        self.setGeometry(100, 100, 640, 480)

        # Set up the video capture
        self.video = cv2.VideoCapture('http://192.168.0.33:4747/video')
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

        self.label = QLabel(self)
        self.label.setGeometry(QRect(10, 60, 620, 400))
        self.label.setObjectName("label")
    
    # bounding box test
    def test_m1(self,frame):
        height, width = frame.shape[:2]
        height = height//2

        thickness = 5  # 바운딩 박스 두께
        color = (0, 255, 0)  # 바운딩 박스 색상
        cv2.rectangle(frame, (0, 0), (thickness, height), color, thickness)  # 왼쪽 모서리
        cv2.rectangle(frame, (width - thickness, 0), (width, height), color, thickness)  # 오른쪽 모서리
        cv2.rectangle(frame, (0, 0), (width, thickness), color, thickness)  # 위쪽 모서리
        cv2.rectangle(frame, (0, height - thickness), (width, height), color, thickness)  # 아래쪽 모서리

        return frame
    
    # lib test
    def test_m2(self,frame):
        faces, confidences = cvlib.detect_face(frame)
        for face in faces:
            # 바운딩 박스 좌표
            (startX, startY) = face[0], face[1]
            (endX, endY) = face[2], face[3]
            
            # 바운딩 박스 그리기
            cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
        return frame

    def method(self,frame_handler,frame):
            methods = [self.test_m1,self.test_m2]
            return methods[frame_handler](frame)
    
    def update_frame(self):
        pTime = 0
        # Read a frame from the video capture
        while True:
            ret, frame = self.video.read()

            # Read a frame and apply method
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Image Processing
                frame = self.method(self.frame_handler,frame)

                # FPS
                cTime = time.time()
                fps = 1/(cTime-pTime)
                pTime = cTime
                cv2.putText(frame,f'FPS: {int(fps)}',(40,80),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)

                image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(image)
                self.label.setPixmap(pixmap)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.video.release()
        cv2.destroyAllWindows()


    def button1_clicked(self):
        # Perform the first image processing operation
        # with torch.no_grad():
        #     outputs = model(image)
        print('btn1 clk')
        self.frame_handler = 0

        # TODO: Add code to process the outputs and display the result

    def button2_clicked(self):
        # Perform the second image processing operation
        # TODO: Add code to perform the operation and display the result
        print('btn2')
        self.frame_handler = 1

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
