import sys
import cv2

from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel

# func.py
from func import *

WIDTH = 680
HEIGHT = 480

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Config Values
        self.handler = 2
        self.pTime = 0

        # Set up the UI
        self.setWindowTitle("Face Landmarks Detection")
        self.setGeometry(100, 100, WIDTH, HEIGHT)

        # Set up the video capture
        # self.video = cv2.VideoCapture('http://192.168.0.33:4747/video')
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

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
        self.label.setGeometry(QRect(10, 60, WIDTH-20, HEIGHT-80))
        self.label.setObjectName("label")
    
    def update_frame(self):
        ret, frame = self.video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = processing(frame,self.handler)
            frame, self.pTime = plot_fps(frame,self.pTime)

            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.label.setPixmap(pixmap)

    def button1_clicked(self):
        print('btn1')
        self.handler = 0

    def button2_clicked(self):
        print('btn2')
        self.handler = 1

    def button3_clicked(self):
        print('btn3')
        self.handler = 2

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
