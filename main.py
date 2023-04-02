import sys
import cv2
import torch
from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel

# Load the face landmarks detection model
# model = torch.hub.load('pytorch/vision', 'faster_rcnn_resnet50_fpn', pretrained=True)
# model.eval()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

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

    def update_frame(self):
        # Read a frame from the video capture
        while True:
            ret, frame = self.video.read()

            # Convert the frame to a QImage
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        # TODO: Add code to process the outputs and display the result

    def button2_clicked(self):
        # Perform the second image processing operation
        # TODO: Add code to perform the operation and display the result
        pass
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
