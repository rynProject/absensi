import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox
import pickle
import tkinter as tk
from tkinter import messagebox

class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Face Recognition'
        self.left = 10
        self.top = 10
        self.width = 320
        self.height = 200
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        button = QPushButton('Start', self)
        button1 = QPushButton('Cancel', self)
        button.move(75, 70)
        button1.move(150, 70)
        button.clicked.connect(self.on_click)
        button1.clicked.connect(QCoreApplication.instance().quit)
        self.show()

    def confirmDialog(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        buttonReply = QMessageBox.question(self, 'Konfirmasi', "Sudah Benar?",
                                           QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if buttonReply == QMessageBox.Yes:
            print('Yes clicked.')
        else:
            print('No clicked.')

        self.show()

    @pyqtSlot()
    def on_click(self):
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("train.yml")
        labels = {"person_name": 1}
        with open("labels.pickle", "rb") as f:
            labels = pickle.load(f)
            labels = {v: k for k, v in labels.items()}

        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
            for (x, y, w, h) in faces:
                print(x, w, y, h)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                id_, conf = recognizer.predict(roi_gray)
                if conf > 60:
                    font = cv2.QT_FONT_NORMAL
                    id = 0
                    id += 1
                    name = labels[id_]
                    color = (255, 0, 0)
                    stroke = 2
                    cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), (2))
                    self.confirmDialog()
                    
                else:
                    color = (255, 0, 0)
                    stroke = 2
                    font = cv2.QT_FONT_NORMAL
                    cv2.putText(cap, "UNKNOWN", (x, y), font, 1, color, stroke, cv2.LINE_AA)
                    cv2.rectangle(cap, (x, y), (x + w, y + h), (255, 0, 0), (2))
            cv2.imshow('Face Recognition', frame)
            k = cv2.waitKey(20) & 0xff
            if k == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())