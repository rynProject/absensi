import sys
import cv2
import mysql.connector
from mysql.connector import Error
from datetime import datetime
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox
import pickle

# create database connection
try:
    myconn = mysql.connector.connect(host='localhost',
                                         port = '8080',
                                         database='facerecognition',
                                         user='root',
                                         password='')
    if myconn.is_connected():
        db_Info = myconn.get_server_info()
        print("Connected to MySQL Server version ", db_Info)
        cursor = myconn.cursor()
        cursor.execute("select database();")
        record = cursor.fetchone()
        print("You're connected to database: ", record)

except Error as e:
    print("Error while connecting to MySQL", e)
finally:
    if (myconn.is_connected()):
        cursor.close()
        myconn.close()
        print("MySQL connection is closed")

#create time for attendance using current time
date = datetime.utcnow()
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
cursor = myconn.cursor()

class App(QWidget):
    # method pembuatan frame utama
    def __init__(self):
        super().__init__()
        self.title = 'Absensi'
        self.left = 10
        self.top = 10
        self.width = 320
        self.height = 200
        self.initUI()

    # method pembuatan User Interface
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        button = QPushButton('Absen', self)
        button1 = QPushButton('Cancel', self)
        button.move(75, 70)
        button1.move(150, 70)
        button.clicked.connect(self.on_click)
        button1.clicked.connect(QCoreApplication.instance().quit)
        self.show()

    @pyqtSlot()
    # method click button
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

                # jika tingkat kemiripan lebih dari 60% maka tentukan nama
                if conf > 60:
                    font = cv2.QT_FONT_NORMAL
                    id = 0
                    id += 1
                    name = labels[id_]
                    color = (255, 0, 0)
                    stroke = 2
                    cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), (2))

                    # Close Camera
                    cap.release()
                    cv2.destroyAllWindows()

                    # Konfirmasi nama
                    buttonReply = QMessageBox.information(self, 'Konfirmasi', "Anda Akan Absen Dengan Nama : "+name+" Anda Yakin?",
                                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                    if buttonReply == QMessageBox.Yes:
                        insert = "INSERT INTO absen (nama, waktu_absen, tanggal) VALUES (%s, %s, %s)"
                        val = (name, current_time, date)
                        cursor.execute(insert, val)
                        myconn.commit()
                        QMessageBox.information(self, 'Sukses', "Anda Sukses Absen Dengan Nama : "+name+".",
                                                       QMessageBox.Ok | QMessageBox.Ok)
                    else:
                        QMessageBox.information(self, "Absensi Dibatalkan", QMessageBox.Ok | QMessageBox.Ok)

                #jika tingkat kemiripan dibawah 60% nama = UNKNOWN(tidak dikenali)
                else:
                    color = (255, 0, 0)
                    stroke = 2
                    font = cv2.QT_FONT_NORMAL
                    cv2.putText(cap, "UNKNOWN", (x, y), font, 1, color, stroke, cv2.LINE_AA)
                    cv2.rectangle(cap, (x, y), (x + w, y + h), (255, 0, 0), (2))
            # title
            cv2.imshow('Absensi Face Recognition', frame)

            # jika kamera tidak mendeteksi wajah tutup face recognition
            k = cv2.waitKey(20) & 0xff
            if k == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())