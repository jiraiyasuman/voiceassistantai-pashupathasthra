import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QTextBrowser
from PyQt5.QtGui import QMovie, QFont
from PyQt5.QtCore import Qt, QThread, QTimer, QTime, QDate
from PyQt5 import QtWidgets
import os

# Backend voice assistant thread
class VoiceAssistantThread(QThread):
    def run(self):
        print("Voice thread started")
        from pashupathastra import main_voice_assistant
        main_voice_assistant()

class PashupathastraUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Pashupathashtra AI")
        self.setGeometry(100, 100, 1024, 768)
        self.setStyleSheet("background-color: black;")
        self.setWindowFlags(Qt.FramelessWindowHint)

        # === Background GIF ===
        self.label_background = QLabel(self)
        self.label_background.setGeometry(0, 0, 1024, 768)
        self.label_background.setAlignment(Qt.AlignCenter)
        gif_path = "Leonardo_Anime_XL_lord_shivas_holographic_face_with_3D_effects_3.gif"
        if not os.path.exists(gif_path):
            print("GIF not found. Please check path.")
            sys.exit()
        self.movie = QMovie(gif_path)
        self.label_background.setMovie(self.movie)
        self.movie.start()

        # === Real-Time Clock (Top-Left) ===
        self.text_time = QTextBrowser(self)
        self.text_time.setGeometry(20, 20, 300, 40)
        self.text_time.setFont(QFont("Consolas", 14))
        self.text_time.setStyleSheet("""
            background: transparent;
            color: cyan;
            border: none;
        """)

        # === Real-Time Date (Top-Right) ===
        self.text_date = QTextBrowser(self)
        self.text_date.setGeometry(704, 20, 300, 40)
        self.text_date.setFont(QFont("Consolas", 14))
        self.text_date.setStyleSheet("""
            background: transparent;
            color: lightgreen;
            border: none;
        """)

        # Timer for updating time and date
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_datetime)
        self.timer.start(1000)  # Update every second
        self.update_datetime()  # Immediate update

        # Start backend voice assistant
        self.voice_thread = VoiceAssistantThread()
        self.voice_thread.start()

    def update_datetime(self):
        current_time = QTime.currentTime().toString("hh:mm:ss AP")
        current_date = QDate.currentDate().toString("dddd, MMMM d, yyyy")
        self.text_time.setText(f" {current_time}")
        self.text_date.setText(f" {current_date}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PashupathastraUI()
    window.show()
    sys.exit(app.exec_())
