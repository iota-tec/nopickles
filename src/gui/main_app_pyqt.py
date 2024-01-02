import os
import random

os.chdir('C:/Users/thory/PycharmProjects/chatopotamus')
import cv2
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QMainWindow
from PyQt5.QtCore import QTimer
import cv2
from PyQt5.QtGui import QImage, QPixmap
from src import face_mgmt
# from src.training_and_prediction import models, predict
import sys
import mysql.connector
import numpy as np
from tensorflow import keras

chato_customer_db = mysql.connector.connect(
    host='localhost',
    user='root',
    password='0301sonaL',
    database='chato_customer',
    auth_plugin='mysql_native_password'
)
customer_cursor = chato_customer_db.cursor()

known_openings = ['Hi {}, what can I get for you today?',
                  'Hey there {}, what are you craving for today?',
                  'Oh Hi {}, whatcha having today?',
                  'Helloooww again, tell me what can I get you {}',
                  'Oh its {}, what can I get for you dear?',
                  'Lovely timing!! We just started brewing everything fresh, what can I get you then?']

ACCENT_DICT = {3: {0: 'en-IN-Wavenet-A', 1: 'en-IN-Wavenet-B'},
               0: {0: 'en-AU-Wavenet-C', 1: 'en-AU-Wavenet-B'},
               1: {0: 'en-IN-Wavenet-A', 1: 'en-IN-Wavenet-B'},
               2: {0: 'en-AU-Wavenet-C', 1: 'en-AU-Wavenet-B'},
               4: {0: 'en-AU-Wavenet-C', 1: 'en-AU-Wavenet-B'}}
gender_dict = {0: "Male", 1: "Female"}
race_dict = {0: 'East Asian Descendent',
             1: 'South Asian Descent',
             2: 'East Asian Descent',
             3: 'South Asian Descent',
             4: 'Diverse/Mixed or Other Ethnicities'
             }

menu_data = {
    "prices": {"coffee": 1.50, "cappuccino": 2.50, "iced coffee": 2, "iced capp": 2.25, "latte": 2, "tea": 1.50,
               "hot chocolate": 2.25, "french vanilla": 2.25, "white chocolate": 2.25,
               "mocha": 2.25, "espresso": 1, "americano": 2.25, "extra shot": 0.25, "soy milk": 0.3,
               "whipped topping": 1, "dark roast": 0.20, "Turkey Bacon Club": 3, "BLT": 2.90,
               "grilled cheese": 4, "chicken wrap": 3.50, "soup": 2.80, "donut": 1.5, "double double": 1.50,
               "triple triple": 1.50, "muffin": 2.40, "bagel": 3, "timbits": 3, "panini": 2.40, "croissant": 3},
    "price multiplier": {"small": 1, "medium": 1.2, "large": 1.4, "extra large": 1.6}
}

CNN_MODEL = keras.models.load_model('resources/age_gender/saved/age_gender_best_yet.h5')


def create_menu_widget(menu_data):
    layout = QGridLayout()

    # List of beverages
    beverages = {"coffee", "cappuccino", "iced coffee", "iced capp", "latte", "tea", "hot chocolate",
                 "french vanilla", "white chocolate", "mocha", "espresso", "americano", "dark roast"}

    # Styling constants
    header_style = "font-family: Lobster; font-size: 13px; color: 888;"
    beverage_style = "font-weight: bold; font-size: 14px; color: #333; background-color: #e0e0e0;"  # Style for beverages
    non_beverage_style = "font-weight: bold; font-size: 14px; color: #333; background-color: #e0e0e0;"  # Style for non-beverages
    price_style = "font-family: Verdana; font-weight: bold; font-size: 14px; color: green; background-color: #f0f0f0;"

    # Headers
    headers = ["ITEM", "SMALL", "MEDIUM", "LARGE", "X-LARGE"]
    for col, header_text in enumerate(headers):
        header_label = QLabel(header_text)
        header_label.setStyleSheet(header_style)
        layout.addWidget(header_label, 0, col)

    row = 1
    for item, price in menu_data["prices"].items():
        item_label = QLabel(item.title())

        # Apply different styles for beverages and non-beverages
        if item in beverages:
            item_label.setStyleSheet(beverage_style)
        else:
            item_label.setStyleSheet(non_beverage_style)

        layout.addWidget(item_label, row, 0)  # Item name

        # Beverage price calculation
        if item in beverages:
            for col, size in enumerate(["small", "medium", "large", "extra large"], 1):
                size_price = price * menu_data["price multiplier"][size]
                price_label = QLabel(f"${size_price:.2f}")
                price_label.setStyleSheet(price_style)
                layout.addWidget(price_label, row, col)
        else:  # Non-beverage items
            for col in range(1, 5):
                if col == 2:  # Medium size column
                    price_label = QLabel(f"${price:.2f}")
                    price_label.setStyleSheet(price_style)
                else:
                    price_label = QLabel("-")
                    price_label.setStyleSheet(header_style)
                layout.addWidget(price_label, row, col)

        row += 1

    # Adjusting grid spacing
    layout.setHorizontalSpacing(10)
    layout.setVerticalSpacing(5)

    return layout


class CameraWidget(QWidget):
    def __init__(self, parent=None):
        super(CameraWidget, self).__init__(parent)
        self.initUI()

    def initUI(self):
        # Initialize camera and timer for updating frames
        self.camera = cv2.VideoCapture(0)  # Assuming first camera
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)  # Update interval in milliseconds

        self.image_label = QLabel(self)
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.setLayout(layout)

    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            # Process frame (draw bounding box)
            processed_frame = face_mgmt.draw_bounding_box(frame)

            # Convert to QImage and set it to image_label
            height, width, channel = processed_frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(processed_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            self.image_label.setPixmap(QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        self.camera.release()


# Main App Class
class MenuApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Menu Display')
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QHBoxLayout(central_widget)

        # Create the menu widget and add it to the layout
        menu_layout = create_menu_widget(menu_data)
        menu_widget = QWidget()
        menu_widget.setLayout(menu_layout)
        layout.addWidget(menu_widget)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QHBoxLayout(central_widget)

        left_layout = QVBoxLayout()

        # Camera widget in the upper quarter of the left half
        camera_widget = CameraWidget()
        left_layout.addWidget(camera_widget)

        # Menu widget in the lower three-quarters of the left half
        menu_layout = create_menu_widget(menu_data)
        menu_widget = QWidget()
        menu_widget.setLayout(menu_layout)
        left_layout.addWidget(menu_widget)

        layout.addLayout(left_layout)

        # Optional: Add another widget or layout to the right side if needed
        # ...

        self.show()

# Main execution
if __name__ == '__main__':
    app = QApplication([])
    ex = MenuApp()
    sys.exit(app.exec_())
