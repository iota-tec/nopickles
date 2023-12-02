import random
import sys
import os
import tkinter as tk
from tkinter import Label, Text, Scrollbar, END, Frame
import cv2
import threading
from PIL import Image, ImageTk

os.chdir('C:/Users/thory/PycharmProjects/chatopotamus')

import mysql.connector
# from PyQt5.QtCore import QTimer, QThread, pyqtSignal
# from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
# from PyQt5.QtGui import QPixmap, QImage
# import cv2
from src import face_mgmt, nlp

chato_customer_db = mysql.connector.connect(
    host='localhost',
    user='root',
    password='0301sonaL',
    database='chato_customer',
    auth_plugin='mysql_native_password'
)
customer_cursor = chato_customer_db.cursor()

person_name = None
known_openings = ['Hi {}, what can I get for you today?',
                  'Hey there {}, what are you craving for today?',
                  'Oh Hi {}, whatcha having today?',
                  'Helloooww again, tell me what can I get you {}',
                  'Oh its {}, what can I get for you dear?',
                  'Lovely timing!! We just started brewing everything fresh, what can I get you then?']


def resize_image(frame, max_size):
    height, width = frame.shape[:2]
    if height > max_size[1] or width > max_size[0]:
        # Calculate the ratio of height and width and resize accordingly
        scaling_factor = min(max_size[0] / width, max_size[1] / height)
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
    return frame

def update_image(label, cap):
    ret, frame = cap.read()
    if ret:
        # Draw bounding boxes on the frame
        frame_with_boxes = face_mgmt.draw_bounding_boxes(frame=frame)

        # Resize the frame to fit the display area
        resized_frame = resize_image(frame_with_boxes, (400, 300))  # Adjust (400, 300) as needed

        # Convert the frame to the correct format and display it
        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(frame_rgb)
        img = ImageTk.PhotoImage(image=im)

        label.config(image=img)
        label.image = img

    label.after(10, update_image, label, cap)


def face_detection_routine(cap, callback):
    while True:
        ret, frame = cap.read()
        if ret:
            # Perform face detection
            face_id, person_name, face_encoding = face_mgmt.match_face(frame=frame, cursor=customer_cursor)
            # Update GUI based on detection
            callback(face_id, person_name, face_encoding)


def on_face_detected(face_id, person_name, face_encoding):
    if person_name:
        opening = random.choice(known_openings).format(person_name)
        intent, entities, (response, messages) = nlp.regular_customer(opening)
    else:
        opening = random.choice(known_openings).format('')
        intent, entities, (response, messages), person_name = nlp.new_customer(opening=opening,
                                                                               face_encoding=face_encoding)

class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, text):
        self.text_widget.insert(END, text)
        self.text_widget.see(END)  # Scroll to the bottom

    def flush(self):
        pass

def main():
    root = tk.Tk()
    root.title("Chatopotamus")
    root.geometry("800x600")

    cap = cv2.VideoCapture(0)

    # Create frames for menu and camera/output
    left_frame = Frame(root, width=400)
    right_frame = Frame(root, width=400)
    left_frame.pack(side='left', fill='y')
    right_frame.pack(side='right', fill='both', expand=True)

    menu = {
        "coffee": 1.50, "cappuccino": 2.50, "iced coffee": 2, "iced capp": 2.25, "latte": 2, "tea": 1.50,
        "hot chocolate": 2.25, "french vanilla": 2.25, "white chocolate": 2.25, "mocha": 2.25, "espresso": 1,
        "americano": 2.25, "extra shot": 0.25, "soy milk": 0.3, "whipped topping": 1, "dark roast": 0.20,
        "Turkey Bacon Club": 3, "BLT": 2.90, "grilled cheese": 4, "chicken wrap": 3.50, "soup": 2.80,
        "donut": 1.5, "double double": 1.50, "triple triple": 1.50, "muffin": 2.40, "bagel": 3, "timbits": 3,
        "panini": 2.40, "croissant": 3
    }

    price_multiplier = {
        "small": 1, "medium": 1.2, "large": 1.4, "extra large": 1.6
    }

    # Items available in multiple sizes
    multi_size_items = ["coffee", "cappuccino", "iced coffee", "latte", "tea", "hot chocolate", "french vanilla",
                        "white chocolate", "mocha", "espresso", "americano"]

    # Formatting the menu
    menu_text = "Menu\n-----\n"
    menu_text += "Item\t\tSmall\tMedium\tLarge\tExtra Large\n"
    menu_text += "---------------------------------------------\n"

    for item in menu:
        if item in multi_size_items:
            menu_text += f"{item.title()}\t"
            for size in price_multiplier:
                price = menu[item] * price_multiplier[size]
                menu_text += f"${price:.2f}\t"
            menu_text += "\n"
        else:
            menu_text += f"{item.title()}\t\t${menu[item]:.2f}\n"

    menu_text = menu_text.title()

    menu_label = Label(left_frame, text=menu_text, justify='left', anchor='nw')
    menu_label.pack(anchor='nw', padx=20, pady=20)

    # Create the label for displaying the image in the right frame
    label = Label(right_frame)
    label.pack(anchor='ne', padx=20, pady=20)

    # Create the text widget with scrollbar for displaying output in the right frame
    text_frame = Frame(right_frame)
    text = Text(text_frame, height=10, wrap='word')
    scrollbar = Scrollbar(text_frame, command=text.yview)
    text.configure(yscrollcommand=scrollbar.set)

    text.pack(side='left', fill='both', expand=True)
    scrollbar.pack(side='right', fill='y')
    text_frame.pack(fill='both', expand=True, padx=20, pady=20)

    # Redirect standard output
    sys.stdout = StdoutRedirector(text)

    # Start the routine to update the image
    update_image(label, cap)

    # Start the face detection in a separate thread
    threading.Thread(target=face_detection_routine, args=(cap, on_face_detected), daemon=True).start()

    root.mainloop()
    cap.release()

if __name__ == "__main__":
    main()
