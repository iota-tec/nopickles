import os
import sys

os.chdir('C:/Users/thory/PycharmProjects/chatopotamus')

import random
import cv2
import mysql.connector
import audio_mgmt, face_mgmt, nlp


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

ACCENT_DICT = {3: {0: 'en-IN-Wavenet-A', 1: 'en-IN-Wavenet-B'},
               0: {0: 'en-AU-Wavenet-C', 1: 'en-AU-Wavenet-B'},
               1: {0: 'en-AU-Wavenet-C', 1: 'en-AU-Wavenet-B'},
               2: {0: 'en-AU-Wavenet-C', 1: 'en-AU-Wavenet-B'},
               4: {0: 'en-AU-Wavenet-C', 1: 'en-AU-Wavenet-B'}}
cap = cv2.VideoCapture(0)

try:
    while True:
        while True:
            ret, frame = cap.read()

            if not ret:
                continue

            face_id, person_name, face_encoding = face_mgmt.match_face(frame=frame, cursor=customer_cursor)

            if face_encoding is None:
                continue

            if person_name:
                opening = random.choice(known_openings).format(person_name)
                intent, entities, (response, messages) = nlp.regular_customer(opening)
            else:
                opening = random.choice(known_openings).format('')
                intent, entities, (response, messages), person_name = nlp.new_customer(opening=opening,
                                                                                       face_encoding=face_encoding)

            print(messages)

            # Add a wait key and break from the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    # Release the video capture object when done
    cap.release()
    chato_customer_db.close()
    # chato_audio_db.close()
