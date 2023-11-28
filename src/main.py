import os

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
                  'Hey again {}, what are you craving for today?',
                  'Oh Hi {}, whatcha having today?',
                  'Helloooww again, tell me what can I get you {}']

while True:
    face_id, person_name, face_encoding = face_mgmt.match_face(cursor=customer_cursor)

    if person_name:
        opening = random.choice(known_openings).format(person_name)
        intent, entities, (response, messages) = nlp.regular_customer(opening)

    else:
        opening = random.choice(known_openings).format('')
        intent, entities, (response, messages), person_name = nlp.new_customer(opening=opening, face_encoding=face_encoding)

    print(messages)

chato_customer_db.close()
# chato_audio_db.close()
