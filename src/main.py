import os
import sys

import keras.models

os.chdir('C:/Users/thory/PycharmProjects/chatopotamus')

import random
import cv2
import mysql.connector
import audio_mgmt, face_mgmt, nlp
from training_and_prediction import models, predict

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
cap = cv2.VideoCapture(0)

CNN_MODEL = keras.models.load_model('resources/age_gender/saved/age_gender_best_yet.h5')

try:
    while True:  # Main loop for watching faces
        new_face = True
        person_name = None

        ret, frame = cap.read()
        if not ret:
            continue

        face_id, person_name, face_encoding = face_mgmt.match_face(frame=frame, cursor=customer_cursor)
        if face_encoding is None:
            continue

        # Predict age, gender, and ethnicity for each new face
        if new_face:
            age, gender, race = predict.predict_age_gender_race(frame, CNN_MODEL)
            accent = ACCENT_DICT[race][gender]
            print(f'Age: {abs(int(age))}, Gender: {gender_dict[gender]}, Ethnicity:{race_dict[race]}')
            new_face = False

        messages = []
        intents = []
        entity_tags = []
        total_price = 0
        continue_interaction = True

        opening = random.choice(known_openings).format(person_name if person_name else '')
        audio_mgmt.speak(opening, accent=accent)

        # Interaction loop
        while continue_interaction:

            customer_request = audio_mgmt.speech_to_text()

            if not customer_request:
                # Handling case where no input is received
                continue_interaction = False
                break

            if person_name:
                intents, entity_tags, messages, total_price, continue_interaction = \
                    nlp.regular_customer(customer_request, accent, messages, intents, entity_tags, total_price)
            else:
                intents, entity_tags, messages, total_price, continue_interaction, person_name = \
                    nlp.new_customer(customer_request, face_encoding, accent, customer_cursor, messages, intents, entity_tags, total_price)

            # Check if interaction should continue
            if not continue_interaction:
                audio_mgmt.speak('Visit again, Bye!', accent=accent)
                break  # End of one customer interaction

        print(messages)

finally:
    # Release the video capture object and close database connection when done
    cap.release()
    chato_customer_db.close()
