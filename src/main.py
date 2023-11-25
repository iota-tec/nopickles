import mysql.connector
import audio_mgmt, face_mgmt, nlp

chato_audio_db = mysql.connector.connect(
    host='localhost',
    user='root',
    password='0301sonaL',
    database='chato_audio',
    auth_plugin='mysql_native_password'
)
audio_cursor = chato_audio_db.cursor()

chato_customer_db = mysql.connector.connect(
    host='localhost',
    user='root',
    password='0301sonaL',
    database='chato_customer',
    auth_plugin='mysql_native_password'
)
customer_cursor = chato_customer_db.cursor()

face_id, person_name = face_mgmt.match_face(cursor=customer_cursor)

print(person_name)
if person_name:
    audio_mgmt.speak(f'Hi {person_name}, what can I get for you today?')
    request = audio_mgmt.speech_to_text()  # Customer spoke this
else:
    audio_mgmt.speak(f'Hi, how can I help you today?')
    request = audio_mgmt.speech_to_text()

# NLP working here...

audio_mgmt.speak(request)

chato_customer_db.close()
chato_audio_db.close()