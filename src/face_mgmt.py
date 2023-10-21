import os.path
from typing import Union, Optional, Any
import face_recognition

import cv2
import numpy as np

CASCADE_LOCATION = '../resources/haarcascades/'
FRONTAL_FACE = cv2.CascadeClassifier(os.path.join(CASCADE_LOCATION, 'haarcascade_frontalface_default.xml'))


def convert_to_encoding(file: Optional[str] = None) -> np.ndarray:
    if file:
        image = cv2.imread(file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = FRONTAL_FACE.detectMultiScale(gray)
        encoding = face_recognition.face_encodings(faces)[0]
        return encoding

    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FRONTAL_FACE.detectMultiScale(gray)
        for (x, y, w, h) in faces:
            face_frame = frame[y:y + h, x:x + w]
            encoding = face_recognition.face_encodings(face_frame)[0]
            return encoding


def store_into_database(file: Optional[str] = None, *, cursor: Any) -> None:
    encoding = convert_to_encoding(file)
    query = 'INSERT INTO faces(face_encoding) VALUES (%s)'

    encoding_bytes = encoding.tobytes()

    val = encoding_bytes,
    cursor.execute(query, val)


def read_encoding_from_database(face_id: list, cursor: Any) -> np.ndarray:
    query = f'SELECT face_encoding FROM faces WHERE face_id=%s'
    cursor.execute(query, (face_id,))
    row = cursor.fetchone()

    if row is None:
        raise FileNotFoundError(f"Encoding with ID {face_id} not found in the database.")

    encoding, = row

    return np.frombuffer(encoding)

# IMMEDIATELY PENDING:
# Modify ``read_encoding_from_database`` for batch operation
# Create function for matching face
