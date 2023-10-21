import os.path
from typing import Union, Optional, Any, List, Tuple
import face_recognition
import cv2
import mysql
import numpy as np


def convert_to_encoding(file: Union[str, None] = None) -> Union[Tuple[str, Any], Tuple[None, Any]]:
    """
    Convert a given image file or video feed into face encoding.

    Parameters:
    - file (Union[str, None]): The path to the image file or None for webcam feed.

    Returns:
    - Union[Tuple[str, Any], Tuple[None, Any]]: A tuple containing the person's name and encoding
      if an image file is provided, otherwise None and the encoding from the webcam feed.
    """

    if file:
        person_name = os.path.basename(file).split('.')[0]

        image = cv2.imread(file)
        face_locations = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(image, face_locations)

        # Returning only the first encoding, multiple encoding support pending.
        return person_name, encodings[0]

    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()

        face_locations = face_recognition.face_locations(frame)
        encodings = face_recognition.face_encodings(frame, face_locations)

        return None, encodings[0]


def store_into_database(file: Optional[str] = None, *, cursor: Any) -> None:
    """
    Store a person's face encoding into the database.

    Parameters:
        - file (Optional[str]): The path to the image file. Default is None, in which case it will capture
          frames from live video feed.
        - cursor (Any): The database cursor for executing SQL commands.

    Returns:
        - None: The function performs database insertion and returns nothing.
    """

    person_name, encoding = convert_to_encoding(file)
    query = 'INSERT INTO faces(person_name, face_encoding) VALUES (%s, %s)'

    encoding_bytes = encoding.tobytes()
    val = (person_name, encoding_bytes)

    cursor.execute(query, val)


def read_encoding_from_database(face_id: Union[int, List[int]], cursor: Any) -> List[np.ndarray]:
    """
    Fetch face encoding(s) from the database given the face ID(s).

    Parameters:
    - face_id (Union[int, List[int]]): A single face ID or a list of face IDs.
    - cursor (pymysql.cursors.Cursor): Database cursor for executing SQL queries.

    Returns:
    - List[np.ndarray]: A list of numpy arrays representing face encodings.

    Raises:
    - FileNotFoundError: If any of the face IDs do not have an associated encoding.
    """

    # Prepares variables for batch operation if face_id is a list, and for single operation if face_id is a single index
    placeholder = ', '.join(['%s'] * len(face_id)) if not isinstance(face_id, int) else '%s'
    face_id = (face_id,) if isinstance(face_id, int) else face_id

    query = f'SELECT face_encoding FROM faces WHERE face_id IN ({placeholder})'
    cursor.execute(query, face_id)
    rows = cursor.fetchall()

    # Check if any row has None value
    if any(None in row for row in rows):
        idx = next((i for i, row in enumerate(rows) if None in row), None)
        raise FileNotFoundError(f"Encoding with ID {idx} not found in the database.")

    encodings = [np.frombuffer(row[0]) for row in rows]

    return encodings

# IMMEDIATELY PENDING:
# Create function for matching face
