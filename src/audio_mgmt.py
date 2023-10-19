import os
from typing import Any, Union, Tuple
import deepspeech
import numpy as np
import pyttsx3
from gtts import gTTS
import pygame as pg
import pandas as pd
import wave
from pydub import AudioSegment
import mysql.connector
import json
import io


def convert_to_16k(filename: str) -> Union[AudioSegment, str]:
    """
    Convert a given audio file to 16K frames and 1 channel.

    Args:
        filename (str): The name of the audio file, must end with '.wav'

    Returns:
        AudioSegment: Audio data with 16K frame rate and 1 channel.
    """
    # Check if the input file is a WAV file
    if filename.endswith('.wav'):
        # Load the audio file
        audio = AudioSegment.from_wav(filename)

        # Set the frame rate to 16K and channels to 1
        return audio.set_frame_rate(16000).set_channels(1)
    else:
        return "Unsupported file format. Only '.wav' files are supported."


def store_into_database(file: str, transcript: str, cursor: Any) -> None:
    """
    Store an audio file and its transcript into a MySQL database.

    Args:
        file (str): Path to the audio file
        transcript (str): The transcript of the audio
        cursor (MySQLCursor): Cursor object for database operations

    Note:
        This function assumes that you've already connected to the database.
        Committing the transaction is done outside this function.
    """

    # SQL query to insert new record into 'audio_files' table
    query = 'INSERT INTO audio_files(audio_data, transcript, meta_data) VALUES (%s, %s, %s)'

    # Convert audio to 16K frame rate and 1 channel
    audio = convert_to_16k(file)

    # Prepare to convert audio into bytes
    buffer = io.BytesIO()

    # Export audio data into bytes
    audio.export(buffer, format='wav')
    audio_bytes = buffer.getvalue()

    # Prepare metadata as a JSON string
    meta_data = json.dumps({
        'filename': file,
        'length': len(audio),
        'frame_rate': audio.frame_rate,
        'channels': audio.channels
    })

    # Values to be inserted into the table
    val = (audio_bytes, transcript, meta_data)

    # Execute the SQL query
    cursor.execute(query, val)


def read_wav_from_database(file_id: int, cursor: Any) -> Tuple[np.ndarray, str, int]:
    """
    Reads a .wav audio file, its transcript, and meta data from the database by file ID.

    Args:
        file_id (int): The ID of the audio file to be read from the database.
        cursor (Any): The MySQL cursor object.

    Returns:
        Tuple[np.ndarray, str, int]: A tuple containing the audio data as a NumPy array,
                                     the transcript as a string, and the frame rate as an integer.

    Raises:
        FileNotFoundError: If the row with the specified file ID does not exist.
    """

    query = f'SELECT audio_data, transcript, meta_data FROM audio_files WHERE id={file_id}'
    cursor.execute(query)
    row = cursor.fetchone()

    if row is None:
        raise FileNotFoundError(f"File with ID {file_id} not found in the database.")

    audio_bytes, transcript, meta_data_str = row
    meta_data = json.loads(meta_data_str)

    audio_array = np.frombuffer(audio_bytes, np.int16)

    return audio_array, transcript, meta_data['frame_rate']


# Initialize TTS engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)


def speak(response: str) -> None:
    """
    This function speaks out loud whatever string is passed in response argument.

    Args:
        response: A string to synthesize

    Returns: None
    """
    engine.say(response)
    engine.runAndWait()
