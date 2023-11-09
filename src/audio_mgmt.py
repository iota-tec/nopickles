from typing import Any, Union, Tuple
import numpy as np
import pyttsx3
from pydub import AudioSegment
import json
import io
import pyaudio
import webrtcvad
import threading
import keyboard


# This is for training
def convert_file_to_16k(filename: str) -> Union[AudioSegment, str]:
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
    elif filename.endswith('.m4a'):
        audio = AudioSegment.from_file(filename, format="m4a")
        return audio.set_frame_rate(16000).set_channels(1)

    else:
        return "Unsupported file format. Only '.wav' files are supported."


# This is for training
def store_into_database(file: str, transcript: str, cursor: Any) -> None:
    """
    Store an audio file and its transcript into audio_files table of MySQL database.

    Args:
        file (str): Path to the audio file
        transcript (str): The transcript of the audio
        cursor (MySQLCursor): Cursor object for database operations

    Note:
        This function assumes that you've already connected to the database.
        Committing the transaction is done outside this function.
    """

    query = 'INSERT INTO audio_files(audio_data, transcript, meta_data) VALUES (%s, %s, %s)'

    audio = convert_file_to_16k(file)

    # Export audio data into bytes using AudioSegment's ``export`` method
    buffer = io.BytesIO()
    audio.export(buffer, format='wav')
    audio_bytes = buffer.getvalue()

    meta_data = json.dumps({
        'filename': file,
        'length': len(audio),
        'frame_rate': audio.frame_rate,
        'channels': audio.channels
    })

    val = (audio_bytes, transcript, meta_data)
    cursor.execute(query, val)


# This is for training
def read_audio_from_database(file_id: int, cursor: Any) -> Tuple[np.ndarray, str, int]:
    """
    Reads an audio file, its transcript, and meta data from the database by file ID.

    Args:
        file_id (int): The ID of the audio file to be read from the database.
        cursor (Any): The MySQL cursor object.

    Returns:
        Tuple[np.ndarray, str, int]: A tuple containing the audio data as a NumPy array,
                                     the transcript as a string, and the frame rate as an integer.

    Raises:
        FileNotFoundError: If the row with the specified file ID does not exist.
    """

    query = f'SELECT audio_data, transcript, meta_data FROM audio_files WHERE id=%s'
    cursor.execute(query, (file_id,))
    row = cursor.fetchone()

    if row is None:
        raise FileNotFoundError(f"File with ID {file_id} not found in the database.")

    audio_bytes, transcript, meta_data_str = row
    meta_data = json.loads(meta_data_str)

    audio_array = np.frombuffer(audio_bytes, np.int16)

    return audio_array, transcript, meta_data['frame_rate']


def listen() -> np.ndarray:
    """
     Listens to audio input from the default microphone, detects speech using VAD (Voice Activity Detection),
    and stops recording upon detecting silence for a specific duration or if 'q' is pressed. The captured
    audio is returned as a NumPy array.

    The function starts a separate thread for recording audio while monitoring the keyboard for a stop
    signal (pressing 'q') in the main thread.

    Returns:
        np.ndarray: A 1D numpy array containing the recorded audio data.
    """
    frames = []
    got_speech = False
    num_silent_chunks = 0
    stop_flag = [False]  # Using a list to make it mutable inside the threaded function

    def record_audio():
        vad = webrtcvad.Vad()
        vad.set_mode(1)

        p = pyaudio.PyAudio()
        sample_rate = 16000
        chunk_duration = 30
        chunk_size = int(sample_rate * chunk_duration / 1000)

        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_size,
        )

        print("Listening...")
        nonlocal frames, got_speech, num_silent_chunks
        while True:
            chunk = stream.read(chunk_size)
            chunk_np = np.frombuffer(chunk, dtype=np.int16)

            is_speech = vad.is_speech(chunk, sample_rate)

            if got_speech and not is_speech:
                num_silent_chunks += 1
            elif is_speech:
                got_speech = True
                num_silent_chunks = 0
                frames.append(chunk_np)

            if num_silent_chunks > 50 or stop_flag[0]:
                print("Detected silence or 'q' pressed. Stopping recording...")
                stream.stop_stream()
                stream.close()
                p.terminate()
                break

    audio_thread = threading.Thread(target=record_audio)
    audio_thread.start()

    print("Press 'q' to stop the recording.")
    keyboard.wait('q')
    stop_flag[0] = True
    audio_thread.join()

    return np.concatenate(frames, axis=0)


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