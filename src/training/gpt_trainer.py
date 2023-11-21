from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
import tensorflow as tf
import os
import joblib
import re
import os
import numpy as np

model = TFGPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


def process_conversational_data(file_path):
    sets_of_eight = []
    current_set = []

    with open(file_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()

    for i, line in enumerate(all_lines):
        # Trim the newline character from the end of the line and convert to lower case
        line = line.rstrip('\n').lower()

        # Replace prices with <price> tag
        line = re.sub(r'\$\d+(\.\d{2})?', '<price>', line)

        if i > 0:
            # Extract the first word (speaker) of the current and previous lines
            speaker_current = line.split(':')[0]
            speaker_previous = all_lines[i - 1].rstrip('\n').lower().split(':')[0]

            # Check if the current line has a different speaker than the previous line
            if speaker_current != speaker_previous:
                current_set.append(line)

                # Add the set to sets_of_eight every 8 lines
                if len(current_set) == 8:
                    sets_of_eight.append(current_set)
                    current_set = []
            else:
                # If the same speaker is found consecutively, reset the current set
                current_set = [line]  # Start a new set with the current line
        else:
            # For the very first line, just add it to the current set
            current_set.append(line)

    # Check if there's a leftover set with less than 8 lines
    if current_set:
        sets_of_eight.append(current_set)

    return sets_of_eight


def prepare_for_gpt2(sets_of_eight, intent):
    processed_sequences = []
    for set in sets_of_eight:
        # Concatenate the lines in the set into a single string
        concatenated_sequence = " ".join(set)
        # Add the GPT-2 end-of-text token
        sequence_with_token = f"{concatenated_sequence} <{intent}> <|endoftext|>"
        processed_sequences.append(sequence_with_token)
    return processed_sequences


def save_file(filepath, value):
    with open(filepath, 'wb') as f:
        joblib.dump(value, f)
