""" Neural Net Architecture

This file contains neural net architecture.
"""
from deepface import DeepFace
import torch
import sounddevice as sd

from src.config import LANGUAGE, MODEL_ID, SAMPLE_RATE, SPEAKER


def compute_embedding(img_path):
    embedding_objs = DeepFace.represent(img_path=img_path)
    embedding = embedding_objs[0]["embedding"]
    return embedding


def text2speech(text):
    model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                         model='silero_tts',
                                         language=LANGUAGE,
                                         speaker=MODEL_ID)
    audio = model.apply_tts(text=text,
                            speaker=SPEAKER,
                            sample_rate=SAMPLE_RATE)
    sd.play(audio, SAMPLE_RATE)
