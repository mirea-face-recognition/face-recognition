""" Neural Net Architecture

This file contains neural net architecture.
"""
from deepface import DeepFace
import torch
import sounddevice as sd
from streamlit import cache_resource

from config import LANGUAGE, MODEL_ID, SAMPLE_RATE, SPEAKER


def compute_embedding(img_path):
    embedding_objs = DeepFace.represent(img_path=img_path)
    embedding = embedding_objs[0]["embedding"]
    return embedding


@cache_resource
def get_model():
    model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                              model='silero_tts',
                              language=LANGUAGE,
                              speaker=MODEL_ID)
    return model


def text2speech(text):
    audio = get_model().apply_tts(
        text=text,
        speaker=SPEAKER,
        sample_rate=SAMPLE_RATE)
    sd.play(audio, SAMPLE_RATE)
