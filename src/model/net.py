""" Neural Net Architecture

This file contains neural net architecture.
"""
from deepface import DeepFace


def compute_embedding(img_path):
    embedding_objs = DeepFace.represent(img_path=img_path)
    embedding = embedding_objs[0]["embedding"]
    return embedding
