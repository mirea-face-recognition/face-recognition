""" Neural Net Architecture

This file contains neural net architecture.
"""
import face_recognition


def compute_embedding(img_path):
    image = face_recognition.load_image_file(img_path)
    encoding = face_recognition.face_encodings(image)[0]
    return encoding.tolist()
