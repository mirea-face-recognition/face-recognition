""" UI for Neural Net Inference

This is main file for UI inference developed using Streamlit framework.
"""
import sys

import streamlit as st
from PIL import Image

sys.path.append('C:\\Users\\melik\\PycharmProjects\\face-recognition\\src')

from model.net import compute_embedding
from data.dataset import FacesData


faces_data = FacesData()
photo = st.camera_input(label='Сделайте фото!')
if photo:
    image = Image.open(photo)
    image.save('src/data/photo.png')
    embedding = compute_embedding('src/data/photo.png')
    result = faces_data.get_most_similar(embedding)
    if result:
        st.write(result)
    else:
        full_name = st.text_input(label='ФИО:')
        if st.button('Зарегистрировать'):
            faces_data.add(full_name, embedding)
