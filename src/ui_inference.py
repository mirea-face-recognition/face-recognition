""" UI for Neural Net Inference

This is main file for UI inference developed using Streamlit framework.
"""
import sys

import streamlit as st
from PIL import Image

# sys.path.append('/src')


from model.net import compute_embedding, text2speech
from data.dataset import FacesData


faces_data = FacesData()
photo = st.camera_input(label='Сделайте фото!')
if photo:
    image = Image.open(photo)
    image.save('data/photo.png')
    embedding = compute_embedding('data/photo.png')
    result = faces_data.get_most_similar(embedding)
    if result:
        st.write(result)
        text2speech(result)
    else:
        full_name = st.text_input(label='ФИО:')
        if st.button('Зарегистрировать'):
            faces_data.add(full_name, embedding)
