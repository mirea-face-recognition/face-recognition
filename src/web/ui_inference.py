""" UI for Neural Net Inference

This is main file for UI inference developed using Streamlit framework.
"""
import streamlit as st


model = ...  # singleton model instance
photo = st.camera_input(label='take a photo')
if photo:
    processed_photo = ...  # preprocess photo before pass it to neural net
    embedding = ...
    result = '<Имя пользователя>'  # call specialized methods to get name with embedding
    if result:
        st.write(result)
    else:
        full_name = st.text_input(label='ФИО:')
        if st.button('Зарегистрировать'):
            # register new user
            pass
