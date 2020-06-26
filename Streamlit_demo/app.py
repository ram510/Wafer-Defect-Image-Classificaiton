import numpy as np
import streamlit as st
from matplotlib.pyplot import imread

from tensorflow.keras.models import load_model
import tensorflow as tf

class_indices = {'Center': 0,
                 'Donut': 1,
                 'Edge-loc': 2,
                 'Edge-ring': 3,
                 'Loc': 4,
                 'Near-Full': 5,
                 'None': 6,
                 'Random': 7,
                 'Scratch': 8}

saved_model = load_model('C://Users//rchilakamarr//PycharmProjects//Streamlit_demo//static//to_deploy.h5')


def predict(filepath):
    pred_list = saved_model.predict(filepath)
    keys = list(class_indices.keys())
    st.info('wafer defect classifed as ' + str(keys[pred_list.argmax()]))


def file_upload():
    st.title("Wafer Classification")

    uploaded_file = st.file_uploader("Choose an image...", )

    if uploaded_file is not None:
        # img_shape = (64, 65, 4)
        st.image(uploaded_file,caption="Image uploaded")
        arr = imread(uploaded_file, format='jpg')
        arr = tf.image.resize(arr, (64, 65))
        arr = tf.keras.preprocessing.image.array_to_img(arr)
        arr = np.expand_dims(arr, axis=0)
        predict(arr)


file_upload()
