# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 11:04:35 2022

@author: pulkit_jain
"""

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import pandas as pd
import cv2
import os
import streamlit as st
import numpy as np

os.chdir(r"C:\Users\pulkit_jain\Downloads\Computer Vision Dataset")
model=load_model('gender_detection_model.model')

header= st.container()
upload_file=st.container()
test_interface=st.container()

st.markdown(
        """
        <style>
        .main{
        background-color:#F5F5F5;
        }
        </style>
        """,
        unsafe_allow_html=True
        )



with header:
    st.title("CNN Model")
    
    
with upload_file:
    st.header("Upload File here")
    uploaded_file = st.file_uploader('Please upload the image. File format must be in .png or .jpeg')

with test_interface:
    if uploaded_file != None:
        files_bytes = np.asarray(bytearray(uploaded_file.read()),dtype=np.uint8)
        opencv_image = cv2.imdecode(files_bytes,1)
        im_rgb = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
        st.image(im_rgb,caption = 'Input')
        opencv_image = cv2.resize(opencv_image, (96, 96))
        image = img_to_array(opencv_image)
        image=np.expand_dims(image,axis = 0)
        images = [image]
        images = np.array(images,dtype='float')/255.0
        result = model.predict(image)
        print(result)
        result = ((result > 0.5)+0).ravel()
        result=result.flat[0]

        if result==1.0:
            st.header("It's a MALE!!")
        else:
            st.header("It's a FEMALE!!")
        
        












