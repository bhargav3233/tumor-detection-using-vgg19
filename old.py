# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 21:38:46 2022

@author: Bhargav
"""

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow.keras as keras
import tensorflow_hub as hub
import matplotlib.pyplot as plt

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html = True)

st.title('Brain Cancer Detection')

def main() :
    file_uploaded = st.file_uploader('Choose an image...', type = 'jpg')
    if file_uploaded is not None :
        image = Image.open(file_uploaded)
        st.write("Uploaded Image.")
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        st.pyplot(figure)
        result, confidence = predict_class(image)
        st.write('Prediction : {}'.format(result))
        st.write('Confidence : {}%'.format(confidence))

def predict_class(image) :
    with st.spinner('Loading Model...'):
        classifier_model = keras.models.load_model('tumor_vgg19.h5', compile = False)

    shape = (224, 224,3) 
    model = keras.Sequential([hub.KerasLayer(classifier_model, input_shape = shape)])
    test_image = image.resize((224, 224))
    test_image = keras.preprocessing.image.img_to_array(test_image)
    test_image /= 255.0
    test_image = np.expand_dims(test_image, axis = 0)
    class_name = ['No_Tumor', 'Meningioma_Tumor', 'Glioma_Tumor', 'Pituitary_Tumor']
    prediction = model.predict(test_image)
    confidence = round(100 * (np.max(prediction[0])), 2)
    final_pred = class_name[np.argmax(prediction)]
    return final_pred, confidence

footer = """<style>
a:link , a:visited{
    color: white;
    background-color: transparent;
    text-decoration: None;
}
a:hover,  a:active {
    color: red;
    background-color: transparent;
    text-decoration: None;
}
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: transparent;
    color: black;
    text-align: center;
}
</style>
<div class="footer">
<p align="center"> <a href="https://www.linkedin.com/in/bhargav-reddy-8602b1221/">Developed with ❤ by Bhargav</a></p>
</div>
        """

st.markdown(footer, unsafe_allow_html = True)

if __name__ == '__main__' :
    main()
