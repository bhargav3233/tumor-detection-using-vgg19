# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 10:24:17 2022

@author: Bhargav
"""

import streamlit as st
import tensorflow as tf
import streamlit as st
from tensorflow.keras.applications.imagenet_utils import decode_predictions


@st.cache(allow_output_mutation=True)

def load_model():
  model=tf.keras.models.load_model('D:/Tumor/tumor_vgg19.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()
  
st.write("""
         # Brain Tumor Detection
         """
         )


file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)




def upload_predict(upload_image, model):
    size = (180,180)    
    image = ImageOps.fit(upload_image, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img, dsize=(224, 224),interpolation=cv2.INTER_CUBIC)
    
    img_reshape = img_resize[np.newaxis,...]

    prediction = model.predict(img_reshape)
    pred_class=decode_predictions(prediction,top=1)
    
    return pred_class

          
        
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = upload_predict(image, model)
    label = tf.nn.softmax(prediction[0])
    st.write(prediction)
    st.write(label)
    if label == 0:
        st.write("This Person has a brain tumor")
    else:
        st.write("This person scan is healthy")
   
    
   
    
   
    
"""img = image
   #data = np.ndarray(shape=(224, 224) 
   size = (224, 224)
   image = ImageOps.fit(image, size, Image.ANTIALIAS)
   
   #image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
   #turn the image into a numpy array
   image = np.asarray(image)
   
   img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   
   img_resize = cv2.resize(img, dsize=(224, 224),interpolation=cv2.INTER_CUBIC)
   # Normalize the image
   img = (img.astype(np.float32) / 127.0) - 1
   
   #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
   
   #img_reshape = img[np.newaxis,...]
   # Load the image into the array
   data[0] = img"""