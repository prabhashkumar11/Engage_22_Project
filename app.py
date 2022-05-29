import streamlit as st 
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import io
import streamlit as st
import cv2
from PIL import Image, ImageOps
import streamlit.components.v1 as components


# configuring default page
st.set_page_config( page_icon=":tada", layout="wide")

# removing defalut settings
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)
padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)





# header
components.html(
    """
    <div style="color:blue;text-align:center; 
        background-image: linear-gradient(to right top, #483640, #453442, #413345, #3b3248, #33324b, #2a334b, #20334a, #163348, #123342, #14333b, #193234, #1f302f);
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19); border-radius: 25px; width:100%;" >

      <h1 style="color:white;text-align:center; 
          font-family: Arial, Helvetica, sans-serif;  font-size: 60px;">Face Product Recommendation System</h1>
    </div>
    """
    
)
 
with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)

# loading model

@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('mymodel3.hdf5')
    return model

model =load_model()
# st.write("""
#           # Face Product Recommendation System
#         """
#         ) 
file = st.file_uploader("Please upload an face image",type=['JPEG','jpg'])
import cv2

import numpy as np


# prediction function

def import_and_predict(image_data,model):
    
    size=(180,180)
    image = ImageOps.fit(image_data,size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)

    return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)      
    st.image(image, width=260)
    predictions = import_and_predict(image, model)
    class_names = ['not face','normal skin' , 'oily skin'] 
    # conditions for normal skin 
    if np.argmax(predictions) == 1:  
        string = "Your Face is most likely of Normal Skin. You should use following products "
        st.subheader(string)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.header("Product 1")
            st.image("./Products/normal skin products/himalaya-natural-glow-kesar-face-wash-500x500.jpg",caption="Himalaya Natural Glow Kesar face wash", width=220)
            link='check out this [Product](https://himalayawellness.in/products/natural-glow-kesar-face-wash)'
            st.markdown(link,unsafe_allow_html=True) 

        with col2:
            st.header("Product 2")
            st.image("./Products/normal skin products/download.jpg",caption="Vitamin C Face Wash", width=220)
            link='check out this [Product](https://mamaearth.in/product/vitamin-c-face-wash-with-vitamin-c-and-turmeric-for-skin-illumination-100ml)'
            st.markdown(link,unsafe_allow_html=True)
            
            st.header("Product 4")
            st.image("./Products/normal skin products/Ponds purity face wash.jpg",caption="Ponds Purity Face Wash", width=220)
            link='check out this [Product](https://www.amazon.in/Ponds-Pollution-Activated-Charcoal-Facewash/dp/B07B5J4H7H)'
            st.markdown(link,unsafe_allow_html=True)

        with col3:
            st.header("Product 3")            
            st.image("./Products/normal skin products/Neutrona deep cleaning.jpg",caption="Neutrona Deep Cleaning Face Wash", width=220)
            link='check out this [Product](https://www.amazon.in/Neutrogena-Clean-Facial-Cleanser-200ml/dp/B006LXDQRY)'
            st.markdown(link,unsafe_allow_html=True)
            
    # condition for oily skin
    
    elif  np.argmax(predictions) == 2:
        string = "Your Face is most likely of Oily Skin . You Should use following Products "  
        st.subheader(string)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.header("Product 1")
            st.image("./Products/oily skin products/download.jpg",caption="Himalaya Natural Glow Kesar face wash", width=220)
            link='check out this [Product](https://www.amazon.in/Himalaya-Fairness-Kesar-Face-150ml/dp/B00NARTS1C)'
            st.markdown(link,unsafe_allow_html=True)


        with col2:
            st.header("Product 2")
            st.image("./Products/oily skin products/natural ayurvedic.png",caption="Natural Vibes ~ Ayurvedic Tea Tree Face Wash", width=220)
            link='check out this [Product](https://www.amazon.in/Natural-Vibes-Ayurvedic-blemishes-Essential/dp/B084SPW3XK)'
            st.markdown(link,unsafe_allow_html=True)
            st.header("Product 4")
            st.image("./Products/oily skin products/biotquie.jpg",caption="Biotique BIO Honey Gel Face Wash", width=220)
            link='check out this [Product](https://www.amazon.in/Biotique-Honey-Refreshing-Foaming-150ml/dp/B00KCLZ6VU?th=1)'
            st.markdown(link,unsafe_allow_html=True)
          

        with col3:
            st.header("Product 3")
            
            st.image("./Products/oily skin products/green tea.jpg",caption="Plum Green Tea Pore Cleansing Face Wash", width=220)
            link='check out this [Product](https://plumgoodness.com/products/green-tea-pore-cleansing-face-wash)'
            st.markdown(link,unsafe_allow_html=True)

       # conditions for uploading image other than of face   
    else:
        st.subheader("Please Upload Image of Face")
    


# footer

footer="""<style>
a:link , a:visited{
color: white;
font-family: Arial, Helvetica, sans-serif; 
font-size: 60px;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-image: linear-gradient(to right top, #483640, #453442, #413345, #3b3248, #33324b, #2a334b, #20334a, #163348, #123342, #14333b, #193234, #1f302f);
color: white;
text-align: center;
}
p {
    font-family: Arial, Helvetica, sans-serif;
    font-size: 30px;
}
</style>
<div class="footer">
<p>Developed with ❤ by <a style=' text-align: center; font-family: Arial, Helvetica, sans-serif;  font-size: 30px;' href="https://github.com/prabhashkumar11" target="_blank">Prabhash Kumar</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
