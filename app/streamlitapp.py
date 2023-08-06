# Import all of the dependencies
import streamlit as st
import os 
import imageio 
import pathlib
import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.markdown("<h1 style='text-align: center;'>Abstract</h1>", unsafe_allow_html=True) 
    st.info('This project, developed by Amith A G as his MCA final project at KVVS Institute Of Technology, focuses on implementing the LipNet deep learning model for lip-reading and speech recognition. The project aims to demonstrate the capabilities of the LipNet model through a Streamlit application.')

st.markdown("<h1 style='text-align: center;'>LipNet</h1>", unsafe_allow_html=True) 
# Generating a list of options or videos 
code_dir = pathlib.Path(__file__).parent.resolve()
files_location = code_dir / ".." / "data" / "s1"  
files_location = files_location.resolve()  

# Convert the files_location to a list of files
options = os.listdir(files_location)

selected_video = st.selectbox('Choose video', options)

# Generate two columns 
col1, col2 = st.columns(2)

if options: 

    # Rendering the video 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = str(files_location / selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering inside of the app
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)


    with col2: 
        st.info('This is all the machine learning model sees when making a prediction')
        video,annotations= load_data(tf.convert_to_tensor(file_path))
        imageio.mimsave('animation.gif', video, fps=10)
        st.image('animation.gif', width=400) 

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
        
