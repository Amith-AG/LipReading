import streamlit as st
import os 
from moviepy.editor import VideoFileClip
import imageio
import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.markdown("<h1 style='text-align: center; color: white;'>Abstract</h1>", unsafe_allow_html=True) 
    st.info('This project, developed by Amith A G as his MCA final project at KVVS Institute Of Technology, focuses on implementing the LipNet deep learning model for lip-reading and speech recognition. The project aims to demonstrate the capabilities of the LipNet model through a Streamlit application.')

st.markdown("<h1 style='text-align: center; color: white;'>LipNet</h1>", unsafe_allow_html=True) 
# Generating a list of options or videos 
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(repo_root, 'data', 's1')
options = os.listdir(data_dir)

# Generate two columns 
col1, col2 = st.columns(2)

if options: 

    # Rendering the video 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('..', 'data', 's1', selected_video)
        output_path = os.path.join('test_video.mp4')
    
        # Convert the video using moviepy
        video_clip = VideoFileClip(file_path)
        video_clip.write_videofile(output_path, codec='libx264')
    
        # Display the video in the app
        video = open(output_path, 'rb')
        video_bytes = video.read()
        st.video(video_bytes)


    with col2: 
        st.info('This is all the machine learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
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
