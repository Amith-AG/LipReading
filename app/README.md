# LipNetStreamlit
The LipNet project is a deep learning model for lip-reading and speech recognition. It aims to recognize spoken words by analyzing lip movements in video sequences. The project consists of three Python files:

1. `modelutil.py`: This file contains the code for loading the LipNet model using TensorFlow and Keras. It defines the architecture of the model, which includes convolutional, LSTM, and dense layers. The model weights are loaded from a checkpoint file.

2. `utils.py`: This file provides utility functions used in the project. It includes functions for loading video frames, alignments (transcriptions of spoken words), and other data preprocessing tasks. The functions make use of libraries such as OpenCV and TensorFlow.

3. `streamlitapp.py`: This file implements a web application using Streamlit. It utilizes the LipNet model and the utility functions from the previous files. The Streamlit app allows users to choose a video, displays the video, processes the frames using the LipNet model, and shows the predicted speech transcriptions.

The Streamlit app layout includes a sidebar with project information, such as an abstract and a logo. The main section displays a selection box for choosing a video. Once a video is selected, it is displayed in one column, and the processed frames and predicted transcriptions are shown in the other column.

The app demonstrates the capabilities of the LipNet model and provides a user-friendly interface to interact with the lip-reading and speech recognition system.

Note: The code provided here assumes that the necessary dependencies and data files are available in the specified locations. Make sure to install the required libraries and set up the data files accordingly before running the application.
