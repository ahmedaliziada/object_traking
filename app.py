import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import time


def convert_color(image):
    """Convert the input image to RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



# Set page configuration
st.set_page_config(
    page_title="Object Tracking App",
    page_icon=":guardsman:",
    layout="wide",
)


with st.sidebar:
    st.header("🎯 Object Tracking App")
    st.markdown(
        """
        This app allows you to upload a video and see object tracking in action using OpenCV's Background Subtraction method.
        """)
    st.markdown("---")
    st.header("🎨 Customization")
    color = st.color_picker("Pick a color", "#FF0000")
    st.markdown("---")
    st.header("speed")
    speed = st.slider("Processing Speed", 1, 60, 30)


    
    hex_color = color.lstrip('#')
    rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    # color hex --> rgb
    #explain:
    #1. `color.lstrip('#')` removes the '#' character from the beginning of the hex color string.
    #2. `tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))` creates a tuple of RGB values by slicing the hex color string into pairs of characters (representing red, green, and blue) and converting each pair from hexadecimal to decimal using `int(..., 16)`.


uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.close()
    
    capture = cv2.VideoCapture(temp_file.name)



    background_subtractor= cv2.createBackgroundSubtractorMOG2()

    col1,col2= st.columns(2)

    with col1:
        st.subheader('Processed Frame')
        fram1 = st.empty()
        
    with col2:
        # st.subheader('Foreground Mask')
        frame2 = st.empty()

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)

    f_id = 0

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        fg_mask = background_subtractor.apply(frame)
        (contours, _) = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x,y,w,h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x,y), (x+w, y+h), rgb_color, 2)

        f_id += 1
        
        progress_bar.progress(f_id / frame_count)

        fram1.image(convert_color(frame), caption='Processed Frame')
        # frame2.image(fg_mask, caption='Foreground Mask')
        time.sleep(0.001)


    capture.release()
