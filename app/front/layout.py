from requests_toolbelt.multipart.encoder import MultipartEncoder
import requests
import streamlit as st
import os
import numpy as np
from PIL import Image
import json

# predicted = ['еда', 'вкусноста', 'сладости']

backend = 'http://fastapi:8000/predict'
# backend_for_test = 'http://fastapi:8000/predict_test'
test_image_path = 'test_data'

def show_body():
    st.header("Predict hashtags")
    image = st.file_uploader("Upload your image to predict hashtags", type=["png", "jpg", "jpeg"])
    use_test = st.checkbox("Use test data")
    tags = None
    if use_test:
        img = st.selectbox(label="test images", options=os.listdir('test_data'))
        image = os.path.join(test_image_path, img)
    if st.button('Give me my tags!'):

        if image is None:
            st.write("Insert an image!")  # handle case with no image
        else:
            st.success("Uploaded")
            st.image(Image.open(image), use_column_width=True)
            tags = get_tags(image, backend)
            # st.multiselect(label="we think this tags can be used", options=tags, default=tags, )
            st.text("we think this tags can be used")
            st.markdown(' '.join(['#' + tag for tag in tags]))
    return tags


def show_sidebar():
    st.sidebar.subheader(
        "Predict hashtags for your photo"
    )
    st.sidebar.markdown(
        """
        Github of our project https://github.com/snv-ds/MADE--graduate-work
    """
    )


@st.cache
def get_tags(image, backend, use_test_img=False):
    if use_test_img:
        image = np.fromfile(image, dtype=np.uint8)
    predicted = process(image, backend)
    return json.loads(predicted.text)['result']


def process(image, server_url: str, use_test_img=False):
    m = MultipartEncoder(
        fields={'file': ('filename', image, 'image/jpeg')}
    )
    r = requests.post(server_url,
                      data=m,
                      headers={'Content-Type': m.content_type},
                      timeout=8000)
    return r
