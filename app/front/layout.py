from requests_toolbelt.multipart.encoder import MultipartEncoder
import requests
import streamlit as st
from PIL import Image
import json

# predicted = ['еда', 'вкусноста', 'сладости']

backend = 'http://fastapi:8000/predict'


def show_body():
    st.header("Predict hashtags")
    image = st.file_uploader("Upload your image to predict hashtags", type=["png", "jpg", "jpeg"])

    if st.button('Give me my tags!'):

        if image is None:
            st.write("Insert an image!")  # handle case with no image
        else:
            st.success("Uploaded")
            st.image(Image.open(image), width=450)
            predicted = process(image, backend)
            tags = json.loads(predicted.text)['result']
            st.multiselect(label="we think this tags can be used", options=tags, default=tags)

            st.markdown(' '.join(['#' + tag for tag in tags]))


def show_sidebar():
    st.sidebar.subheader(
        "Predict hashtags for your photo"
    )

    st.sidebar.markdown(
        """

    """
    )


def process(image, server_url: str):

    m = MultipartEncoder(
        fields={'file': ('filename', image, 'image/jpeg')}
        )
    r = requests.post(server_url,
                      data=m,
                      headers={'Content-Type': m.content_type},
                      timeout=8000)
    return r
