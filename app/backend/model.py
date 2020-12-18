import cv2
import numpy as np

from model_class import HastTagTodel, SimpleLayersModel


def get_model():
    model = HastTagTodel()
    return model


def get_tags(model, binary_image, max_size=512, use_test_img=False):
    if use_test_img:
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    else:
        image = np.fromstring(binary_image, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    tags = model.predict(image)
    mock = {'result': tags}

    return mock
