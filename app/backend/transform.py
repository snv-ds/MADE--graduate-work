# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 13:08:02 2020

@author: ilya-
"""

import cv2
import numpy as np
import torch 

class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


class Pad(object):

    def __init__(self, max_size=1.0, p=0.1):
        self.max_size = max_size
        self.p = p

    def __call__(self, image):
        if np.random.uniform(0.0, 1.0) > self.p:
            return image
        h, w, _ = image.shape
        size = int(np.random.uniform(0, self.max_size) * min(w, h))
        image_ = cv2.copyMakeBorder(image, size, size, size, size, borderType=cv2.BORDER_CONSTANT, value=0.0)
        return image_


class Crop(object):
    def __init__(self, min_size=0.5, min_ratio=0.5, max_ratio=2.0, p=0.25):
        self.min_size = min_size
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.p = p

    def __call__(self, image):
        if np.random.uniform(0.0, 1.0) > self.p:
            return image
        h, w, _ = image.shape
        aspect_ratio = np.random.uniform(self.min_ratio, self.max_ratio)  # = w / h
        if aspect_ratio > 1:
            w_ = int(np.random.uniform(self.min_size, 1.0) * w)
            h_ = int(w / aspect_ratio)
        else:
            h_ = int(np.random.uniform(self.min_size, 1.0) * h)
            w_ = int(h * aspect_ratio)

        x = np.random.randint(0, max(1, w - w_))
        y = np.random.randint(0, max(1, h - h_))
        crop_image = image[y: y + h_, x: x + w_, :]
        return crop_image


class Resize(object):
    def __init__(self, size, keep_aspect=False):
        self.size = size
        self.keep_aspect = keep_aspect

    def __call__(self, image):
        image_ = image.copy()
        if self.keep_aspect:
            # padding step
            h, w = image.shape[:2]
            k = min(self.size[0] / w, self.size[1] / h)
            h_ = int(h * k)
            w_ = int(w * k)

            interpolation = cv2.INTER_AREA if k <= 1 else cv2.INTER_LINEAR
            image_ = cv2.resize(image_, None, fx=k, fy=k, interpolation=interpolation)

            dh = max(0, (self.size[1] - h_) // 2)
            dw = max(0, (self.size[0] - w_) // 2)
            image_ = cv2.copyMakeBorder(image_, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=0.0)

        if image_.shape[0] != self.size[1] or image_.shape[1] != self.size[0]:
            image_ = cv2.resize(image_, self.size)

        return image_


class Flip(object):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, image):
        if np.random.uniform() > self.p:
            return image
        return cv2.flip(image, 1)

class ToPILImage(object):

    def __call__(self, image):
        #print(image.shape)
        #return np.transpose(image, (0, 3, 1, 2))
        
        return torch.tensor(np.transpose(image, (2, 0, 1)), dtype=torch.float32)