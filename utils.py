import os
import time
import urllib
import numpy as np
import cv2

from keras.models import load_model
IMAGE_SIZE = (224,224)
model = load_model('resnet50custom.h5')

def get_predictions(raw_image):
	img = raw_image.astype("float")
	img = img[..., ::-1]
	img = cv2.resize(img, IMAGE_SIZE)
	img = img_to_array(img)
	img = np.expand_dims(img, axis=0)
	proba = model.predict(img)
	predictions = path,proba[0]
	return predictions