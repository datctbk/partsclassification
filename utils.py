import os
import time
import urllib
import numpy as np
import cv2

from keras.models import load_model
IMAGE_SIZE = (224,224)
model = load_model('resnet50custom.h5')

def get_predictions(raw_image):
	# load input image and grab its spatial dimensions
	nparr = np.fromstring(raw_image.data, np.uint8)
	# decode image
	image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	(H, W) = image.shape[:2]
	img = image.astype("float")
	img = img[..., ::-1]
	img = cv2.resize(img, IMAGE_SIZE)
	img = img_to_array(img)
	img = np.expand_dims(img, axis=0)
	proba = model.predict(img)
	predictions = path,proba[0]
	return predictions