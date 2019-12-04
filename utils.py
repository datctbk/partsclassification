import os
import io
import time
import urllib
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import base64
from PIL import Image
IMAGE_SIZE = (224,224)
model = load_model('resnet50custom.h5')
'''
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
	predictions = proba[0]
	return predictions
'''
def preprocess_image(image, target_size):
	image = image.astype("float")
	image = image[..., ::-1]
	image = image.resize(target_size)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	return image

def get_predictions(request):
	message = request.get_json(force=True)
	encoded = message['image']
	decoded = base64.b64decode(encoded)
	image = Image.open(io.BytesIO(decoded))
	processed_image = preprocess_image(image, target_size=(224, 224))
	prediction = model.predict(processed_image).tolist()
	response = {
		'prediction': {
			'broken': prediction[0][0],
			'good': prediction[0][1]
		}
	}
	return jsonify(response)