# request.py dependencies
import requests
import json
import cv2
import base64




url = "http://35.246.62.128:5000/predict"
headers = {"content-type": "base64"}
# encode image
with open("test/IMG_0585.JPG", "rb") as img_file:
  my_string = base64.b64encode(img_file.read())


  # send HTTP request to the server
  response = requests.post(url, data=my_string, headers=headers)
  predictions = response.json()
  # annotate the image
  for pred in predictions:
     # print prediction
     print(pred)
   # extract the bounding box coordinates
   
# save annotated image
#cv2.imwrite("annotated_image.jpg", image)