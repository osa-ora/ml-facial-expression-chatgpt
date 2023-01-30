import json
from flask import Flask, jsonify, request
import io
import numpy as np
from PIL import Image
import requests
import tensorflow as tf
import base64
import os
from os.path import splitext,basename
import uuid
# Define our application
app = Flask(__name__)

# Define different expression labels
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
# Load the model 
model = tf.keras.models.load_model('facial_expression_recognition_model.h5')

# Method to do some image processing
def preprocess_image(decoded_image):
        # Convert the decoded image to a BytesIO object
        image = io.BytesIO(decoded_image)
        # Open the image with PIL
        image = Image.open(image)
        # Convert the PIL Image to a numpy array
        image = np.array(image)
        # Reshape the numpy array to match the input shape expected by the model
        image = image.reshape(-1, 48, 48, 1)
        return image

# Method to map the predicted value to an acutal expression label
def map_prediction_to_emotion(prediction):
    class_index = np.argmax(prediction[0])
    print(class_index)
    emotion = emotions[class_index]
    return emotion

# Health checks methods
@app.route('/')
@app.route('/status')
def status():
    return jsonify({'status': 'ok'})

# Predict method that can be invoked with a photo to predict the facical expression
# it supports only POST method and it expect JSON formatted in the form of {"image": "encoded-base64-face-image-48x48-grayscale"}
# the image could be either jpg or png
@app.route("/predict", methods=["POST"])
def predict():
    print("Received a request to make a prediction")
    if request.method == "POST":
        print("Received a POST request")

        # Get the encoded image from the request body
        encoded_image = request.get_json().get('image')
        decoded_image = base64.b64decode(encoded_image)
        image = preprocess_image(decoded_image)


        # Pass the preprocessed input to the model for prediction
        prediction = model.predict(image)
        print("Predicted emotion:", prediction)
        prediction = map_prediction_to_emotion(prediction)
        print("Predicted emotion:", prediction)
        # Return the prediction as a response
        response = {
            "expression": prediction
        }
        return jsonify(response)
    else:
        return "Invalid request method, only Post is supported"
if __name__ == '__main__':
    app.run(port=8080)
