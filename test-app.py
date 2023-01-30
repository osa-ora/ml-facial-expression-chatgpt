from PIL import Image
import requests
import numpy as np
import base64
from json import dumps

# some images to test the application
test_images = ['test-images/face_happy.png','test-images/face_sad.jpg','test-images/osa-ora.jpg']
service_url = 'http://localhost:5000/predict'

# You can also do the from the console by executing
# (echo -n '{"image": "'; base64 test-images/face_sad.jpg; echo '"}') | curl -H "Content-Type: application/json" -d @- http://localhost:5000/predict
{"expression":"Sad"}
# change the image name and application URL

# Loop over the array elements
for test_image in test_images:
    print(test_image)
    # Open the image file
    with open(test_image, "rb") as image_file:
        #encode the image to base64
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    content = {"image": encoded_image}
    # convert payload to json format
    json_data = dumps(content)

    #print(json_data)
    headers = {"Content-Type" : "application/json"}
    # invoke the service
    response = requests.post(service_url, data=json_data, headers=headers)

    #print(response.content)

    # Check the response status code
    if response.status_code == 200:
        # Get the expression prediction
        expression = response.json()["expression"]
        print("Detected Expression:", expression)
    else:
        # Print the error message
        print("Error:", response.content)
