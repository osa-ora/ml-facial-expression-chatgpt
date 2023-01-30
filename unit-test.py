import unittest
import base64
import json
import requests
from app import app

class PredictTestCase(unittest.TestCase):

    def setUp(self):
        # set up the test client
        self.url = "http://localhost:8080/predict"
        self.header = {'Content-Type': 'application/json'}
        self.app = app.test_client()
    def test_predict_sad(self):
        with open("test-images/face_sad.jpg", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        print("Test Sad Expression")
        self.data = '{"image": "' + encoded_string + '"}'
        # send a POST request to the predict endpoint with the test image
        response = self.app.post(self.url, headers=self.header, data=self.data)
        #response = requests.post(self.url, headers=self.header, data=self.data)

        # check that the response is 200 OK
        self.assertEqual(response.status_code, 200)
        returnvalue=json.loads(response.get_data(as_text=True))
        print(returnvalue)
        # check that the response data is as expected
        self.assertIn('Sad', returnvalue["expression"])
        
    def test_predict_happy(self):
        with open("test-images/face_happy.png", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        print("Test Happy Expression")
        self.data = '{"image": "' + encoded_string + '"}'
        # send a POST request to the predict endpoint with the test image
        #response = requests.post(self.url, headers=self.header, data=self.data)
        response = self.app.post(self.url, headers=self.header, data=self.data)
        # check that the response is 200 OK
        self.assertEqual(response.status_code, 200)
        returnvalue=json.loads(response.get_data(as_text=True))
        print(returnvalue)
        # check that the response data is as expected
        self.assertIn('Happy', returnvalue["expression"])

if __name__ == '__main__':
    unittest.main()
