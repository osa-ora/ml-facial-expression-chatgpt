import unittest
import base64
import requests

class PredictTestCase(unittest.TestCase):

    def setUp(self):
        # set up the test client
        self.url = "http://localhost:5000/predict"
        self.header = {'Content-Type': 'application/json'}
    def test_predict_sad(self):
        with open("test-images/face_sad.jpg", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        print("Test Sad Expression")
        self.data = '{"image": "' + encoded_string + '"}'
        # send a POST request to the predict endpoint with the test image
        response = requests.post(self.url, headers=self.header, data=self.data)

        # check that the response is 200 OK
        self.assertEqual(response.status_code, 200)
        print(response.json())
        # check that the response data is as expected
        self.assertIn("Sad", response.json().get('expression'))
        
    def test_predict_happy(self):
        with open("test-images/face_happy.png", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        print("Test Happy Expression")
        self.data = '{"image": "' + encoded_string + '"}'
        # send a POST request to the predict endpoint with the test image
        response = requests.post(self.url, headers=self.header, data=self.data)

        # check that the response is 200 OK
        self.assertEqual(response.status_code, 200)
        print(response.json())
        # check that the response data is as expected
        self.assertIn("Happy", response.json().get('expression'))

if __name__ == '__main__':
    unittest.main()
