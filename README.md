Red Hat OpenShift Data Science (RHODS) is an easy-to-configure MLOps platform for building and deploying AI/ML models and intelligent applications. In this demo we will go through how to use it to build and deploy ML applications easily.

The project is detecting the facial expression for images with size 48x48 & grayscale, here we will demo this application using OpenShift and OpenShift Data Science showing how efficient we can use OpenShift in handling the different milestones in ML application.


## 1) Get The Environment

We need an OpenShift cluster to use, the easiest way is to utilize the Red Hat OpenShift Developer Sandbox which gives us a free trial of OpenShift and it includes access to OpenShift Data Science.

Go to: https://developers.redhat.com/developer-sandbox and register for the sandbox.

<img width="1407" alt="Screen Shot 2023-01-27 at 17 52 04" src="https://user-images.githubusercontent.com/18471537/215103044-31e01e34-ee65-4c82-adc3-e87c8955af81.png">

Open the sandbox to land on OpenShift console, then navigate to the Data Science as in the following picture:

<img width="1340" alt="Screen Shot 2023-01-27 at 17 55 04" src="https://user-images.githubusercontent.com/18471537/215103713-50ea46ea-ba35-411d-827d-804b1fc232cb.png">

After authentication and authorization, you will see Red Hat OpenShift Data Science Console.

<img width="733" alt="Screen Shot 2023-01-27 at 17 56 20" src="https://user-images.githubusercontent.com/18471537/215103960-051c0644-fa5b-4db3-bb56-274997c11206.png">

## 2) Create Jupyter Notebook Server

From the console, lunch the Jupyter notebook application and let's create a notebook to work on.

As you can see the notebook already come with different notebook server template to use that's optimized and libraries are pre-installed so we can pick what we need or we can build our own custom template.

<img width="399" alt="Screen Shot 2023-01-27 at 17 59 12" src="https://user-images.githubusercontent.com/18471537/215104574-9890d109-713a-4d65-956d-bdb866c402ba.png">

Select options as above Tensor Flow and small as this is a sandbox environment. Once the server is started, you need to check our the code by going to the Git section and pick clone the code

<img width="364" alt="Screen Shot 2023-01-27 at 18 02 03" src="https://user-images.githubusercontent.com/18471537/215105180-3a68a8a3-a200-49b1-9b3c-c3811d9e0223.png">

<img width="289" alt="Screen Shot 2023-01-27 at 18 03 26" src="https://user-images.githubusercontent.com/18471537/215105503-4625ef86-99f8-4164-b00e-6ceef0a0207c.png">
Clone our git repository: https://github.com/osa-ora/ml-facial-expression-chatgpt

## 3) ML Development & Testing 

You can now run the Notebook using the file : model-build.ipynb which contains all the logic to download the dataset (from kaggle.com), split it to test and train datasets, build the model, train it and save the model.

<img width="1142" alt="Screen Shot 2023-01-31 at 13 15 37" src="https://user-images.githubusercontent.com/18471537/215718404-7e43b03e-0950-42cc-b46d-d23ced00ccf7.png">

To run the model, you can just run the app.py file which will listen to port 8080, now you can execute the test-app.py 
<img width="871" alt="Screen Shot 2023-01-31 at 13 18 40" src="https://user-images.githubusercontent.com/18471537/215719155-a7261bd2-9bfe-4223-800e-af885875439e.png">

if you get an error that flask module is not recognized, you can just execute "pip install flask" from a terminal.

Another alternative to test the application is to run the unit-test.py from the terminal.

<img width="868" alt="Screen Shot 2023-01-31 at 13 20 53" src="https://user-images.githubusercontent.com/18471537/215719643-84349afb-1c38-4d34-bb6f-789581be07b2.png">









