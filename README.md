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

## 4) ML Deployment

Once we have and ready ML application/APIs we need to experience how to deploy them on OpenShift.
You have 2 choices, one is utilizing the Source2Image OpenShift capability and giving all the build process to OpenShift or build a pipeline so you can enrich the lifecycle and automate many parts and integrate many options such as security and others.

### Using Source2Image
Open OpenShift Developer console, select "Add" and then select "from Git" Repository:

<img width="179" alt="Screen Shot 2023-01-27 at 18 41 31" src="https://user-images.githubusercontent.com/18471537/215113469-8ef128f5-bf04-4f1c-8741-11da0d6cb864.png">

Provide the required details as following:
In the Git Repo URL field, enter: "https://github.com/osa-ora/ml-facial-expression-chatgpt"
Select "main" branch
From the Builder Image version list, select "Python 3.8-ubi7"
Resources type “Deployment”
Then click on the create button.

Once created, click on the route to open the root which is mapped to the health check API and we can see the status is ok.

Copy the route URL and invoke it with any facial images (48x48 grayscale, either png or jpg). 

```
(echo -n '{"image": "'; base64 {IMAGE-NAME-HERE}; echo '"}') | curl -H "Content-Type: application/json" -d @- {{ROUTE-URL-HERE}/predict

//if everything is okay it will return response like:
{'expression': 'RESULT'}
```

Now, we can manually run the BuildConfig to build new application version or we can auto-trigger it with new Git commits.

### Using OpenShift Pipeline
By using OpenShift Pipeline based on Tekton we can incorporate many options such as notification, testing, security scanning, etc.

The pipelines will do the following:

Send slack message "Started" (if slack parameter is set to true)
Git the code from the GitHub repository
Run the unit tests
Build the application using source2image
Deploy the application (for first time pipeline execution)
Expose a route to the application (for first time pipeline execution)
Run health check to test the application is running
Send a final slack message with the pipeline results (if slack parameter is set to true)


```
/login to OpenShift cluster
oc login ...
//download the script
curl https://raw.githubusercontent.com/osa-ora/ml-facial-expression-chatgpt/main/script/init.sh > init.sh
chmod +x init.sh
//execute the script with 2 parameters: the name of "ML" project, and slack channel webhook url (to send notifications)
./init.sh ooransa-dev https://hooks.slack.co...{fill in your slack url here}

```
This will create and execute the pipeline in the specified project. If everything is configured properly, you will see something like:

<img width="1493" alt="Screen Shot 2023-01-31 at 15 28 47" src="https://user-images.githubusercontent.com/18471537/215748137-d5f22039-6e90-40de-8e36-0a30c576ad53.png">

The pipeline execution will start and hopefully everything is completed successfully.



Now, we can automate the build and deployment using the tekton pipeline for more efficient ML-OPs.


<img width="195" alt="Screen Shot 2023-01-31 at 13 46 32" src="https://user-images.githubusercontent.com/18471537/215725502-9376eaf9-da40-45cc-990e-babb0fc10c19.png">

Copy the route URL and invoke it with any car images that you have to test the application. (should be either png or jpg)

```
(echo -n '{"image": "'; base64 {IMAGE-NAME-HERE}; echo '"}') | curl -H "Content-Type: application/json" -d @- {{ROUTE-URL-HERE}/predictions

//if everything is okay it will return response like:
{'expression': 'RESULT'}
```

Other useful workshops and materials about Red Hat OpenShift Data Science: 
https://developers.redhat.com/learn/openshift-data-science





