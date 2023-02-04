Red Hat OpenShift Data Science (RHODS) is an easy-to-configure MLOps platform for building and deploying AI/ML models and intelligent applications. In this demo we will go through how to use it to build and deploy ML applications easily.

The project aim to detect the facial expression for images with size 48x48 & in grayscale, here we will demo this application using OpenShift and OpenShift Data Science showing how efficient and easily we can use OpenShift in handling the different milestones in ML application.

This ML project built by working with ChatGPT which gives the core ML codes and I enrich it with some features and additional files. This is how the ML is changing the way we used to work even in building applications.  

<img width="822" alt="Screen Shot 2023-01-31 at 16 00 09" src="https://user-images.githubusercontent.com/18471537/215754143-bc5e54cb-9b1f-469e-ac5b-3d840d3f9889.png">

It was nice experience to use a ML application to build another ML model, it is quiet simple and it can be also written from the different source codes available all over the internet but it is just funny to use ChatGPT in this project :)

As a typical ML application, it has the following milestones:

![Blog-4-MLOPs-3](https://user-images.githubusercontent.com/18471537/215968348-99943373-6a45-4c12-8745-426f615b3a65.png)

In our demo, we will download data that was collected and labelled (from kaggle.com), then we will move to data exploration and model training and tuning. Finally we will deploy the model and monitor it for further improvements.  

For effieient deployment, we will need to build a pipeline and include some steps like sending notifications, running unit tests and doing sanity or smoke test at the end to make sure everything working as planned.  

It can be further improved by using GitOps to handle the deployment.

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

### A) Using Source2Image
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

One note regarding Python source2Image, for the simple Python application you can just name the entry point for your application as in our case as app.py and it will be the entry point for your container, otherwise you can just set either APP_MODULE, APP_FILE or APP_SCRIPT environment variables pointing to the entry point for your container.

```
(echo -n '{"image": "'; base64 {IMAGE-NAME-HERE}; echo '"}') | curl -H "Content-Type: application/json" -d @- {{ROUTE-URL-HERE}/predict

//if everything is okay it will return response like:
{'expression': 'RESULT'}
```

Now, we can manually run the BuildConfig to build new application version or we can auto-trigger it with new Git commits.

### B) Using OpenShift Pipeline
By using OpenShift Pipeline based on Tekton we can incorporate many options such as notification, testing, security scanning, etc.

The pipelines will do the following:

- Send slack message "Started" (if slack parameter is set to true)
- Git the code from the GitHub repository
- Run the unit tests
- Build the application using source2image
- Deploy the application (for first time pipeline execution)
- Expose a route to the application (for first time pipeline execution)
- Run health check to test the application is running
- Send a final slack message with the pipeline results (if slack parameter is set to true)


```
/login to OpenShift cluster
oc login ...
//download the script
curl https://raw.githubusercontent.com/osa-ora/ml-facial-expression-chatgpt/main/script/init.sh > init.sh
chmod +x init.sh
//execute the script with 2 parameters: the name of "ML" project, and slack channel webhook url (to send notifications)
./init.sh ooransa-dev https://hooks.slack.co...{fill in your slack url here}

```
This will create and execute the pipeline in the specified project. If everything is configured properly, you will see a pipeline like:

<img width="1493" alt="Screen Shot 2023-01-31 at 15 28 47" src="https://user-images.githubusercontent.com/18471537/215748137-d5f22039-6e90-40de-8e36-0a30c576ad53.png">

The pipeline execution will start and hopefully everything is completed successfully.

<img width="1597" alt="Screen Shot 2023-01-31 at 15 55 25" src="https://user-images.githubusercontent.com/18471537/215753306-04a65d5e-b876-4daf-bf75-e5b8e69496ac.png">

Now, we can automate the build and deployment using the tekton pipeline for more efficient ML-OPs.

<img width="195" alt="Screen Shot 2023-01-31 at 13 46 32" src="https://user-images.githubusercontent.com/18471537/215725502-9376eaf9-da40-45cc-990e-babb0fc10c19.png">

Copy the route URL and invoke it with any car images that you have to test the application. (should be either png or jpg)

```
(echo -n '{"image": "'; base64 {IMAGE-NAME-HERE}; echo '"}') | curl -H "Content-Type: application/json" -d @- {{ROUTE-URL-HERE}/predictions

//if everything is okay it will return response like:
{'expression': 'RESULT'}
```

### C) Using OpenShift Pipeline & GitOps (end-to-end)
We can extend now our demo to include also GitOps for deployment, but you need to make sure the OpenShift cluster has the OpenShift GitOps operator available or already installed there. 

The new pipeline script will do the following tasks:

- Send slack message "Started" (if slack parameter is set to true)
- Git the code from the GitHub repository
- Run the unit tests
- Build the application using source2image
- Tag the image (default is dev tag)
- Define the GitOps Configurations for the application
- Run health check to test the application is running
- Send a final slack message with the pipeline results (if slack parameter is set to true)

It should be noted that, GitOps depends on the Git repository as a source of truth for all your application configurations, so you can modify any definition in the Git Repository in the gitops folder (it defines the application components) and cicd/app.yaml which defined the GitOps configurations (including application name and project/namespace).

Prerequisite: 

- OpenShift Pipeline already installed on OpenShift
- OpenShift GitOps already installed on OpenShift with ArgoCD instance is created with privilege to deploy on the target namespace/project
- Login to OpenShift cluster (using a privileged user)


```
/login to OpenShift cluster
oc login ...
//download the script
curl https://raw.githubusercontent.com/osa-ora/ml-facial-expression-chatgpt/main/script/init-gitops.sh > init-gitops.sh
chmod +x init-gitops.sh
//execute the script with 2 parameters: the name of "ML" project, and slack channel webhook url (to send notifications)
./init-gitops.sh ooransa-dev https://hooks.slack.co...{fill in your slack url here}

```
This will create and execute the pipeline in the specified project. If everything is configured properly, you will see a pipeline like:

<img width="1483" alt="Screen Shot 2023-02-02 at 13 55 53" src="https://user-images.githubusercontent.com/18471537/216292559-a868c6ae-19ee-4e6d-bc94-cdf9b1340d9c.png">

The pipeline execution will start and hopefully everything is completed successfully.

<img width="1729" alt="Screen Shot 2023-02-02 at 13 57 12" src="https://user-images.githubusercontent.com/18471537/216292891-94901021-8944-4b2d-9893-203a05c93194.png">

Now, we can automate the build and delegate the deployment to GitOps for more efficient ML-OPs.

If you look at OpenShift GitOps ArgoCD, you can see the application is automatically sync and deployed as per our GitOps configurations:

<img width="1017" alt="Screen Shot 2023-02-02 at 13 58 15" src="https://user-images.githubusercontent.com/18471537/216293100-87da261a-bc0f-4fb5-bff5-82acad061626.png">

Copy the route URL and invoke it with any car images that you have to test the application. (should be either png or jpg)

```
(echo -n '{"image": "'; base64 {IMAGE-NAME-HERE}; echo '"}') | curl -H "Content-Type: application/json" -d @- {{ROUTE-URL-HERE}/predictions

//if everything is okay it will return response like:
{'expression': 'RESULT'}
```

For other useful workshops and materials about Red Hat OpenShift Data Science: 
https://developers.redhat.com/learn/openshift-data-science
