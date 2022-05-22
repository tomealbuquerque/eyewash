# Hackathon Retail 4.0 | EyeWash Team
Deliverable of the EyeWash Team to the Hackathon Retail 4.0 "Challenge #1: Car Wash services proposal based on AI, weather rules and computer vision".

Installation Instructions available [here](/project_description/docs/installation_instructions.md).

![Dashboard](/project_description/images/main_image.png)

# 1. Delimitation about the problem
## The team vision about the challenge selected and why the challenge was selected by the team
All the team members are passionate about driving their cars through the beautiful roads of Portugal. Hence, they consider that it is of utmost importance to daily track the maintenance status of their vehicles. When they knew about this hackathon, they realized that it was an interesting opportunity to put their data science skills into practice, while solving a challenge that will ultimately influence their lives as car drivers. Moreover, as we searched about the topic, we understood that car washing impacts several dimensions of the car's longevity. For instance, besides of just keeping our cars in a good look, car washing allows us to prevent car damage [1]:
 - Exposure to sun rays can blemish the paint;
 - Leaf stains can gradually cause damage to the paint of your car;
 - Bird droppings, pollen and various insects may cause chemical reactions and may even eat through the paint
 - Salt can cause erosive reactions and therefore lead to rust.
 
 On the other hand, as sustainability is, more than ever, an important variable of our daily routines, it is more likely that we will do the washing of our vehicles in an automatic washing station, instead of doing it ourselves at home. Therefore it is fundamental that the providers of this type of services (i.e., Galp) are aware of their market to design offers that best fit the needs of their possible customers. Assuming that we are able to gather the right information, we may come up with a solution based on artificial intelligence (AI) to optimize this process, thus maximizing the margins of the providers and the satisfaction of customers.

# 2. Hypothesis of solution
## Detailed description about how the challenge can be solved in the team perspective
To solve this problem, we need to identify an attribute that easily relates with the likelihood of a user to buy a car washing service. The most intuitive attribute is, indeed, the degree of dirtiness of cars. Assuming that we can train an AI model that is capable of extracting the degree of dirtiness from an image of a car (e.g., photograph, video-frame), the main goal would be to convert this insight into a relevant car-washing offer. For instance, Galp has four different offers, i.e., “simple”, “super”, “special” and “extra” [2]. Hence, besides detecting the degree of dirtiness of the car, it is necessary to decide whether the type of offer will be better suited for the user. To answer this question, a nice-to-have feature of the solution would be an AI model that can automatically detect the brand/model of the user’s car. The intuition here would be to use this information along with the degree of dirtiness of the car to come up with a (tentative) personalized recommendation for the user. On the other hand, besides working as a kind of recommendation system, the final solution would work as a means of gathering customer data (this process would have to be aligned with the General Data Protection Regulation of the European Union: EU-GDPR). Why gather customer data? If Galp is able to populate a database of the customers that bought a car washing service along with their car brand models and the type of car washing offer that they chose, at long term, it would be possible to devise a recommendation system that would recommend a special type of offer for the customer, based on their car brand and model.

# 3. Team members
## Who and how the team organize ours roles and responsibilities
[Paulo Maia](https://www.linkedin.com/in/paulo-maia-410874119/) - Team Captain, Project Manager, Full Stack Development and Data Science

[Tiago Fontes](https://www.linkedin.com/in/tiago-rf-fontes/) - Full Stack Developer and Data Science

[Tiago Gonçalves](https://www.linkedin.com/in/tiagofilipegoncalves/) - Computer Vision and Data Science

[Tiago Vieira](https://www.linkedin.com/in/tiagovmvieira/) - Full Stack Developer

[Tomé Albuquerque](https://www.linkedin.com/in/tome-albuquerque-ba087170/) - Computer Vision and Data Science



# 4. Team strategy
## Detailed description about the plan of resolution of the challenge

![Team Strategy](/project_description/images/team_strategy.png)

We addressed this challenge using a “divide and conquer” strategy, i.e., we tackled the problem by dividing it into smaller issues. The plan of resolution of the challenge consisted of developing:
 - AI-based models - we approached this task from four different points of view:
  
  - Car Detection Model - assuming that Galp will use conventional cameras to detect the cars of their customers, it is of utmost importance that we have a model that is able to detect the cars we want to address. Therefore, we ended up using a pre-trained model (YOLOv4 [3]) to detect and crop the cars from an image.
  
  - Dirtiness Level Detection Model - after the detection of the car, we have to understand if it is dirty or not, so that we can propose a service of car washing. Hence, we trained a simple convolutional neural network (CNN) named MobileNet_v2 [4] from Pytorch Library with our in-house dirtiness level detection data set (see Section 6). Our model outputs a probability that is used as the dirtiness level of the car (i.e., the higher the probability, the higher the dirtiness level of the model). To follow the training progress of the model we used Weights & Biases tool as the next figure shows. One of the most prominent ways of producing a class-specific heatmap in a Convolutional Neural Network (CNN) model is using Gradient-Weighted Class Activation Mapping (Grad-CAM). It uses the gradients of any target concept (e.g., "dirty car" or "clean car") in a classification network to build a coarse localization map highlighting the essential regions in the image for predicting the concept. Grad-CAM requires three things: an input image, a trained Convolutional Neural Network (CNN), and a particular class of interest. Any CNN model with differentiable layers can be used to calculate Grad-CAM. This way,  we also added a GradCAM module to this model to highlight the locations of the image where the model detected dirtiness with higher confidence; this result is converted into an heatmap that can be placed on top of the original input image.
  ![Weights and Biases](/project_description/images/weights_biases.png)
  
  - Car Model Detection Model - besides detecting the dirtiness level of the cars, we find it insightful to have a record of the car model and brand (see Section 7 to understand why this is a nice-to-have feature). To do this, we trained a ResNet50 deep neural network [5] on the Stanford Cars Dataset (see Section 6). The output of this model is a string with the car brand and model.

  - Decision Process Pipeline - this module wraps up the complete pipeline. It receives an input image, detects all the cars in this image, crops these cars into smaller images, computes the car dirtiness level detection and predicts the car model and brand.
 
 - Application Programming Interface (API) - to facilitate the deployment of our AI-based models, we decided to build an API. The referred API is a RESTful based API, thus makes use of a representational state transfer (REST) architectural style and allows for interaction with RESTful web services. REST APIs work using ‘request’ and ‘responses’. When an API requests information from a web application or web server, it will receive a response. In order to deploy the AI-based models, a few endpoints were implemented:
  
  - Car Detection Model - The purpose of this endpoint is to detect the car on an ingested picture, thus sending a response containing the coordinates of the detected cars in a bounding box format. In order to achieve this, this endpoint uses the YOLOv4 model that was developed. This endpoint consists in a POST request, providing on the BODY an image  file on one of the formats jpg, jpeg, or png.
  ![POST Car Detection](/project_description/images/car_detection_post.png)
  
  The endpoint returns an array with n arrays containing the coordinates of the detected car on the image, being n the number of cars detected.
  ![POST Car Detection Array](/project_description/images/car_detection_array.png)

  - Car Model Detection Model - The purpose of this endpoint is to detect the brand model on an ingested picture which contains a unique car, so that a crop operation was previously performed. In order to achieve this, this endpoint uses the ResNet50 deep neural network that was developed. This endpoint consists in a POST request, providing on the BODY an image file on one of the formats jpg, jpeg, or png.
  ![POST Car Model Detection](/project_description/images/car_model_detection.png)
  
  The endpoint returns a dictionary containing two keys, the predicted class of brand models, and the probability associated.
  ![POST Car Model Detection Dict](/project_description/images/car_model_detection_dict.png)

  - Dirtiness Level Detection Model - The purpose of this endpoint is to evaluate whether an image is clean or dirty, based on a threshold limit. In order to achieve this, this endpoint uses a simple convolutional neural network (GradCAM). This endpoint consists in a POST request, providing on the body an image file on one of the formats jpg, jpeg, or png.
  ![POST Dirtiness Level Detection Model](/project_description/images/dirtiness_level_detection.png)
  The endpoint returns a dictionary containing three keys, the predicted class of dirtiness, probability or dirtiness associated, and the activation map path that contains the path to an heatmap map which measures the level of dirtiness of an ingested image.
  ![POST Dirtiness Level Detection Model Dict](/project_description/images/dirtiness_level_detection_dict.png)

  - Decision Process Pipeline - The purpose of this endpoint is to establish the decision process of the project. This endpoint consists in a POST request, providing on the body an image file on one of the formats jpg, jpeg, or png.
  ![POST Decision Process Pipeline](/project_description/images/decision_process.png)
  
  The endpoint returns an array with n dictionaries, containing the overall data about each car detected on the image, being n the number of cars detected. The referred data includes the dirty class predicted and the correspondent probability, path to the activation map path, recommended washing program, brand model prediction and the correspondent probability, the bounding box, and lastly the timestamp of the process. The overall data is also posted on a MongoDB database.
  ![POST Decision Process Pipeline Dict](/project_description/images/decision_process_dict.png)

 - Dashboard - to facilitate the use of our AI-based models, we decided to build an interactive dashboard that allows the end-user to test and interact with our algorithms (via the Eyewash API). The dashboard has five different modes/tabs:
  - Main - this tab contains visual examples of a clean and dirty car on different photographs, and video that shows our dirtiness algorithm working in real-time.
  - Car Detection - this tab allows the user to interact with the “Car Detection Model” and visualize the outputs of this module.
  - Dirtiness Level Detection - this tab allows the user to interact with the “Dirtiness Level Detection Model” and visualize the outputs of this module.
  - Car Model Detection - this tab allows the user to interact with the “Car Model Detection Model” and visualize the outputs of this module.
  - Decision Process Pipeline - this tab allows the user to interact with the whole decision process pipeline algorithm.
  ![Dashboard API Scheme](/project_description/images/eyewash_api_dashboard.png)

 - MongoDB Database - as previously mentioned, at the end of the decision process endpoint, the overall data resulting from the AI-based models is stored in a single MongoDB collection, thus establishing a Database to the process. The choice fell on this NoSQL database once the overall data is returned on dictionaries from the AI-based models, a data structure similar to a JSON file, which is the base object of this mapped based database. Another point is this database capability of being fully managed in a cloud service.

 - Repository - the repository has the following structure:
  - “app” - this directory contains all the files related to the development of the APIs for our AI-based models;
  - “car_model_classifier” - this directory contains all the files related to the “Car Model Detection” AI model, including data processing files, model training files and model inference files;
  - “clean_dirty_cars_classifier” - this directory contains all the files related to the “Dirtiness Level Detection” AI model, including data processing files, model training files, model inference files and heatmap (GradCAM) generation files;
  - “dashboard” - this directory contains all the development files related to the Streamlit dashboard that we developed for the final demo;
  - “detect_cars” - this directory contains all the files related to the “Car Detection” AI model, including data (i.e., images), data processing files, model training files and model inference files;
  - “eyewash-terraform” - this directory contains all the Terraform files necessary to build a virtual machine (VM) at the DigitalOcean [6] cloud service with the minimum requirements to deploy our demo application;
  - “mongodb” - this directory contains a sub-directory “database” that acts has the storage path for the MongoDB Database we built for this demo application;
  - “shared” - this directory works as a shared directory between all the Docker containers that compose our demo application, including the “tmp” sub-directory that allows us to write and read temporary files that may be created during the usage of the dashboard and the APIs.


# 5. Technologic prerequisites
## What kind of technological tools are needed for the solution (what Galp needs to validate the functionality of the deliverable), all the tools used should be free of charges to enable Galp to evaluate, if some tools aren't available the project will be disqualified.
![Scheme of the Final Application](/project_description/images/final_app_scheme.png)
The final solution was deployed using Docker containers, to guarantee reproducibility between different systems and to facilitate cloud deployment. We recommend having a free Weights&Bias account to track model training.

The deployment instructions are available [here](/project_description/docs/deployment_instructions.md).

Regarding the computational infrastructure, we refer the jury to the characteristics of the Virtual Machine we built at the DigitalOcean platform, using Terraform:
 - Operating System (OS): Ubuntu 18.04 (LTS) x64
 - RAM: 8GB
 - Storage: 160 GB


# 6. Used data sets
## What data sets are used and where they can be picked (attention to EULA and legal use about the data, ilegal data set will be disqualified)
There are two use cases which required the build and gathering of different data sets:
 - Dirtiness Level Detection - to build an artificial intelligence model able to detect dirtiness, we created a web-scraper that searched for the terms “dirty car; dusty car; carro sujo; carro com poeira”. This in-house data set has 145 images with cars with different dirtiness levels and 500 random car images from Stanford cars dataset.
 - Car Model Detection - to build an artificial intelligence model to detect car brands and models, we used the Stanford Cars Dataset [7]. This data set contains 16,185 images of 196 classes of cars, which are typically at the level of Make, Model, Year ( e.g. 2012 Tesla Model S or 2012 BMW M3 coupe). This data set first appeared in the paper “3D Object Representations for Fine-Grained Categorization” by Jonathan Krause, Michael Stark, Jia Deng, Li Fei-Fei [8].


# 7. How the solution can be used
## A manualized application of the final product with use examples.
This prototype assumes that Galp has an image acquisition infrastructure (e.g., cameras). To test the actionability of this minimum viable product, we would like to run a test-pilot during a year. During the two/three months of this test-pilot, the car drivers would enter a Galp station and a sequence of events would happen:
 1. Detection of the car;
 2. Detection of the level of dirtiness;
 3. Car Washing Service Recommendation:
    - 30% of probability of getting a random recommendation;
    - 70% of probability of getting a recommendation based on the dirtiness level:
        - Between 0 and 0.2 Dirtiness Score: Nothing
        - Between 0.2 and 0.4 Dirtiness Score: Simple Program
        - Between 0.4 and 0.6 Dirtiness: Super Program
        - Between 0.6 and 0.8 Dirtiness: Special Program
        - Between 0.8 and 1.0 Dirtiness: Extra Program
 4. The user accepts, not accepts or selects a different car washing service;
 5. The car brand and model is detected;
 6. The level of dirtiness, the choice of the user and the car brand and model are saved into the database.

After this first phase, the main idea is to use the information about the choices of the users and their car brand and models to train a car washing recommendation system model based on this information. Afterwards, the sequence of events would be:
 1. Detection of the car;
 2. Detection of the level of dirtiness;
 3. Car Washing Service Recommendation: 30% of probability of getting a random recommendation, 70% of probability of getting a recommendation based on the dirtiness level and previous user’s choices (e.g., “the users with your car brand preferred…”).
 4. The user accepts, not accepts or selects a different car washing service;
 5. The car brand and model is detected;
 6. The level of dirtiness, the choice of the user and the car brand and model are saved into the database;
 7. The recommendation system model is updated.


# 8. Metrics and results
## How the team evaluates the solution delivered and how the metrics can be afforded
In a few weeks, the team was able to build an in-house database with different images of cars with variable dirtiness levels. We trained an AI-based model to automatically compute the level of dirtiness of a car that attained almost 100% accuracy (see confusion-matrix below). Moreover, we used state-of-the-art AI-based algorithms to perform car detection and car model and brand detection from high-resolution images. We built and deployed a final demo application based on a dashboard that allows the end-user to interact and test the different steps of the entire decision process pipeline we are proposing Galp to implement for a year in a test-pilot. We assured that the initial tests are completely free and that the amount of resources needed to perform the pilot-test is relatively low and cheap. Hence, we believe we are delivering a completely actionable solution that may impact the business model paradigm of Galp regarding the car-washing stations of the future.
![Metrics Training](/project_description/images/metrics_training.png)

![Confusion Matrix Training](/project_description/images/confusion_matrix_training.png)