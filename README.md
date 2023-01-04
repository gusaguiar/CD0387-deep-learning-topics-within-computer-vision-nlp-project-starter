# Image Classification using AWS SageMaker

This project uses AWS Sagemaker to train a pretrained model for image classification using the dog breed classification dataset. The project demonstrates good machine learning engineering practices, including the use of Sagemaker profiling, debugger, and hyperparameter tuning.

## Requirements

To run this project, you will need the following:

- An AWS account
- The AWS CLI and AWS SDK for Python (Boto3) installed on your machine
- A Sagemaker notebook instance
- The dog breed classification dataset
- Setup
- Clone this repository to your local machine
- Create a Sagemaker notebook instance
- Follow the instructions in the notebook to train and deploy the image classification model

## Dataset

The dog breed classification dataset is a dataset used for training machine learning models to classify images of dogs into different breeds. The dataset typically includes a large number of images of dogs, each labeled with the breed of the dog depicted in the image. The goal of using the dataset is to train a machine learning model to recognize the characteristics of different dog breeds and use this knowledge to classify new images of dogs into the appropriate breed.

The dog breed classification dataset is often used for a variety of purposes, such as research, education, and product development. For example, researchers may use the dataset to study the features and patterns that are characteristic of different dog breeds, or to develop machine learning algorithms that can accurately classify dog breeds. Educators may use the dataset to teach students about machine learning and image classification, or to build hands-on projects that demonstrate the capabilities of machine learning. Product developers may use the dataset to build applications or systems that can classify dog breeds, such as a web or mobile app that can identify the breed of a dog based on a user-provided image.

Overall, the dog breed classification dataset is an important resource for anyone interested in machine learning, image classification, or the classification of animals. It provides a rich and diverse set of data that can be used to train and evaluate machine learning models, and can help researchers, educators, and product developers develop new and innovative solutions to real-world problems.

## Hyperparameter Tuning

Choosing a pre-trained Resnet18 model and using transfer learning to train the model on a specific dataset (such as the dog breed classification dataset) is a good choice because it allows you to leverage the knowledge and capabilities of a well-known and widely-used model, while still being able to fine-tune the model to the specific characteristics and patterns of the dataset. This can help you achieve good results more quickly and with less data, and is a common approach in many machine learning applications. There are several reasons why choosing a pre-trained Resnet18 model and using transfer learning to train the model on a specific dataset (such as the dog breed classification dataset) might be a good choice for classifying dog breeds:

- Pre-trained models are trained on large and diverse datasets, which can help them learn general features and patterns that are useful for a wide range of tasks. By using a pre-trained model as a starting point, you can leverage the knowledge and capabilities that the model has already learned, which can help you achieve good results more quickly and with less data.

- Transfer learning can be an effective way to fine-tune a pre-trained model for a specific task or dataset. By training the model on a new dataset, you can adapt the model to the specific characteristics and patterns of that dataset, which can help improve the model's performance.

- The Resnet18 model is a well-known and widely-used model that has achieved good results on a variety of image classification tasks. It is a good choice for many tasks because it is relatively lightweight and efficient, yet still has good performance.

To further optimize the performance of the model, we used Sagemaker's hyperparameter tuning capabilities to search for the optimal combination of hyperparameters for the model. This included:

- Defining a range of values for each hyperparameter to be tuned

- Running multiple training jobs with different combinations of hyperparameters

- Using the Sagemaker Tuner to select the best combination of hyperparameters based on the results of the training jobs

Remember that your README should:
- Include a screenshot of completed training jobs
- Logs metrics during the training process
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs

## Debugging and Profiling

To improve the performance and accuracy of the model, we used Sagemaker profiling and debugging tools to identify and address bottlenecks and issues in the training process. This included:

- Using the Sagemaker Profiler to identify and optimize CPU and memory utilization
- Using the Sagemaker Debugger to identify and fix issues with the model's training and evaluation processes

To perform and evaluate the results of model debugging and profiling in Sagemaker, you can follow these steps:

1. Set up the Sagemaker debugger and profiler. To do this, you will need to specify the model and the training data that you want to use for debugging and profiling. You can also specify any additional parameters or configurations, such as the instance type or the data processing options.

2. Start the training job using the Sagemaker Python SDK. This will initiate the training process and enable the debugger and profiler to collect data and identify issues with the model's training and evaluation processes.

3. Monitor the progress of the training job using the Sagemaker Python SDK or the AWS Management Console. You can check the status of the job and view the output of the debugger and profiler as it becomes available.

4. Analyze the results of the debugging and profiling to identify and address any issues with the model's training and evaluation processes. You can do this by examining the output of the debugger and profiler, and using this information to make changes to the model or the training data.

5. Evaluate the results of the debugging and profiling by comparing the performance of the model before and after the debugging and profiling process. You can do this by measuring metrics such as accuracy, precision, and recall, and comparing these metrics to determine whether the debugging and profiling process has improved the model's performance.

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

**TODO** Remember to provide the profiler html/pdf file in your submission.

## Model Deployment

Once the model was trained and optimized, we deployed it to a Sagemaker endpoint for real-time inference. The endpoint can be used to classify new images and make predictions about dogs breeds.

**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.

## Standout Suggestions

### Package the model as a Docker Container
To package a model trained on AWS Sagemaker as a Docker container, you can follow these steps:

1. Install Docker on your local machine. You can download Docker from the official website (https://www.docker.com/) and follow the instructions to install it.

2. Create a Dockerfile that specifies the instructions for building the Docker image. The Dockerfile should include instructions for installing any dependencies that are required to run the model, as well as instructions for copying the model artifacts (such as the model's trained weights and biases) into the image.

3. Build the Docker image using the Dockerfile. To do this, navigate to the directory where the Dockerfile is located, and run the following command:

`docker build -t <image_name>`

4. Test the Docker image to ensure that it is working correctly. To do this, run the following command:

`docker run -p 5000:5000 <image_name>`

5. Push the Docker image to a Docker registry, such as Docker Hub or AWS ECR. This will make it easy to deploy the image to other environments, such as production servers or Kubernetes clusters.

6. To deploy the Docker image to a production server or Kubernetes cluster, you can use a tool like Docker Compose or Helm to manage the deployment process. This will allow you to easily scale the deployment and manage the underlying infrastructure.

Overall, packaging a model trained on AWS Sagemaker as a Docker container is a good way to make the model portable and easy to deploy to a variety of environments. By following these steps, you can create a Docker image that contains the model and all of the necessary dependencies, and then deploy the image to a production server or Kubernetes cluster for real-time inference.


### Batch Transform

To create a batch transform that performs inference on the whole test set using a model that you trained on AWS Sagemaker, you can follow these steps:

1. Create an S3 bucket to store the test set and the output of the batch transform. You can do this using the AWS Management Console or the AWS CLI.
2. Upload the test set to the S3 bucket. You can do this using the AWS Management Console or the AWS CLI.
3. Create a batch transform job using the AWS Sagemaker Python SDK. To do this, you will need to specify the following parameters:
4. The S3 bucket and prefix where the test set is stored
5. The S3 bucket and prefix where the output of the batch transform should be stored
6. he Amazon SageMaker model to use for the batch transform
7. The number of instances to use for the batch transform
8. Any additional parameters or configurations for the batch transform, such as the instance type or the data processing options
9. Start the batch transform job using the AWS Sagemaker Python SDK. This will initiate the batch transform process and perform inference on the whole test set using the specified model.
10. Monitor the progress of the batch transform job using the AWS Sagemaker Python SDK or the AWS Management Console. You can check the status of the job and view the output of the batch transform as it becomes available.

Overall, creating a batch transform that performs inference on the whole test set using a model that you trained on AWS Sagemaker is a good way to quickly and efficiently perform inference on large datasets. By following these steps, you can use the AWS Sagemaker Python SDK to create and manage batch transform jobs, and monitor their progress as they run.

### Model Explainability

Amazon Sagemaker Clarity is a tool that can be used to make machine learning models trained on AWS Sagemaker more interpretable. To use Amazon Sagemaker Clarity to make a model more interpretable, you can follow these steps:

1. Install the Amazon Sagemaker Clarity Python library. You can do this by running the following command:

`pip install sagemaker-clarity`

2. Import the Amazon Sagemaker Clarity library and create a Clarity object. To do this, you will need to specify the name of the Amazon SageMaker model that you want to make more interpretable.

`from sagemaker_clarity import Clarity

clarity = Clarity(model_name='<model_name>')`

3. Use the Clarity object to generate explanations for the model's predictions. You can do this by calling the explain method and specifying the input data that you want to explain. The explain method will return a list of explanations, each of which includes information about the features and values that contributed to the model's prediction.

`explanations = clarity.explain(input_data)`

4. Visualize the explanations using the plot method. This will generate a bar chart that shows the relative contributions of each feature to the model's prediction.

`clarity.plot(explanations)`

5. Overall, using Amazon Sagemaker Clarity is a good way to make a machine learning model trained on AWS Sagemaker more interpretable. By following these steps, you can use the Amazon Sagemaker Clarity library to generate explanations for the model's predictions and visualize the contributions of different features to those predictions. This can help you understand how the

