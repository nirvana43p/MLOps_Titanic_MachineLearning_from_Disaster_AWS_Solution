# MLOps - Titanic MachineLearning from Disaster (AWS Solution)

 A solution to build, train and deploy a random forest classifier in aws sagemaker, automating all the infrastructure with cloudformation.

## Task

 Build a complete solution to train and deploy an ML model on Amazon Web Services (AWS).

The services used were:

- Amazon SageMaker (for training and deploying an ML model)
- AWS Lambda (To invoke the endpoint and process requests)
- Amazon API Gateway (to create a Rest API and fully integrate with lambda and the ML endpoint)
- AWS CloudFormation (to automate the infrastructure deployment step)
- AWS SDK and CLI (to automate various tasks in the cloud)


## Task

Cloud services have had a great impact on IT (information technology) in the last 10 years, due to the flexibility, scalability, reliability and security of their services. In this project, a complete architecture is proposed to train and deploy a Machine Learning model using AWS as a cloud provider. The model training was done by creating a job in SageMaker using a pre-built container Image (in this case, a scikit-learn image), and its deployment was done by creating an endpoint from a configuration endpoint. The deployment infrastructure was automated with cloudFormation and the training infrastructure was automated with the python SDK.


