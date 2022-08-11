# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 15:28:35 2022

    This is a file of Random Fores built-in Algorithm in AWS.
    
@author: J. Ivan Avalos-Lopez
"""

import numpy as np
import pandas as pd

import re

from botocore.client import ClientError
import boto3

import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.serializers import CSVSerializer


# Setting aws credentials
# Do not expose your credentials on internet
profile_name = "-"
aws_access_key_id = "-"
aws_secret_access_key = "-"
region_name = "-"

# Execution role for SageMaker tasks
role_execution = "-"



# Setting data path 
datapath = "./data/"


# Creating a session
boto_session = boto3.Session(aws_access_key_id=aws_access_key_id,
 	                         aws_secret_access_key=aws_secret_access_key,
 	                         region_name=region_name,
 	                         profile_name=profile_name)


# Setting bucket name, model, folders and objects
bucket_name = "rf-built-in-algorithm" 

trainingValidation_folder = r'titanicFromDisaster/trainingValidation/'
test_folder = r'titanicFromDisaster/test/'

# Location in s3
s3_model_output_location = r's3://{0}/titanicFromDisaster/model'.format(bucket_name)
s3_trainingValidation_file_location = r's3://{0}/{1}'.format(bucket_name,trainingValidation_folder)
s3_test_file_location = r's3://{0}/{1}'.format(bucket_name,test_folder)


# SageMaker settings
job_train_name = "rf-scikit"
FRAMEWORK_VERSION = "0.23-1" # sklearn version


def create_bucket(client, bucket_name):
    """
        Function that create a bucket with public access block on

    Parameters
    ----------
    client : boto3 client s3
    
    bucket_name : string

    Returns
    -------
    None.

    """
    
    # List all the buckets
    bucket_names = [buckets["Name"] for buckets in client.list_buckets()["Buckets"]]
    
    if bucket_name not in bucket_names:
    	try:
    		# Create the bucket
    		client.create_bucket(Bucket = bucket_name)
    		# block public access - ON
    		client.put_public_access_block(
            Bucket=bucket_name,
            PublicAccessBlockConfiguration={
    	            'BlockPublicAcls': True,
    	            'IgnorePublicAcls': True,
    	            'BlockPublicPolicy': True,
    	            'RestrictPublicBuckets': True
    	        },
    	    )
    		print("{} created".format(bucket_name))
    	except ClientError as e:
    		print(e)
    else:
    	print("{} already exists".format(bucket_name))


def write_to_s3(client, bucket_name, folder, file, data_Path):
    """
        Function to upload files in a bucket

    Parameters
    ----------
    client : boto3 client s3
    bucket_name : string
        Name of the bucket
    folder : string
        Name of the folder
    file : string
        Name of the file
     file : data_path
         Folder of the file

    Returns
    -------
    None.

    """
    
    # list files
    files = client.list_objects_v2(Bucket = bucket_name).get("Contents")

    key = folder  + file
    
    
    try:
        if files is None:
            # Writting a file
            with open(data_Path + file, "rb") as f:
                client.put_object(Body = f,
    		                      Bucket = bucket_name,
    		                      Key = key)
                print("{} created".format(key))
        else:
            file_names = [file["Key"] for file in files]
            if key not in file_names:
                # Writting a file
                with open(data_Path + file, "rb") as f:
                    client.put_object(Body = f,
        		                      Bucket = bucket_name,
        		                      Key = key)
                    print("{} created".format(key))
            else:
                print("{} already created".format(key))
                
            
    except ClientError as e:
        print(e)
    
    


if __name__ == "__main__":
    # Creating a s3 client
    s3_client = boto_session.client("s3")
    
    # Creating the bucket
    create_bucket(s3_client, bucket_name)
    
    
    # uploading training, validation and test files
    trainingValidation_file = "trainValidation_clean.csv"
    write_to_s3(s3_client, bucket_name, trainingValidation_folder, trainingValidation_file, datapath)
    test_file = "test_clean.csv"
    write_to_s3(s3_client, bucket_name, test_folder, test_file, datapath)
    
    
    # Checking s3 paths
    print(s3_model_output_location)
    print(s3_trainingValidation_file_location)
    print(s3_test_file_location)
    
    
    # Creating a training job with Python sdk
    sm_boto3 = boto_session.client("sagemaker") # SageMaker client

    sagemaker_session = sagemaker.Session(boto_session=boto_session,
                         sagemaker_client=sm_boto3) # SageMaker session
    
    FRAMEWORK_VERSION = "0.23-1" # sklearn version
    
    
    # Defining the estimator
    sklearn_estimator = SKLearn(
        entry_point="script_rfClassifier.py",
        sagemaker_session = sagemaker_session,
        role=role_execution,
        instance_count=1,
        instance_type="ml.c5.xlarge",
        framework_version=FRAMEWORK_VERSION,
        base_job_name=job_train_name,
        output_path= s3_model_output_location,
        metric_definitions=[{"Name": "Accuracy", "Regex": "^Random Forest 3 : [TrainValidTest]+ ACCU : 0\.[0-9]+$"}]
    )
    
    # Launching a training job
    # launching training job, with asynchronous call
    trainpath = s3_trainingValidation_file_location + trainingValidation_file
    testpath = s3_test_file_location + test_file
    
    sklearn_estimator.fit({"train": trainpath, "test": testpath}, wait=True, logs = True)
    
    
    # Check logs of the last training job
    print(sklearn_estimator.latest_training_job.wait(logs="None"))
    
    # Getting th model artifact s3 path
    artifact = sm_boto3.describe_training_job(
                TrainingJobName=sklearn_estimator.latest_training_job.name
            )["ModelArtifacts"]["S3ModelArtifacts"]
    print("Model artifact persisted at " + artifact)
    
    
    # Define a SKLearnModel to create an endpoint 
    model = SKLearnModel(
            model_data=artifact,
            role=role_execution,
            sagemaker_session = sagemaker_session,
            entry_point="script_rfClassifier.py",
            framework_version=FRAMEWORK_VERSION,
        )
    
    # Deploy our model (model - endpoint configuration - endpoint)
    predictor = model.deploy(instance_type="ml.c5.large", initial_instance_count=1)
    
    # the SKLearnPredictor does the serialization from pandas for us
    predictor.serializer = CSVSerializer()
    x_test = np.array([[3,1,34.5,0,7.8292,1,1,0,0],[1,0,23.0,0,82.2667,2,2,0,0]])
    print(predictor.predict(x_test))
    
    sm_boto3.delete_endpoint(EndpointName=sm_boto3.list_endpoints()['Endpoints'][0]['EndpointName'])
