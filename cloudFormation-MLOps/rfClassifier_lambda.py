import boto3
import re
import math
import dateutil
import json
import os

# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
client = boto3.client(service_name='sagemaker-runtime')


def transform_feature_withCode(feature, codification):
    
    """ This is a function to transform sex feature
    
    Args:
        feature (string):
        codification (dict)

    return:
        None
    """
    assert feature in codification.keys(), "{} invalid feature".format(feature)
    return codification[feature]
    
    
def get_IsChildWoman(age, child_age, sex):
    
    """ This is a function to get IsChildWoman feature
    
    Args:
        age (float):
        child_age (float)
        sex (float)
        
    return:
        None
    """
    return 1 if (age <= child_age) | (sex == 0) else 0
    

def transform_name(name, codification):
    
    """ This is a function to perform a transformation of the title based on a codification.
        Specifically, it transforms a name into a number depending of its title name
        For example: 
            Braund, Mr. Owen Harris --> 1
            Futrelle, Mrs. Jacques Heath (Lily May Peel) --> 2
    
    Args:
        title (string): name of title
        codification (dict): the mapping name to code number
    return:
        None
        """
    regex = "\.|".join(list(codification.keys())[:-1])+"\." # get the regex to match titles
        
    title = re.findall(regex,name)
        
    if title:
        title = title[0][0:-1]
        return codification.get(title,0)
    else:
        return codification["Other"]
        

def transform_data(data, codification_title, codification_sex, codification_embarked, features_ml, features_inference):
    try:
        assert len(data) == len(features_inference), "number of feature does not match for inference"
        data_withFeatures = {feature:value for value,feature in zip(data, features_inference)}
        
        # get title name
        name = data_withFeatures["Name"]
        title_code =  transform_name(name, codification_title)
        print("transforming {0} to {1} code".format(name, title_code))
        del data_withFeatures["Name"]
        data_withFeatures["Title_Name"] = title_code
        
        # get Sex codification
        sex = data_withFeatures["Sex"]
        sex_code = transform_feature_withCode(sex, codification_sex)
        print("transforming {0} to {1} code".format(sex, sex_code))
        data_withFeatures["Sex"] = sex_code
        
        
        # get embarked codification
        embarked = data_withFeatures["Embarked"]
        embarked_code = transform_feature_withCode(embarked, codification_embarked)
        print("transforming {0} to {1} code".format(embarked, embarked_code))
        data_withFeatures["Embarked"] = embarked_code
        
        
        # get IsChildWoman
        child_age = 8.25 # According to Titanic_MachineLearning_from_Disaster notebook
        age =  data_withFeatures["Age"]
        IsChildWoman = get_IsChildWoman(age, child_age, sex)
        data_withFeatures["IsChildWoman"] = IsChildWoman
        
        # build sibPar Feature
        data_withFeatures["SibPar"] = data_withFeatures["SibSp"]*data_withFeatures["Parch"] 
        
        # get rid of Cabin, Ticket and SibSp
        del data_withFeatures["Cabin"]
        del data_withFeatures["Ticket"]
        del data_withFeatures["SibSp"]
        
        assert len(data_withFeatures) == len(features_ml), "number of feature does not match ML"
        
        # Returning transformed data
        return ",".join([str(data_withFeatures[feature_ml]) for feature_ml in features_ml])

    except Exception as err:
        print('Error when transforming: {0},{1}'.format(data,err))
        raise Exception('Error when transforming: {0},{1}'.format(data,err))


def lambda_handler(event, context):
    try:
        print("Received event: " + json.dumps(event, indent=2))

        request = json.loads(json.dumps(event))
        
        # Name codification
        codification_title = {"Mr": 1,
                "Mrs" : 2,
                "Miss" : 3,
                "Master" : 4,
                "Rev" : 5,
                "Dr" : 6,
                "Other": 7}

        # sex codification
        codification_sex = {
            "male" : 1,
            "female": 0
            
        }
        
        # embarked codification
        codification_embarked = {
            'S': 2,
            'C': 0,
            'Q': 1,
            "Nan" : -1
        }
        
        features_ml = ["Pclass","Sex","Age","Parch","Fare","Embarked",
                        "Title_Name","SibPar","IsChildWoman"]
                        
        features_inference = ["Pclass","Name","Sex","Age","SibSp","Parch",
                                "Ticket","Fare","Cabin","Embarked"]
        
        transformed_data = [transform_data(instance['features'], codification_title, codification_sex, codification_embarked, features_ml, features_inference) for instance in request["instances"]]
        
        # fix one sample test problem
        oneSample = False
        if len(transformed_data) == 1:
            transformed_data.append(",".join([str(0)]*len(features_ml)))
            oneSample = True

        # rf-classifier accepts data in CSV. It does not support JSON.
        # So, we need to submit the request in CSV format
        # Prediction for multiple observations in the same call
        
        result = client.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                               Body=('\n'.join(transformed_data).encode('utf-8')),
                               ContentType='text/csv')


        result = result['Body'].read().decode('utf-8')

        print(result)
        result = result[1:-1].split(",")
        predictions = ["live" if float(r) > 0.0 else "dead"  for r in result]
        
        if oneSample:
            predictions = [predictions[0]]

        return {
            'statusCode': 200,
            'isBase64Encoded':False,
            'body': predictions
        }

    except Exception as err:
        return {
            'statusCode': 400,
            'isBase64Encoded':False,
            'body': 'Call Failed {0}'.format(err)
        }
