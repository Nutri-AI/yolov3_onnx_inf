import os
import boto3
import logging

import numpy as np
from PIL import Image
import requests
from io import BytesIO
from botocore.exceptions import ClientError
import cv2


def s3_get_client():
    s3_client = boto3.client(
        's3',
        region_name= 'ap-northeast-2',
        aws_access_key_id= '',
        aws_secret_access_key= ''
    )
    return s3_client
    
# get presigned url
def get_url(bucket_name, object_name, expiration=3600):
    s3_client = s3_get_client()
    try:
        url = s3_client.generate_presigned_url(
            ClientMethod= 'get_object',
            Params={
                'Bucket': bucket_name,
                'Key':object_name
            },
            ExpiresIn=expiration
        )
    except ClientError as e:
        logging.error(e)
        return None
    
    return url

def get_img(obj_name):
    bucket = 'nutriai'
    key = obj_name
    url = get_url(bucket,key)
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img_arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    return img_arr, key

    



