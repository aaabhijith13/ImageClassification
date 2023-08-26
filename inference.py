import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests

# Setting up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'


# Based on https://github.com/pytorch/examples/blob/master/mnist/main.py
def Net():
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False   
    model.fc = nn.Sequential(
                   nn.Linear(2048, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, 133))
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_fn(model_dir):
    logger.info("In model_fn. Model directory is - %s", model_dir)
    model = Net().to(device)
    try:
        with open(os.path.join(model_dir, "model.pth"), "rb") as f:
            logger.info("Loading the dog-classifier model")
            checkpoint = torch.load(f, map_location=device)
            model.load_state_dict(checkpoint)
            logger.info('MODEL-LOADED')
        model.eval()
    except Exception as e:
        logger.error("Error during model loading: ", exc_info=True)
        raise
    return model


def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    logger.debug(f'Request body CONTENT-TYPE is: {content_type}')
    try:
        if content_type == JPEG_CONTENT_TYPE: 
            return Image.open(io.BytesIO(request_body))
        if content_type == JSON_CONTENT_TYPE:
            request = json.loads(request_body)
            url = request['url']
            img_content = requests.get(url).content
            return Image.open(io.BytesIO(img_content))
    except Exception as e:
        logger.error("Error during input processing: ", exc_info=True)
        raise
    raise Exception(f'Requested unsupported ContentType in content_type: {content_type}')


def predict_fn(input_object, model):
    logger.info('In predict fn')
    try:
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        input_object = test_transform(input_object)
        with torch.no_grad():
            prediction = model(input_object.unsqueeze(0))
        return prediction
    except Exception as e:
        logger.error("Error during prediction: ", exc_info=True)
        raise


def output_fn(predictions, content_type):
    logger.info('Serializing the generated output.')
    try:
        if content_type == JSON_CONTENT_TYPE:
            res = predictions.cpu().numpy().tolist()
            return json.dumps(res), content_type
    except Exception as e:
        logger.error("Error during output serialization: ", exc_info=True)
        raise
    raise Exception(f'Requested unsupported ContentType in Accept: {content_type}')
