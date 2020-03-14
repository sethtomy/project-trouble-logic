# Project Trouble

## Logic

This server will take an image of a dog and return the top five breeds and probability of that 
classification. This is currently setup with base weights of a ResNet50 network training on 
imagenet but will be adapted to this InceptionResnetV2 model (https://github.com/sethtomy/MuttClassifier) 
that was trained on the StanfordDogs dataset and achieved an accuracy of 90% on the test set.

## Endpoints

### GET `/`

This is a default route for testing, returns

```json
{
    "success": true
}
```

### POST `/predict`

Provide `form-data` with key of `image`. Value will be the image of the dog you are
getting the classification of.

## Guide

### Build Locally

* `pip install -r requirements.txt`

### Run the server

This will start a flask server on `localhost:5000`.

* `python server.py`