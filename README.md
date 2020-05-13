# Project Trouble

## Logic

This server will take an image of a dog and return the top five breeds and probability of that 
classification. This is currently setup with base weights of a ResNet50 network training on 
imagenet but will be adapted to this InceptionResnetV2 model (https://github.com/sethtomy/MuttClassifier) 
that was trained on the StanfordDogs dataset and achieved an accuracy of 90% on the test set.

This project is targeted towards use on the Raspberry Pi running on Raspbian.

## Endpoints

### GET `/`

This is a default route for testing, returns classification of a default image(The dog this project is named after)

```json
[
    {
        "German shepherd, German shepherd dog, German police dog, alsatian": 0.30196078431372547
    },
    {
        "bucket, pail": 0.15294117647058825
    },
    {
        "tub, vat": 0.10980392156862745
    },
    {
        "malinois": 0.09411764705882353
    },
    {
        "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin": 0.047058823529411764
    }
]
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
