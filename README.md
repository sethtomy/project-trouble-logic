# Project Trouble

## Logic

This server will take a

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