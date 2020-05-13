import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io

# Set CPU as available physical device
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
my_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(devices=my_devices, device_type='GPU')

tf.debugging.set_log_device_placement(True)

app = flask.Flask(__name__)
model = None


def load_model():
    global model
    model = ResNet50(weights="imagenet")


def prepare_image(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize and preprocess
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    return image


@app.route("/", methods=["GET"])
def default():
    return flask.jsonify({'success': True})


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    if flask.request.files.get("image"):
        # read the image in PIL format
        image = flask.request.files['image'].read()
        image = Image.open(io.BytesIO(image))

        image = prepare_image(image, target=(224, 224))

        preds = model.predict(image)
        results = imagenet_utils.decode_predictions(preds)
        data["predictions"] = []

        for (imagenetID, label, prob) in results[0]:
            r = {"label": label, "probability": float(prob)}
            data["predictions"].append(r)

        data["success"] = True

    return flask.jsonify(data)


if __name__ == "__main__":
    load_model()
    app.run(host='0.0.0.0', port=80)