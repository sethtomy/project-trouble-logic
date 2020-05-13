import numpy as np
import distro
from PIL import Image
import flask
import io

DISTRO = distro.id()
print('Current platform is: ' + DISTRO)
if DISTRO == 'raspbian':
    import tflite_runtime.interpreter as tflite
else:
    import tensorflow.lite as tflite

MODEL_FILE = '../data/inception_v4_299_quant.tflite'
LABEL_FILE = '../data/labels.txt'

app = flask.Flask(__name__)
interpreter = None
input_details = None
output_details = None


@app.route('/', methods=['GET'])
def default():
    image_path = '../data/trouble.jpg'
    image = Image.open(image_path)
    input_data = prepare_image(image)
    classify(input_data)

    return flask.jsonify({'success': True})


@app.route('/predict', methods=['POST'])
def predict():
    if flask.request.files.get('image'):
        # read the image in PIL format
        image = flask.request.files['image'].read()
        image = Image.open(io.BytesIO(image))
        input_data = prepare_image(image)
        classify(input_data)

        return flask.jsonify({'success': True})

def load_model():
    # Load TFLite model and allocate tensors.
    global interpreter
    interpreter = tflite.Interpreter(model_path=MODEL_FILE)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    global input_details, output_details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


def prepare_image(pil_image):
    # Resize image
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    resized_image = pil_image.resize((width, height))

    # Add N dimension
    return np.expand_dims(resized_image, axis=0)


def classify(input_data):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    print(results)
    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(LABEL_FILE)
    for i in top_k:
        print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000)
