import distro
import numpy as np

DISTRO = distro.id()
print('Current platform is: ' + DISTRO)
if DISTRO == 'raspbian':
    import tflite_runtime.interpreter as tflite
else:
    import tensorflow.lite as tflite

MODEL_FILE = '../data/inception_v4_299_quant.tflite'
LABEL_FILE = '../data/labels.txt'
interpreter = None
input_details = None
output_details = None


def classify(image):
    input_data = prepare_image(image)
    results = feed_nn(input_data)
    return prepare_results(results)


def prepare_results(results):
    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(LABEL_FILE)
    top_k_labels_and_percentages = \
        [{str.split(labels[index], ':')[1]: (float(results[index]) / 255.0)} for index in top_k]
    return top_k_labels_and_percentages


def load_model():
    # Load TFLite model and allocate tensors.
    global interpreter
    print('loading model...')
    interpreter = tflite.Interpreter(model_path=MODEL_FILE)
    interpreter.allocate_tensors()
    print('model loaded')

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


def feed_nn(input_data):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return np.squeeze(output_data)