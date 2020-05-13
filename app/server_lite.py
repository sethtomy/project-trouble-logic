import numpy as np
import distro
from PIL import Image

DISTRO = distro.id()
print('Current platform is: ' + DISTRO)
if DISTRO == 'raspbian':
    import tflite_runtime.interpreter as tflite
else:
    import tensorflow.lite as tflite

MODEL_FILE = '../data/inception_v4_299_quant.tflite'
LABEL_FILE = '../data/labels.txt'


def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


if __name__ == '__main__':
    # Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=MODEL_FILE)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32

    # Resize image
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    image_path = '../data/trouble.jpg'
    img = Image.open(image_path).resize((width, height))

    # Add N dimension
    input_data = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    print(results)
    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(LABEL_FILE)
    for i in top_k:
        print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))
