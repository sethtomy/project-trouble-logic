from PIL import Image
import flask
import io
from .logic import classify, load_model

app = flask.Flask(__name__)


@app.route('/', methods=['GET'])
def default():
    image_path = '../data/trouble.jpg'
    image = Image.open(image_path)
    res = classify(image)
    return flask.jsonify(res)


@app.route('/predict', methods=['POST'])
def predict():
    if flask.request.files.get('image'):
        # read the image in PIL format
        image = flask.request.files['image'].read()
        image = Image.open(io.BytesIO(image))
        res = classify(image)
        return flask.jsonify(res)


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000)
