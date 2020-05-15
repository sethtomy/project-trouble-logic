from PIL import Image
import flask
from flask_cors import CORS
import io

from logic import classify, load_model

app = flask.Flask(__name__)
CORS(app)


@app.before_request
def log_request_info():
    app.logger.debug('Headers: %s', flask.request.headers)
    app.logger.debug('Body: %s', flask.request.get_data())


@app.route('/', methods=['GET'])
def default():
    image_path = '../data/trouble.jpg'
    image = Image.open(image_path)
    res = classify(image)
    return flask.jsonify(res)


@app.route('/predict', methods=['POST'])
def predict():
    print(flask.request.files)
    if flask.request.files.get('image'):
        # read the image in PIL format
        print('Got image')
        image = flask.request.files['image'].read()
        image = Image.open(io.BytesIO(image))
        res = classify(image)
        return flask.jsonify(res)
    return 'Must pass image in form-data with label "image"', 415


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000)
