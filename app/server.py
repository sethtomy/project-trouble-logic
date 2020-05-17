from PIL import Image
import flask
from flask_cors import CORS
import io

from logic import classify, load_model

app = flask.Flask(__name__)
CORS(app)


@app.route('/predict', methods=['POST'])
def predict():
    print(flask.request.files)
    if flask.request.files.get('image'):
        print('Got image')
        image = flask.request.files['image'].read()
        image = Image.open(io.BytesIO(image))
        res = classify(image)
        return flask.jsonify(res)
    return 'Must pass image in form-data with label "image"', 415


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000)
