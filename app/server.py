from PIL import Image
import flask
from flask_cors import CORS
import io
import pyheif

from logic import classify, load_model

app = flask.Flask(__name__)
CORS(app)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        print(flask.request.files)
        file = flask.request.files.get('image')
        file_type = file.filename.split('.')[1].lower()
        if file_type == 'heic':
            print('Got HEIC image.')
            heif_file = pyheif.read_heif(file)
            image = Image.frombytes(mode=heif_file.mode, size=heif_file.size, data=heif_file.data)
        else:
            # read the image in PIL format
            print('Got non-HEIC image.')
            image = flask.request.files['image'].read()
            image = Image.open(io.BytesIO(image))
        res = classify(image)
        print('Successfully classified image.')
        return flask.jsonify(res)
    except:
        return 'Must pass image in form-data with label "image"', 415


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000)
