from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from predict import Predictor
import pathlib
import base64

root_dir = pathlib.Path(__file__).parent
upload_folder = root_dir / "dump"
os.makedirs(upload_folder, exist_ok=True)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/process_image", methods=['POST'])
@cross_origin()

def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file to a temporary folder
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    # Read the saved file and convert it to a base64 string
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    # Create a Predictor instance and run the prediction
    predictor = Predictor(encoded_string, file_path)
    result = predictor.predict()

    # Delete the temporary uploaded file
    os.remove(file_path)

    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080) # for AWS
