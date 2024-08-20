import base64
from PIL import Image
import numpy as np
import tensorflow as tf
from pathlib import Path

class Predictor:
    def __init__(self, imgstring, filename):
        self.imgstring = imgstring
        self.filename = filename

    def decodeImage(self):
        imgdata = base64.b64decode(self.imgstring)
        with open(self.filename, 'wb') as f:
            f.write(imgdata)

    def preprocess_image(self, image_path, target_size):
        try:
            # Open the image and convert to RGB mode
            image = Image.open(image_path).convert('RGB')
            
            # Resize the image to the target size
            image = image.resize(target_size)
            
            # Convert image to numpy array and normalize pixel values to [0, 1]
            image_array = np.array(image) / 255.0
            
            # Ensure the image has 3 channels (RGB)
            if image_array.shape[-1] != 3:
                raise ValueError(f"Expected 3 channels, got {image_array.shape[-1]} channels.")
            
            # Expand dimensions to create a batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
        
        except Exception as e:
            print(f"Error during image preprocessing: {e}")
            return None

    def predict(self):
        try:
            # Decode the image
            self.decodeImage()
            
            # Preprocess the image
            test_image = self.preprocess_image(self.filename, target_size=(224, 224))
            
            if test_image is not None:
                # Load model
                root_dir = Path(__file__).parent
                models_dir = root_dir / "models"
                model_path = models_dir / "experiment_0.h5"
                model = tf.keras.models.load_model(model_path)

                # Predict the result
                result = np.argmax(model.predict(test_image), axis=1)
                
                prediction = 'Normal' if result[0] == 1 else 'Adenocarcinoma Cancer'
                return {"prediction": prediction}
            else:
                return {"error": "Failed to preprocess image."}
        
        except Exception as e:
            print(f"Error during prediction: {e}")
            return {"error": str(e)}

