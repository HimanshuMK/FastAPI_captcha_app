from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import string

app = FastAPI()

# Load your TensorFlow model
model = tf.keras.models.load_model('model\captcha_model_v3.keras')

# Define the character set
CHAR_SET = string.ascii_lowercase + string.digits

def preprocess_image(image: Image.Image):
    # Convert the image to grayscale
    image = image.convert('L')
    # Resize the image to the input size required by your model
    image = image.resize((200, 50))  #model expected input size
    # Convert the image to a numpy array and normalize it
    image_array = np.array(image) / 255.0
    # Add batch dimension and channel dimension if necessary
    image_array = np.expand_dims(image_array, axis=(0, -1))  # shape becomes (1, height, width, 1)
    return image_array

def decode_prediction(prediction):
    # Decode the prediction into a CAPTCHA string
    captcha = ''
    for char_probs in prediction:
        char_index = np.argmax(char_probs, axis=-1)
        captcha += CHAR_SET[char_index]
    return captcha

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image file
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        # Preprocess the image
        input_data = preprocess_image(image)
        # Make prediction
        prediction = model.predict(input_data)
        # prediction.shape is (5, 1, 36)
        prediction = np.squeeze(prediction, axis=1)  # Now prediction.shape is (5, 36)
        # Decode the prediction
        captcha = decode_prediction(prediction)
        return {'prediction': captcha}
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("templates/index.html", "r") as file:
        return file.read()

app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
