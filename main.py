from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import string

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to the specific origins you want to allow
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your TensorFlow model
model = tf.keras.models.load_model('model/captcha_model_v3.keras')

# Define the character set
CHAR_SET = string.ascii_lowercase + string.digits

def preprocess_image(image: Image.Image):
    image = image.convert('L')
    image = image.resize((200, 50))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=(0, -1))
    return image_array

def decode_prediction(prediction):
    captcha = ''
    for char_probs in prediction:
        char_index = np.argmax(char_probs, axis=-1)
        captcha += CHAR_SET[char_index]
    return captcha

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        input_data = preprocess_image(image)
        prediction = model.predict(input_data)
        prediction = np.squeeze(prediction, axis=1)
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
