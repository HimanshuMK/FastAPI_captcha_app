# CAPTCHA Recognition

This is a FastAPI-based application for recognizing CAPTCHA images using a TensorFlow model.

## File Structure


```plaintext
captcha_app
│----model
|      |----captcha_model.keras
│----static
|      |----style.css
│----templates
|      |----index.html
│----main.py
|----requirements.txt
```

## Project Structure

- `main.py`: The main FastAPI application file.
- `model/captcha_model.keras`: The pre-trained TensorFlow model for CAPTCHA recognition.
- `static/style.css`: CSS file for styling the web interface.
- `templates/index.html`: HTML file for the web interface.

## Requirements

- Python 3.7+
- TensorFlow
- FastAPI
- Uvicorn
- Pillow
- numpy

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/HimanshuMK/FastAPI_captcha_app.git
    ```
2. Navigate to the project directory:
    ```bash
    cd FastAPI_captcha_app
    ```
3. Create a virtual environment:
    ```bash
    python -m venv myenv
    ```
4. Activate the virtual environment:
    - On Windows:
        ```bash
        myenv\Scripts\activate
        ```
    - On macOS/Linux:
        ```bash
        source myenv/bin/activate
        ```
5. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1. Start the FastAPI application:
    ```bash
    uvicorn main:app --reload
    ```
2. Open your browser and go to `http://127.0.0.1:8000` to access the CAPTCHA prediction interface.

## Usage

1. Select a CAPTCHA image file using the file input.
2. Click the "Predict" button to get the CAPTCHA prediction.
3. The prediction result will be displayed below the button.

## Screenshots
![Screenshot 2024-07-17 194212](https://github.com/user-attachments/assets/4f626ed5-17f4-4112-a5b5-354a1fc81a15)
![Screenshot 2024-07-17 194253](https://github.com/user-attachments/assets/ecbd29bc-00db-48ee-a934-1b75c5b585b5)


