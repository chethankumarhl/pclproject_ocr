import google.generativeai as genai
import PIL.Image
import io
import cv2
import numpy as np
import requests
import os
from flask import Flask, request, render_template
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure API Keys
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # Load Gemini API key from .env
OCR_API_KEY = os.getenv("OCR_API_KEY")  # Load OCR.space API key from .env

app = Flask(__name__)

def enhance_image(image_bytes):
    """Enhance image for better text readability."""
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        image = cv2.bilateralFilter(image, 9, 75, 75)  # Reduce noise while preserving edges
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Otsu's binarization
        _, buffer = cv2.imencode('.jpg', image)
        return io.BytesIO(buffer)
    except Exception:
        return io.BytesIO(image_bytes)

def extract_text_gemini(image_bytes):
    """Extract handwritten text using Gemini AI."""
    try:
        enhanced_image_bytes = enhance_image(image_bytes)
        image = PIL.Image.open(enhanced_image_bytes)
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = "Extract only the exact text visible in the image."
        response = model.generate_content([image, prompt])
        return response.text.strip() if hasattr(response, 'text') else "No text found."
    except Exception as e:
        return f"Error: {e}"

def extract_text_ocr_space(image_bytes):
    """Extract printed text using OCR.space API."""
    try:
        files = {"image": ("image.jpg", image_bytes, "image/jpeg")}
        response = requests.post(
            "https://api.ocr.space/parse/image",
            files=files,
            data={"apikey": OCR_API_KEY, "language": "eng"}
        )
        return response.json()["ParsedResults"][0]["ParsedText"].strip()
    except Exception as e:
        return f"Error: {e}"

def determine_text_type(image_bytes):
    """Determine whether the text is handwritten or printed."""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edge_density = np.sum(edges) / edges.size
    return "handwritten" if edge_density > 0.15 else "printed"  # Adjusted threshold for better classification

@app.route("/", methods=["GET", "POST"])
def upload_image():
    extracted_text = None
    text_type = None

    if request.method == "POST":
        image_file = request.files.get("image")
        if image_file:
            image_bytes = image_file.read()
            text_type = determine_text_type(image_bytes)
            if text_type == "handwritten":
                extracted_text = extract_text_gemini(image_bytes)
            else:
                extracted_text = extract_text_ocr_space(image_bytes)
    
    return render_template("index.html", extracted_text=extracted_text, text_type=text_type)

if __name__ == "__main__":
    app.run(debug=True)
