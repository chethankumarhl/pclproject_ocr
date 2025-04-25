import google.generativeai as genai
import PIL.Image
import io
import cv2
import numpy as np
import requests
from flask import Flask, request, render_template

# Configure Gemini API Key
genai.configure(api_key="AIzaSyANg8TrWYFz7uT2NEJYcOr0Euxr8C8XnpM")  # Replace with your actual API key

# OCR.space API Key
OCR_API_KEY = "K83604811988957"  # Replace with your OCR.space API key

app = Flask(__name__)

def enhance_image(image_bytes):
    """Enhance image for better text readability."""
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

        # Improve contrast
        image = cv2.equalizeHist(image)

        # Apply adaptive thresholding
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)

        # Convert to bytes
        _, buffer = cv2.imencode('.jpg', image)
        return io.BytesIO(buffer)

    except Exception:
        return io.BytesIO(image_bytes)

def extract_text_gemini(image_bytes):
    """Extract text from handwriting using Gemini AI."""
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
    """Extract text from printed documents using OCR.space API."""
    try:
        response = requests.post(
            "https://api.ocr.space/parse/image",
            files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            data={"apikey": OCR_API_KEY, "language": "eng"}
        )

        result = response.json()
        if "ParsedResults" in result and result["ParsedResults"]:
            return result["ParsedResults"][0].get("ParsedText", "No text found.")
        else:
            return "OCR API failed to extract text."

    except Exception as e:
        return f"Error: {e}"

@app.route("/", methods=["GET", "POST"])
def upload_image():
    """Render homepage and process image uploads."""
    extracted_text = None
    error = None

    if request.method == "POST":
        image_file = request.files.get("image")
        text_type = request.form.get("text_type")  # "handwriting" or "printed"

        if image_file and text_type:
            image_bytes = image_file.read()

            if text_type == "handwriting":
                extracted_text = extract_text_gemini(image_bytes)
            elif text_type == "printed":
                extracted_text = extract_text_ocr_space(image_bytes)
            else:
                error = "Invalid text type selected."

    return render_template("index.html", extracted_text=extracted_text, error=error)

if __name__ == "__main__":
    app.run(debug=True)
