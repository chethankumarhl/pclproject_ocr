import google.generativeai as genai
import PIL.Image
import io
import cv2
import numpy as np
from flask import Flask, request, render_template

# Configure API Key
genai.configure(api_key="AIzaSyANg8TrWYFz7uT2NEJYcOr0Euxr8C8XnpM")  # Replace with your actual API key

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

def extract_text_from_image(image_bytes):
    """Extract text using Gemini AI."""
    try:
        enhanced_image_bytes = enhance_image(image_bytes)
        image = PIL.Image.open(enhanced_image_bytes)

        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = "Extract only the exact text visible in the image."

        response = model.generate_content([image, prompt])
        return response.text.strip() if hasattr(response, 'text') else "No text found."
    
    except Exception as e:
        return f"Error: {e}"

@app.route("/", methods=["GET", "POST"])
def upload_image():
    """Render homepage and process image uploads."""
    extracted_text = None

    if request.method == "POST":
        image_file = request.files.get("image")
        if image_file:
            extracted_text = extract_text_from_image(image_file.read())

    return render_template("index.html", extracted_text=extracted_text)

if __name__ == "__main__":
    app.run(debug=True)
