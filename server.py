from flask import Flask, request, jsonify
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from io import BytesIO
import base64
import logging

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load Stable Diffusion model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline = None

try:
    if device.type == 'cuda':
        pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to(device)
    else:
        pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")

@app.route('/generate', methods=['POST'])
def generate_image():
    if not pipeline:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        prompt = request.json.get('prompt')
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400

        with torch.no_grad():
            output = pipeline(prompt, num_inference_steps=50)
            image = output.images[0]

        return jsonify({'image': 'data:image/jpeg;base64,' + image_to_base64(image)})

    except Exception as e:
        logging.error(f"Error during image generation: {e}")
        return jsonify({'error': 'Failed to generate image'}), 500

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

# Remove Flask's development server call
