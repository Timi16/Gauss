from flask import Flask, request, jsonify
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)

# Load Stable Diffusion model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to(device)
else:
    pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

@app.route('/generate', methods=['POST'])
def generate_image():
    prompt = request.json['prompt']
    with torch.no_grad():
        output = pipeline(prompt, num_inference_steps=50)
        image = output.images[0]
    # Convert image to base64 and return
    return jsonify({'image': 'data:image/jpeg;base64,' + image_to_base64(image)})

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
