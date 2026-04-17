import sys
import os
import numpy as np
import torch
import gradio as gr
from PIL import Image

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.models.cnn import RegressionCNN
from src.models.unet import UNet
from src.models.cvae import CVAE

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
def load_models():
    cnn = RegressionCNN()
    cnn.load_state_dict(torch.load(os.path.join(BASE_DIR, "outputs/models/cnn_best.pth"), map_location=device))
    cnn.to(device).eval()

    unet = UNet()
    unet.load_state_dict(torch.load(os.path.join(BASE_DIR, "outputs/models/unet_best.pth"), map_location=device))
    unet.to(device).eval()

    cvae = CVAE()
    cvae.load_state_dict(torch.load(os.path.join(BASE_DIR, "outputs/models/cvae_best.pth"), map_location=device))
    cvae.to(device).eval()

    return cnn, unet, cvae


cnn_model, unet_model, cvae_model = load_models()

# Preprocess image
def preprocess_image(image):
    image = image.resize((32, 32))
    image = np.array(image) / 255.0

    if len(image.shape) == 3:
        image = image.mean(axis=2)  # convert to grayscale

    image = image.reshape(1, 1, 32, 32)
    return torch.tensor(image, dtype=torch.float32).to(device)

# Postprocess output
def postprocess_image(tensor):
    img = tensor.detach().cpu().numpy()[0]
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

# Inference
def colorize(image, model_name):
    x = preprocess_image(image)

    if model_name == "CNN":
        model = cnn_model
        output = model(x)

    elif model_name == "U-Net":
        model = unet_model
        output = model(x)

    elif model_name == "CVAE":
        model = cvae_model
        # dummy RGB input required for forward pass
        dummy = torch.zeros((1, 3, 32, 32)).to(device)
        output, _, _ = model(x, dummy)

    return postprocess_image(output)

# Gradio Interface
interface = gr.Interface(
    fn=colorize,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Radio(["CNN", "U-Net", "CVAE"], label="Model", value="U-Net")
    ],
    outputs=gr.Image(type="numpy", label="Colorized Output"),
    title="Image Colorization",
    description="Upload a grayscale image and choose a model to colorize it."
)

if __name__ == "__main__":
    interface.launch()