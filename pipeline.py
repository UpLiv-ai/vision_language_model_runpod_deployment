import os
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# --- 1. Configuration ---
IMAGE_PATH = "./test_of_pipeline.jpg" 
PROMPT = """
VLM Prompt for Image Analysis

Your task is to act as an expert scene analyzer. For the given image, identify the primary object and generate a single, valid JSON object that describes its key attributes. The primary object is the main subject of the image, such as a piece of furniture, a door, or a cabinet.
The output must be only the JSON object and nothing else.

JSON Structure and Field Definitions:

Your response must conform to the a JSON structure. Analyze the primary object in the image and fill in the values accordingly.

object_description (string): A concise but descriptive summary of the primary object. Include its type, primary material, and color.
is_transparent_container (boolean): Set to true only if the object is a container with large transparent surfaces designed to show its contents (e.g., a glass-front cabinet, a display case, a curio).
has_mirror (boolean): Set to true if the object has a mirrored surface (e.g., a bathroom vanity mirror, a mirrored closet door).
has_countertop (boolean): Set to true if the object is or includes a countertop surface (e.g., a kitchen island, a bathroom vanity top, a counter).
has_glass (boolean): Set to true if any part of the object is made of glass. This includes window panes, glass shelves, or glass panels.
has_clear_plastic (boolean): Set to true if any part of the object is made of clear or translucent plastic.

"""
MODEL_ID = "openbmb/MiniCPM-V-4_5"

# --- 2. Load Model and Tokenizer (MODIFIED SECTION) ---

# Use RunPod's persistent volume if available
if os.path.exists('/runpod-volume'):
    base_volume_path = '/runpod-volume'
else:
    base_volume_path = '/workspace'

# Define the path to the local model directory
local_model_path = os.path.join(base_volume_path, MODEL_ID.split('/')[-1])

# Check if the model exists locally, otherwise use the Hub ID
if os.path.exists(local_model_path):
    print(f"✅ Loading model from local path: {local_model_path}")
    path_to_load = local_model_path
else:
    print(f"⚠️ Local model not found. Downloading from Hugging Face Hub: {MODEL_ID}")
    path_to_load = MODEL_ID

print("Loading model and tokenizer...")
model = AutoModel.from_pretrained(
    path_to_load,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    path_to_load,
    trust_remote_code=True
)

# Move model to GPU and set to evaluation mode
model.to(dtype=torch.float16, device='cuda')
model.eval()
print("Model loaded successfully.")

# --- 3. Prepare Image and Prompt ---
try:
    image = Image.open(IMAGE_PATH).convert('RGB')
except FileNotFoundError:
    print(f"Error: Image not found at {IMAGE_PATH}")
    exit()

messages = [{'role': 'user', 'content': PROMPT}]

# --- 4. Run Inference ---
print("Running inference...")
response = model.chat(
    image=image,
    messages=messages,
    tokenizer=tokenizer,
    sampling=True,
    temperature=0.7 
)

# --- 5. Print Result ---
print("\n" + "="*20)
# print(f"Prompt: {PROMPT}")
print(f"Response: {response}")
print("="*20 + "\n")