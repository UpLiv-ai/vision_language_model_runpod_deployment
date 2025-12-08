import os
import torch
import runpod
import base64
import requests
import json
import re
from typing import Dict, Any
from io import BytesIO
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# --- 1. Model Loading (Runs Once at Worker Startup) ---

MODEL_ID = "openbmb/MiniCPM-V-4_5"
DEFAULT_PROMPT = """
VLM Prompt for Image Analysis
Your task is to act as an expert scene analyzer. For the given image, identify the primary object and generate a single, valid JSON object that describes its key attributes. The primary object is the main subject of the image, such as a piece of furniture, a door, or a cabinet. The output must be only the JSON object and nothing else.
JSON Structure and Field Definitions:
Your response must conform to the a JSON structure. Analyze the primary object in the image and fill in the values accordingly.
object_description (string): A concise but descriptive summary of the primary object. Include its type, primary material, and color.
is_transparent_container (boolean): Set to true only if the object is a container with large transparent surfaces designed to show its contents (e.g., a glass-front cabinet, a display case, a curio).
has_mirror (boolean): Set to true if the object has a mirrored surface (e.g., a bathroom vanity mirror, a mirrored closet door).
has_countertop (boolean): Set to true if the object is or includes a countertop surface (e.g., a kitchen island, a bathroom vanity top, a counter).
has_glass (boolean): Set to true if any part of the object is made of glass. This includes window panes, glass shelves, or glass panels.
has_clear_plastic (boolean): Set to true if any part of the object is made of clear or translucent plastic.
"""

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
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(
    path_to_load,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

# Move model to GPU and set to evaluation mode
model.to(dtype=torch.float16, device='cuda')
model.eval()
print("✅ Model loaded successfully and is ready for inference.")


# --- 2. Helper Functions ---

def parse_vlm_output(vlm_text_response: str) -> Dict[str, Any]:
    """Extracts a JSON object from the VLM's text response."""
    # Use regex to find content between the first '{' and the last '}'
    match = re.search(r'\{.*\}', vlm_text_response, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find a valid JSON object in the VLM response:\n{vlm_text_response}")
    
    json_string = match.group(0)
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON from VLM output. String was:\n{json_string}")
        raise e


# --- 3. Handler Function (Runs for Each API Request) ---

def handler(job):
    """
    Processes a single job from the RunPod serverless queue.
    """
    job_input = job.get('input', {})

    # --- Get Image from URL (Azure Blob / Web) ---
    image_url = job_input.get('image_url')
    
    # Fallback support for base64
    image_base64 = job_input.get('image_base64')

    image = None

    if image_url:
        try:
            print(f"Downloading image from: {image_url}")
            response = requests.get(image_url)
            response.raise_for_status()
            
            # Load into BytesIO buffer to handle non-seekable streams
            image = Image.open(BytesIO(response.content)).convert('RGB')
            
        except Exception as e:
            return {"error": f"Failed to download image from URL: {e}"}
            
    elif image_base64:
        try:
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            return {"error": f"Failed to decode base64 image: {e}"}
            
    else:
        return {"error": "No image provided. Please include 'image_url' in the input."}

    # --- Get Prompt ---
    prompt = job_input.get('prompt', DEFAULT_PROMPT)
    messages = [{'role': 'user', 'content': prompt}]

    # --- Run Inference ---
    try:
        # Generate raw string response
        raw_response = model.chat(
            image=image,
            msgs=messages,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.7
        )

        print(f"--- Raw Model Response --- \n{raw_response}\n--------------------------")

        # Parse the raw string into a JSON object
        parsed_json = parse_vlm_output(raw_response)
        
        # Return the actual Python dictionary (which RunPod serializes to JSON)
        return parsed_json

    except ValueError as ve:
        # Handle cases where the model didn't output JSON structure
        return {"error": f"Model output parsing failed: {str(ve)}", "raw_output": raw_response}
    except json.JSONDecodeError as je:
        # Handle cases where the JSON was malformed
        return {"error": f"Invalid JSON generated: {str(je)}", "raw_output": raw_response}
    except Exception as e:
        # General inference errors
        return {"error": f"Inference failed: {e}"}


# --- 4. Start the RunPod Serverless Worker ---
runpod.serverless.start({"handler": handler})