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

# --- 1. Model Loading ---

# UPDATED: Use the Omni 4.5 model ID
MODEL_ID = "openbmb/MiniCPM-o-4_5"

# [Your PROMPT_TEMPLATES remain exactly the same as before...]
GENERIC_PROMPT_TEMPLATE = """
VLM Prompt for Image Analysis
Your task is to act as an expert scene analyzer. For the given image, identify the primary object and generate a single, valid JSON object that describes its key attributes. The output must be only the JSON object and nothing else.
You are given an approximate bounding box for the object (x, y, width, height), where x and y are the top-left pixel location, width extends to the right, and height extends downward. This bounding box is only a rough approximation. You must refine it to better fit the object and output the refined bounding box.
Approximate bbox (x, y, width, height): {bbox}
JSON Structure and Field Definitions:
Your response must conform to a JSON structure. Analyze the primary object in the image and in particular bounding box region and fill in the values accordingly.
object_description (string): {object_description_line}
is_transparent_container (boolean): Set to true only if the object is a container with large transparent surfaces designed to show its contents (e.g., a glass-front cabinet, a display case, a curio).
has_mirror (boolean): Set to true if the object has a mirrored surface (e.g., a bathroom vanity mirror, a mirrored closet door).
has_countertop (boolean): Set to true if the object is or includes a countertop surface (e.g., a kitchen island, a bathroom vanity top, a counter).
has_glass (boolean): Set to true if any part of the object is made of glass. This includes window panes, glass shelves, or glass panels.
has_clear_plastic (boolean): Set to true if any part of the object is made of clear or translucent plastic.
refined_bbox (array of 4 numbers): The refined bounding box in the same format [x, y, width, height], tightly fit to the object.
"""

WINDOW_PROMPT_TEMPLATE = """
VLM Prompt for Image Analysis
Your task is to act as an expert scene analyzer. The object is a window. Focus ONLY on the window frame. Do not describe the glass, view, or anything beyond the frame. The output must be only the JSON object and nothing else.
You are given an approximate bounding box for the window (x, y, width, height), where x and y are the top-left pixel location, width extends to the right, and height extends downward. This bounding box is only a rough approximation. You must refine it to better fit the window frame and output the refined bounding box.
Approximate bbox (x, y, width, height): {bbox}
JSON Structure and Field Definitions:
Your response must conform to a JSON structure. Analyze the window frame and fill in the values accordingly.
object_description (string): A concise description of the window frame only. Include frame material and color.
refined_bbox (array of 4 numbers): The refined bounding box in the same format [x, y, width, height], tightly fit to the window frame.
"""

DOOR_PROMPT_TEMPLATE = """
VLM Prompt for Image Analysis
Your task is to act as an expert scene analyzer. The object is a door. Describe the door and its frame together. The output must be only the JSON object and nothing else.
You are given an approximate bounding box for the door (x, y, width, height), where x and y are the top-left pixel location, width extends to the right, and height extends downward. This bounding box is only a rough approximation. You must refine it to better fit the door and its frame and output the refined bounding box.
Approximate bbox (x, y, width, height): {bbox}
JSON Structure and Field Definitions:
Your response must conform to a JSON structure. Analyze the door and its frame and fill in the values accordingly.
object_description (string): A concise description of the door and its frame. Include materials and colors.
refined_bbox (array of 4 numbers): The refined bounding box in the same format [x, y, width, height], tightly fit to the door and frame.
"""

SURFACE_PROMPT_TEMPLATE = """
VLM Prompt for Image Analysis
Your task is to act as an expert scene analyzer. The object is a {surface} surface. Analyze the surface texture in the image and choose the single best-matching description from the provided list. Factor out lighting effects and shadows; base your choice on the underlying material/PBR texture. The output must be only the JSON object and nothing else.
JSON Structure and Field Definitions:
Your response must conform to a JSON structure. Fill in the values accordingly.
matched_texture_description (string): The exact string from the provided list that best matches the surface texture.
Available {surface} textures:
{options_block}
"""

if os.path.exists('/runpod-volume'):
    base_volume_path = '/runpod-volume'
else:
    base_volume_path = '/workspace'

local_model_path = os.path.join(base_volume_path, MODEL_ID.split('/')[-1])

if os.path.exists(local_model_path):
    print(f"✅ Loading model from local path: {local_model_path}")
    path_to_load = local_model_path
else:
    print(f"⚠️ Local model not found. Downloading from Hugging Face Hub: {MODEL_ID}")
    path_to_load = MODEL_ID

print("Loading model and tokenizer...")

# UPDATED: New initialization arguments for Omni model
model = AutoModel.from_pretrained(
    path_to_load,
    trust_remote_code=True,
    attn_implementation="sdpa", # Recommended in README
    torch_dtype=torch.bfloat16,
    # CRITICAL: We disable audio and tts to save VRAM for your furniture analysis
    init_vision=True,
    init_audio=False,
    init_tts=False
)
tokenizer = AutoTokenizer.from_pretrained(
    path_to_load,
    trust_remote_code=True
)

model.eval().cuda()
print("✅ MiniCPM-o 4.5 loaded successfully (Vision Only Mode).")

# --- 2. Helper Functions ---
# [Keep your existing helper functions: parse_vlm_output, _coerce_image_urls, etc.]

def parse_vlm_output(vlm_text_response: str) -> Dict[str, Any]:
    match = re.search(r'\{.*\}', vlm_text_response, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find a valid JSON object in response:\n{vlm_text_response}")
    json_string = match.group(0)
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {json_string}")
        raise e

def _coerce_image_urls(job_input):
    image_url = job_input.get('image_url')
    image_urls = job_input.get('image_urls')
    if isinstance(image_urls, list): return [u for u in image_urls if u]
    if isinstance(image_url, list): return [u for u in image_url if u]
    if isinstance(image_url, str) and image_url: return [image_url]
    return []

def _load_image_from_url(image_url: str):
    try:
        print(f"Downloading image from: {image_url}")
        response = requests.get(image_url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert('RGB')
    except Exception as e:
        return {"error": str(e)}

def _load_image_from_base64(image_base64: str):
    try:
        image_bytes = base64.b64decode(image_base64)
        return Image.open(BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        return {"error": str(e)}

def _normalize_bbox(bbox_input):
    if bbox_input is None: return None
    if isinstance(bbox_input, (list, tuple)) and len(bbox_input) == 4:
        return [float(v) for v in bbox_input]
    if isinstance(bbox_input, str):
        parts = [p.strip() for p in bbox_input.split(",")]
        if len(parts) == 4: return [float(v) for v in parts]
    return None

def _build_prompt(job_input: Dict[str, Any]) -> str:
    input_category = job_input.get("input_category")
    if input_category in {"wall", "floor", "ceiling"}:
        list_key = f"{input_category}_descriptions"
        options = job_input.get(list_key)
        if not options: raise ValueError(f"Missing '{list_key}'")
        options_block = "\n".join([f"- {str(o)}" for o in options])
        return SURFACE_PROMPT_TEMPLATE.format(surface=input_category, options_block=options_block).strip()

    bbox = _normalize_bbox(job_input.get("bbox"))
    if bbox is None: raise ValueError("Missing or invalid 'bbox'.")

    if input_category == "window": return WINDOW_PROMPT_TEMPLATE.format(bbox=bbox).strip()
    if input_category == "door": return DOOR_PROMPT_TEMPLATE.format(bbox=bbox).strip()

    object_description_line = "A concise but descriptive summary of the primary object. Include its type, primary material, and color."
    if input_category: object_description_line += f" The object should be in the category '{input_category}'."
    
    return GENERIC_PROMPT_TEMPLATE.format(bbox=bbox, object_description_line=object_description_line).strip()


# --- 3. Handler Function ---

def handler(job):
    job_input = job.get('input', {})

    # Load Images
    image_urls = _coerce_image_urls(job_input)
    image_base64 = job_input.get('image_base64')
    
    images = []
    if image_urls:
        for url in image_urls: images.append(_load_image_from_url(url))
    elif image_base64:
        images.append(_load_image_from_base64(image_base64))
    else:
        return {"error": "No image provided."}

    # Validate Images
    for idx, img in enumerate(images):
        if isinstance(img, dict) and img.get("error"):
            return {"error": f"Image {idx} failed: {img['error']}"}

    image_input = images[0] # The model prefers single PIL images for standard chat

    # Build Prompt
    try:
        prompt_text = job_input.get('prompt') or _build_prompt(job_input)
    except ValueError as ve:
        return {"error": str(ve)}

    # UPDATED: Inference Logic for Omni Model
    # The Omni model prefers the image embedded in the content list
    messages = [
        {
            "role": "user", 
            "content": [image_input, prompt_text]
        }
    ]

    try:
        raw_response = model.chat(
            msgs=messages,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.7,
            use_tts_template=False,   # Disable TTS template
            enable_thinking=False     # Standard VLM mode
        )

        print(f"--- Raw Model Response --- \n{raw_response}\n--------------------------")
        return parse_vlm_output(raw_response)

    except Exception as e:
        return {"error": f"Inference failed: {e}"}


runpod.serverless.start({"handler": handler})
# --- 4. Execution Logic ---

# if __name__ == "__main__":
#     # --- LOCAL TESTING MODE ---
#     print("🧪 Running in Local Test Mode...")
    
#     test_input = {
#         "input": {
#             "image_url": "https://images.unsplash.com/photo-1586023492125-27b2c045efd7?q=80&w=2158&auto=format&fit=crop",
#             "input_category": "chair", 
#             "bbox": [340, 200, 150, 250]
#         }
#     }

#     try:
#         result = handler(test_input)
#         print("\n✅ Handler Output:")
#         print(json.dumps(result, indent=4))
#     except Exception as e:
#         print(f"\n❌ Handler Failed: {e}")

# else:
    # runpod.serverless.start({"handler": handler})