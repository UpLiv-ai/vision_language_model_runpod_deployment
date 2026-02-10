import os
import torch
import runpod
import base64
import requests
import json
import re
import numpy as np
from typing import Dict, Any, List, Union, Tuple
from io import BytesIO
from PIL import Image, ImageDraw, ImageOps
from transformers import AutoModel, AutoTokenizer

# --- 1. Model Loading ---

MODEL_ID = "openbmb/MiniCPM-o-4_5"

# --- PROMPT UPDATES: STRICT BREVITY FOR SAM3 ---

GENERIC_PROMPT_TEMPLATE = """
VLM Prompt for Image Analysis
Your task is to act as an expert scene analyzer. You are provided with {num_images} image(s).
For EACH image, identify the primary object contained generally within the hint bounding box.
Your goal is to output a refined bounding box that guarantees the ENTIRE object is inside it.
Do not make the box too tight. It is better to include a little background than to cut off part of the object.

Hint bbox(es) (Normalized 0-1000, [x1, y1, x2, y2]): {bbox}

JSON Structure and Field Definitions:
Your response must conform to a JSON structure.
object_description (string): {object_description_line}
is_transparent_container (boolean): Set to true only if the object is a container with large transparent surfaces.
has_mirror (boolean): Set to true if the object has a mirrored surface.
has_countertop (boolean): Set to true if the object is or includes a countertop surface.
has_glass (boolean): Set to true if any part of the object is made of glass.
has_clear_plastic (boolean): Set to true if any part of the object is made of clear or translucent plastic.
reasoning (string): Briefly explain how you ensured the box contains the whole object (e.g., "Expanded to include chair legs").
refined_bboxes (array of arrays): A list of refined bounding boxes, one for each input image. 
Format: [[x1, y1, x2, y2], ...]. Coordinates must be normalized 0-1000 integers (Top-Left to Bottom-Right).
The list MUST contain exactly {num_images} bounding boxes.
"""

WINDOW_PROMPT_TEMPLATE = """
VLM Prompt for Image Analysis
Your task is to act as an expert scene analyzer. You are provided with {num_images} image(s) of windows.
Focus ONLY on the window frame. Ensure the bounding box encompasses the entire frame, including outer trim. Do not cut off the edges.

Hint bbox(es) (Normalized 0-1000, [x1, y1, x2, y2]): {bbox}

JSON Structure and Field Definitions:
object_description (string): A STRICTLY BRIEF (2-5 words) description of the window frame class and material (e.g., "white wooden window frame" or "black aluminum window"). DO NOT describe the view, glass, or wall.
reasoning (string): Briefly explain how you ensured the box contains the whole frame.
refined_bboxes (array of arrays): A list of refined bounding boxes, one for each input image. 
Format: [[x1, y1, x2, y2], ...]. Coordinates must be normalized 0-1000 integers.
"""

DOOR_PROMPT_TEMPLATE = """
VLM Prompt for Image Analysis
Your task is to act as an expert scene analyzer. You are provided with {num_images} image(s) of doors.
Describe the door and its frame together. Ensure the bounding box encompasses the entire door and complete frame structure. Do not cut it too tight.

Hint bbox(es) (Normalized 0-1000, [x1, y1, x2, y2]): {bbox}

JSON Structure and Field Definitions:
object_description (string): A STRICTLY BRIEF (2-5 words) description of the door class, material, and color (e.g., "white paneled door" or "brown wooden sliding door"). DO NOT describe the room, floor, or handle details.
reasoning (string): Briefly explain how you ensured the box contains the whole door and frame.
refined_bboxes (array of arrays): A list of refined bounding boxes, one for each input image. 
Format: [[x1, y1, x2, y2], ...]. Coordinates must be normalized 0-1000 integers.
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
model = AutoModel.from_pretrained(
    path_to_load,
    trust_remote_code=True,
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
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

def parse_vlm_output(vlm_text_response: str) -> Dict[str, Any]:
    match = re.search(r'\{.*\}', vlm_text_response, re.DOTALL)
    if match:
        json_string = match.group(0)
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            pass
    return {"raw_output": vlm_text_response}

def _coerce_image_urls(job_input) -> List[str]:
    image_url = job_input.get('image_url')
    image_urls = job_input.get('image_urls')
    urls = []
    if isinstance(image_urls, list): urls.extend([u for u in image_urls if u])
    if isinstance(image_url, list): urls.extend([u for u in image_url if u])
    elif isinstance(image_url, str) and image_url: urls.append(image_url)
    
    seen = set()
    unique_urls = []
    for u in urls:
        if u not in seen:
            unique_urls.append(u)
            seen.add(u)
    return unique_urls

def _load_image_from_url(image_url: str):
    try:
        print(f"Downloading image from: {image_url}")
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img = ImageOps.exif_transpose(img) 
        return img.convert('RGB')
    except Exception as e:
        return {"error": str(e)}

def _load_image_from_base64(image_base64: str):
    try:
        image_bytes = base64.b64decode(image_base64)
        img = Image.open(BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)
        return img.convert('RGB')
    except Exception as e:
        return {"error": str(e)}

def _load_image_from_path(image_path: str):
    try:
        if not os.path.exists(image_path):
            return {"error": f"File not found: {image_path}"}
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)
        return img.convert('RGB')
    except Exception as e:
        return {"error": str(e)}

# --- COORDINATE UTILITIES ---

def _normalize_box(box_pixel: List[float], width: int, height: int) -> List[int]:
    x, y, w, h = box_pixel
    x2 = x + w
    y2 = y + h
    x1_n = int((x / width) * 1000)
    y1_n = int((y / height) * 1000)
    x2_n = int((x2 / width) * 1000)
    y2_n = int((y2 / height) * 1000)
    return [max(0, min(1000, x1_n)), max(0, min(1000, y1_n)), max(0, min(1000, x2_n)), max(0, min(1000, y2_n))]

def _denormalize_box_with_padding(box_norm: List[int], width: int, height: int, padding_pct: float = 0.05) -> List[int]:
    x1_n, y1_n, x2_n, y2_n = box_norm
    
    x1 = int((x1_n / 1000) * width)
    y1 = int((y1_n / 1000) * height)
    x2 = int((x2_n / 1000) * width)
    y2 = int((y2_n / 1000) * height)

    w_raw = x2 - x1
    h_raw = y2 - y1

    pad_x = int(w_raw * padding_pct)
    pad_y = int(h_raw * padding_pct)

    final_x1 = max(0, x1 - pad_x)
    final_y1 = max(0, y1 - pad_y)
    final_x2 = min(width, x2 + pad_x)
    final_y2 = min(height, y2 + pad_y)
    
    final_w = final_x2 - final_x1
    final_h = final_y2 - final_y1
    
    return [final_x1, final_y1, final_w, final_h]

def _extract_raw_bboxes(job_input) -> List[List[float]]:
    bbox_single = job_input.get('bbox')
    bboxes_list = job_input.get('bboxes')
    final_bboxes = []

    def normalize_single(b):
        if isinstance(b, (list, tuple)) and len(b) == 4: return [float(v) for v in b]
        return None

    if bboxes_list and isinstance(bboxes_list, list):
        for b in bboxes_list:
            norm_b = normalize_single(b)
            if norm_b: final_bboxes.append(norm_b)
    elif bbox_single:
        norm_b = normalize_single(bbox_single)
        if norm_b: final_bboxes.append(norm_b)
            
    return final_bboxes

def _format_bbox_string(bboxes_norm: List[List[int]]) -> str:
    if not bboxes_norm: return "None"
    parts = []
    for i, b in enumerate(bboxes_norm):
        parts.append(f"Image {i+1}: {b}")
    return ", ".join(parts)

def _build_prompt(job_input: Dict[str, Any], bboxes_norm: List[List[int]], num_images: int) -> str:
    input_category = job_input.get("input_category")

    if input_category in {"wall", "floor", "ceiling"}:
        list_key = f"{input_category}_descriptions"
        options = job_input.get(list_key)
        if not options: raise ValueError(f"Missing '{list_key}'")
        options_block = "\n".join([f"- {str(o)}" for o in options])
        return SURFACE_PROMPT_TEMPLATE.format(surface=input_category, options_block=options_block).strip()

    bbox_str = _format_bbox_string(bboxes_norm)
    
    # Generic Prompt Update: Strict brevity
    object_description_line = (
        "A STRICTLY BRIEF (2-5 words) text prompt for the object, suitable for a segmentation model (e.g. 'yellow armchair' or 'wooden table'). "
        "Do not include background details."
    )
    if input_category: 
        object_description_line += f" The object should be in the category '{input_category}'."

    if input_category == "window": 
        return WINDOW_PROMPT_TEMPLATE.format(bbox=bbox_str, num_images=num_images).strip()
    if input_category == "door": 
        return DOOR_PROMPT_TEMPLATE.format(bbox=bbox_str, num_images=num_images).strip()
    
    return GENERIC_PROMPT_TEMPLATE.format(
        bbox=bbox_str, 
        object_description_line=object_description_line,
        num_images=num_images
    ).strip()


# --- 3. Handler Function ---

def handler(job):
    job_input = job.get('input', {})

    # 1. Load Images
    image_urls = _coerce_image_urls(job_input)
    image_base64 = job_input.get('image_base64')
    image_paths = job_input.get('image_paths')
    
    images = []
    for url in image_urls:
        img = _load_image_from_url(url)
        if isinstance(img, dict) and img.get("error"): return {"error": f"Failed to load image {url}: {img['error']}"}
        images.append(img)
    if image_base64:
        img = _load_image_from_base64(image_base64)
        if isinstance(img, dict) and img.get("error"): return {"error": f"Failed to load base64 image: {img['error']}"}
        images.append(img)
    if isinstance(image_paths, list):
        for path in image_paths:
            img = _load_image_from_path(path)
            if isinstance(img, dict) and img.get("error"): return {"error": f"Failed to load local file {path}: {img['error']}"}
            images.append(img)

    if not images: return {"error": "No valid images provided."}

    # 2. Process BBoxes 
    # Logic: Extract provided bboxes. If missing or partial, fill with [0, 0, w, h]
    raw_bboxes_pixel = _extract_raw_bboxes(job_input)
    norm_bboxes_xyxy = []
    
    for i, img in enumerate(images):
        w, h = img.size
        
        # Determine Pixel BBox
        if i < len(raw_bboxes_pixel):
            # User provided a specific box
            bbox_px = raw_bboxes_pixel[i]
        else:
            # Default fallback: Full Image Dimensions
            # print(f"⚠️ No bbox provided for image {i}. Using full image dimensions [0, 0, {w}, {h}].")
            bbox_px = [0, 0, w, h]
            
        # Convert Pixel [xywh] -> Norm [xyxy] (0-1000)
        norm_box = _normalize_box(bbox_px, w, h)
        norm_bboxes_xyxy.append(norm_box)

    # 3. Build Prompt
    try:
        if job_input.get('prompt'):
            prompt_text = job_input['prompt']
        else:
            prompt_text = _build_prompt(job_input, norm_bboxes_xyxy, len(images))
    except ValueError as ve:
        return {"error": str(ve)}

    # 4. Construct Message
    content_list = []
    content_list.extend(images)
    content_list.append(prompt_text)

    messages = [{"role": "user", "content": content_list}]

    # 5. Inference
    try:
        raw_response = model.chat(
            msgs=messages,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.7,
            use_tts_template=False,
            enable_thinking=False
        )

        print(f"--- Raw Model Response --- \n{raw_response}\n--------------------------")
        result_json = parse_vlm_output(raw_response)

        # 6. Post-Process: Denormalize WITH PADDING
        if "refined_bboxes" in result_json and isinstance(result_json["refined_bboxes"], list):
            pixel_bboxes_xywh = []
            for i, box_norm in enumerate(result_json["refined_bboxes"]):
                if i < len(images) and isinstance(box_norm, list) and len(box_norm) == 4:
                    w, h = images[i].size
                    # Use new padding function
                    box_px = _denormalize_box_with_padding(box_norm, w, h, padding_pct=0.05)
                    pixel_bboxes_xywh.append(box_px)
            
            result_json["refined_bboxes"] = pixel_bboxes_xywh
            result_json["_note"] = "refined_bboxes returned in [x, y, w, h] pixel format with safety padding for SAM."

        return result_json

    except Exception as e:
        return {"error": f"Inference failed: {e}"}


# --- 4. Execution Logic ---

runpod.serverless.start({"handler": handler})

# if __name__ == "__main__":
#     print("🧪 Running in Local Test Mode...")
    
#     img_paths = [
#         "piano_test_img_1.jpg",
#         "piano_test_img_2.jpg",
#         "piano_test_img_3.jpg"
#     ]
    
#     # CASE: Testing NO bounding boxes (Should default to full image)
#     print("ℹ️ Note: Testing with NO input bboxes. Defaults should apply.")
#     orig_bboxes = [] 

#     test_input = {
#         "input": {
#             "image_paths": img_paths,
#             "input_category": "piano",
#             "bboxes": orig_bboxes # Empty list
#         }
#     }

#     try:
#         result = handler(test_input)
#         print("\n✅ Handler Output:")
#         print(json.dumps(result, indent=4))

#         if "refined_bboxes" in result:
#             print("\n🎨 Generating Visualization Images...")
#             refined_bboxes = result["refined_bboxes"]

#             for i, path in enumerate(img_paths):
#                 try:
#                     if not os.path.exists(path):
#                         print(f"⚠️ Warning: Could not find {path} to draw on.")
#                         continue
                    
#                     with Image.open(path) as raw_img:
#                         img = ImageOps.exif_transpose(raw_img)
#                         w, h = img.size
#                         draw = ImageDraw.Draw(img)

#                         # Draw Input (Red) - Either from input or full image default
#                         if i < len(orig_bboxes):
#                             x, y, wb, hb = orig_bboxes[i]
#                             draw.rectangle([x, y, x + wb, y + hb], outline="red", width=8)
#                         else:
#                             # Visualize the default full-image box
#                             draw.rectangle([0, 0, w, h], outline="red", width=8)

#                         # Draw Refined + Padded (Green)
#                         if i < len(refined_bboxes):
#                             rx, ry, rw, rh = refined_bboxes[i]
#                             if rw > 0 and rh > 0:
#                                 draw.rectangle([rx, ry, rx + rw, ry + rh], outline="green", width=5)

#                         save_filename = f"result_{path}"
#                         img.save(save_filename)
#                         print(f"   👉 Saved: {save_filename}")

#                 except Exception as viz_err:
#                     print(f"   ❌ Error visualizing image {i}: {viz_err}")

#     except Exception as e:
#         print(f"\n❌ Handler Failed: {e}")

# else:
#     runpod.serverless.start({"handler": handler})