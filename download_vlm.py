# download_model.py
import os
from huggingface_hub import snapshot_download

# --- Configuration ---
MODEL_ID = "openbmb/MiniCPM-V-4_5"

# Use RunPod's persistent volume if available, otherwise use the workspace
if os.path.exists('/runpod-volume'):
    base_volume_path = '/runpod-volume'
else:
    base_volume_path = '/workspace'

# Define the local directory where the model will be saved
local_dir_path = os.path.join(base_volume_path, MODEL_ID.split('/')[-1])

print(f"Downloading model '{MODEL_ID}' to: {local_dir_path}")

# --- Download the Model ---
snapshot_download(
    repo_id=MODEL_ID,
    repo_type="model",
    local_dir=local_dir_path,
    local_dir_use_symlinks=False,  # Recommended for RunPod storage
    ignore_patterns=["*.safetensors.index.json", "*.safetensors"], # Ignore safetensors if you only need pytorch weights
)

print("\n✅ Download complete!")
print(f"Model is saved in: {local_dir_path}")