import os
import numpy as np
from diffusers import DiffusionPipeline
import torch
import imageio
from PIL import Image


# Set environment variables
os.environ["XFORMERS_IGNORE_MISMATCH"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
torch.backends.cuda.matmul.allow_tf32 = True

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

# Initialize the pipeline
# pipe = DiffusionPipeline.from_pretrained(
#     "cerspense/zeroscope_v2_576w",
#     torch_dtype=torch.float16
# )
# Change this line in future runs:
pipe = DiffusionPipeline.from_pretrained(
    "./zeroscope_model",  # 🟢 Changed to local path 🟢
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.enable_model_cpu_offload()  # Faster than sequential
pipe.enable_xformers_memory_efficient_attention()

# pipe.save_pretrained("./zeroscope_model")

# Generate video frames
prompt = "A dog playing in the park with other dogs."
video_frames = pipe(
    prompt,
    num_frames=32,
    height=224,
    width=384,
    num_inference_steps=25 
).frames[0]  # Extract the actual frames :cite[1]

# Process frames to ensure correct format
processed_frames = []
for i, frame in enumerate(video_frames):
    # Convert PIL Image to NumPy array
    frame = np.array(frame)
    
    # Transpose channels-first (C, H, W) to channels-last (H, W, C)
    if frame.ndim == 3 and frame.shape[0] in [1, 3, 4]:
        frame = frame.transpose(1, 2, 0)  # Fix channel order :cite[7]
    
    # Check the number of channels
    if frame.ndim == 2:
        # Grayscale image, convert to RGB
        frame = np.stack((frame,) * 3, axis=-1)
    elif frame.shape[2] == 4:
        # RGBA image, convert to RGB
        frame = frame[:, :, :3]
    elif frame.shape[2] == 1:
        # Single channel image, convert to RGB
        frame = np.concatenate([frame] * 3, axis=-1)
    
    # Convert to uint8
    if frame.dtype != np.uint8:
        frame = (frame * 255).astype(np.uint8)
    
    processed_frames.append(frame)

# Validate frames (optional debug step)
for frame in processed_frames:
    assert frame.ndim == 3, f"Invalid frame dimensions: {frame.shape}"
    assert frame.shape[2] in [1, 2, 3, 4], f"Invalid channels: {frame.shape}"

# Save the video
imageio.mimsave(
    'outputs/output.mp4',
    processed_frames,
    fps=8,
    codec='libx264',
    output_params=['-pix_fmt', 'yuv420p']
)

print("Video saved successfully!")
currently i am getting a 3 sec video but i need 10 sec video