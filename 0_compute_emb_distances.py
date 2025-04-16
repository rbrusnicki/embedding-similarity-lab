from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# # default: Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="cuda"
# )

# # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-3B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

# Function to process a single image and get its embedding
def get_image_embedding(image_path):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": " "},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Get the image embedding
    image_embeds = inputs.pixel_values
    return image_embeds

# Directory containing the images
image_dir = "frames_covla_1k"
# Where to save the embeddings
output_dir = "embeddings"
embeddings_file = os.path.join(output_dir, "image_embeddings.npy")
metadata_file = os.path.join(output_dir, "image_metadata.csv")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get list of all images in the directory
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(f"Found {len(image_files)} images in {image_dir}")

# Initialize metadata list
metadata = []

# Create an empty .npz file to start
np.savez(os.path.join(output_dir, "image_embeddings.npz"))

# Process each image and save incrementally
for i, image_file in enumerate(tqdm(image_files, desc="Processing images")):
    image_path = os.path.join(image_dir, image_file)
    try:
        # Get full path
        absolute_path = os.path.abspath(image_path)
        # Convert to file:// format
        file_uri = f"file://{absolute_path}"
        
        # Get embedding
        embedding = get_image_embedding(file_uri)
        
        # Save embedding to an individual file
        embedding_file = os.path.join(output_dir, f"embedding_{i}.npy")
        np.save(embedding_file, embedding.detach().cpu().numpy())
        
        # Store metadata
        metadata.append({
            'image_path': image_path,
            'embedding_file': embedding_file,
            'shape': embedding.shape
        })
        
        # Optional: print progress periodically
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(image_files)} images")
    
    except Exception as e:
        print(f"Error processing {image_file}: {e}")

# Check if we have any metadata
if len(metadata) == 0:
    print("No embeddings were successfully extracted. Check the error messages above.")
    exit()

# Save metadata
print(f"Saving metadata to {metadata_file}")
metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv(metadata_file, index=False)

print("Done!")
