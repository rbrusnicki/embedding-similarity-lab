from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.bfloat16,
    #attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# Function to get vision tower embeddings
def get_vision_tower_embeddings(image_path):
    # Get preprocessed image inputs
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
    
    # Move inputs to the same device as the model
    pixel_values = inputs.pixel_values.to(model.device)
    image_grid_thw = inputs.image_grid_thw.to(model.device)
    
    # Extract vision embeddings with no gradient computation
    with torch.no_grad():
        # Pass both required parameters to the vision model
        # The output is directly the embeddings tensor
        vision_embeddings = model.visual(
            pixel_values, 
            grid_thw=image_grid_thw
        )
    
    return vision_embeddings

# Directory containing the images
image_dir = "frames_covla_1k"
# Where to save the embeddings
output_dir = "embeddings"
embeddings_file = os.path.join(output_dir, "vision_tower_embeddings.npy")
metadata_file = os.path.join(output_dir, "vision_tower_metadata.csv")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get list of all images in the directory
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(f"Found {len(image_files)} images in {image_dir}")

# Lists to store embeddings and metadata
all_embeddings = []
metadata = []

# Add this debugging code to your script to see the model structure
def print_model_structure(model, depth=0, max_depth=2):
    for name, module in model.named_children():
        print('  ' * depth + f"{name}: {type(module).__name__}")
        if depth < max_depth:
            print_model_structure(module, depth + 1, max_depth)

# Call this function before trying to use the model
#print_model_structure(model)

# Process each image
for image_file in tqdm(image_files, desc="Processing images"):
    image_path = os.path.join(image_dir, image_file)
    try:
        # Get full path
        absolute_path = os.path.abspath(image_path)
        # Convert to file:// format
        file_uri = f"file://{absolute_path}"
        
        # Get vision tower embedding
        embedding = get_vision_tower_embeddings(file_uri)
        
        
        # Store embedding and metadata
        all_embeddings.append(embedding.detach().cpu().to(torch.float32).numpy())
        metadata.append({
            'image_path': image_path,
            'shape': embedding.shape
        })
        
        # Optional: print progress periodically
        if len(all_embeddings) % 10 == 0:
            print(f"Processed {len(all_embeddings)}/{len(image_files)} images")
    
    except Exception as e:
        print(f"Error processing {image_file}: {e}")

# Check if we have any embeddings
if len(all_embeddings) == 0:
    print("No embeddings were successfully extracted. Check the error messages above.")
    exit()

# Save the embeddings
print(f"Saving {len(all_embeddings)} embeddings to {embeddings_file}")
np.save(embeddings_file, np.array(all_embeddings, dtype=object))

# Save metadata
print(f"Saving metadata to {metadata_file}")
metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv(metadata_file, index=False)

print("Done!")
