import argparse
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser(description="Image Search using Qwen2.5-VL Vision Tower Embeddings")
parser.add_argument("--image_dir", required=True, help="Directory containing images to search")
parser.add_argument("--query_image", required=True, help="Path to query image")
parser.add_argument("--k", type=int, default=5, help="Number of results to return")
parser.add_argument("--build_index", action="store_true", help="Whether to build the index even if it exists")
parser.add_argument("--output_dir", default="search_results", help="Directory to save results")
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Check if index exists
index_file = os.path.join(args.image_dir, "vision_tower_index.npz")
should_create_index = args.build_index or not os.path.exists(index_file)

# Function to get vision tower embeddings
def get_vision_tower_embeddings(model, processor, image_path):
    """Extract embeddings from the vision tower of Qwen2.5-VL"""
    # Prepare input
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
    
    # Process image
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    # Move to device
    pixel_values = inputs.pixel_values.to(model.device)
    
    # Get embeddings
    with torch.no_grad():
        vision_outputs = model.vision_tower(pixel_values)
        vision_embeddings = vision_outputs.last_hidden_state
        # Mean pooling
        pooled_embeddings = vision_embeddings.mean(dim=1)
    
    return pooled_embeddings.detach().cpu().numpy()

# Build search index if needed
if should_create_index:
    print(f"Building index for images in {args.image_dir}...")
    
    # Load model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    
    # Get all image files
    image_files = [f for f in os.listdir(args.image_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'))]
    
    # Compute embeddings for all images
    embeddings = []
    image_paths = []
    
    for img_file in tqdm(image_files):
        try:
            img_path = os.path.join(args.image_dir, img_file)
            img_uri = f"file://{os.path.abspath(img_path)}"
            
            # Get embeddings
            embedding = get_vision_tower_embeddings(model, processor, img_uri)
            
            # Store
            embeddings.append(embedding)
            image_paths.append(img_path)
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    # Save index
    np.savez(index_file, 
             embeddings=np.vstack(embeddings), 
             image_paths=np.array(image_paths))
    
    print(f"Index built for {len(embeddings)} images and saved to {index_file}")
else:
    print(f"Using existing index from {index_file}")

# Load index
data = np.load(index_file, allow_pickle=True)
index_embeddings = data['embeddings']
index_paths = data['image_paths']

# Process query image
print(f"Processing query image: {args.query_image}")

# Load model (if we didn't build the index)
if not should_create_index:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# Get embedding for query image
query_uri = f"file://{os.path.abspath(args.query_image)}"
query_embedding = get_vision_tower_embeddings(model, processor, query_uri)

# Compute similarity with index
similarities = cosine_similarity(query_embedding, index_embeddings).flatten()

# Get top k results
k = min(args.k, len(index_paths))
top_indices = similarities.argsort()[-k:][::-1]
top_paths = [index_paths[i] for i in top_indices]
top_similarities = [similarities[i] for i in top_indices]

# Display results
print("\nSearch Results:")
for i, (path, sim) in enumerate(zip(top_paths, top_similarities)):
    print(f"{i+1}. {os.path.basename(path)} (Similarity: {sim:.4f})")

# Visualize results
plt.figure(figsize=(15, 8))

# Plot query image
plt.subplot(1, k+1, 1)
query_img = Image.open(args.query_image)
plt.imshow(query_img)
plt.title(f"Query: {os.path.basename(args.query_image)}")
plt.axis('off')

# Plot results
for i, (path, sim) in enumerate(zip(top_paths, top_similarities)):
    plt.subplot(1, k+1, i+2)
    result_img = Image.open(path)
    plt.imshow(result_img)
    plt.title(f"#{i+1}: {os.path.basename(path)}\nSimilarity: {sim:.4f}")
    plt.axis('off')

plt.tight_layout()
result_file = os.path.join(args.output_dir, f"search_results_{os.path.basename(args.query_image)}.png")
plt.savefig(result_file)
print(f"\nResults visualization saved to {result_file}")

# Optional: Free GPU memory
del model
torch.cuda.empty_cache()

print("Done!") 