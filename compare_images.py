from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Load the model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# Load the processor
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct"
)

def get_image_pixels(image_path):
    """Process an image and get its pixel values"""
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

    # Get the image pixels
    image_pixels = inputs.pixel_values
    return image_pixels

def get_vision_tower_embeddings(image_path):
    """Extract embeddings from the vision tower"""
    # Get preprocessed image pixels
    pixel_values = get_image_pixels(image_path)
    
    # Move to the same device as the model
    pixel_values = pixel_values.to(model.device)
    
    # Extract vision tower embeddings
    with torch.no_grad():
        # Pass through the vision encoder to get embeddings
        vision_outputs = model.vision_tower(pixel_values)
        
        # Get the last hidden state from the vision outputs
        vision_embeddings = vision_outputs.last_hidden_state
    
    return vision_embeddings

def get_pooled_vision_embeddings(image_path):
    """Get mean pooled vision embeddings"""
    # Get the full vision embeddings 
    vision_embeddings = get_vision_tower_embeddings(image_path)
    
    # Mean pool over the sequence dimension to get a single vector per image
    pooled_embeddings = vision_embeddings.mean(dim=1)
    
    return pooled_embeddings.detach().cpu().numpy()

def compare_images(image_paths):
    """Compare a list of images using vision tower embeddings"""
    # Get embeddings for all images
    embeddings = []
    for image_path in image_paths:
        embedding = get_pooled_vision_embeddings(image_path)
        embeddings.append(embedding)
    
    # Compute cosine similarity between all pairs
    similarity_matrix = cosine_similarity(embeddings)
    
    return similarity_matrix

def visualize_similarity(image_paths, similarity_matrix):
    """Visualize the similarity matrix and the corresponding images"""
    n = len(image_paths)
    
    # Create a figure
    fig = plt.figure(figsize=(15, 10))
    
    # Plot the similarity matrix
    ax1 = fig.add_subplot(121)
    im = ax1.imshow(similarity_matrix, cmap='viridis')
    ax1.set_title('Cosine Similarity Matrix')
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels([os.path.basename(path) for path in image_paths], rotation=45)
    ax1.set_yticklabels([os.path.basename(path) for path in image_paths])
    
    # Add color bar
    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label('Cosine Similarity')
    
    # Plot the images
    ax2 = fig.add_subplot(122)
    ax2.axis('off')
    
    # Create a grid of images
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    for i in range(n):
        ax = fig.add_subplot(rows, cols, i+1)
        img = Image.open(image_paths[i][7:] if image_paths[i].startswith("file://") else image_paths[i])
        ax.imshow(img)
        ax.set_title(f"Image {i+1}: {os.path.basename(image_paths[i])}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('image_similarity.png')
    plt.show()

def find_most_similar(query_image_path, reference_image_paths):
    """Find the most similar image to a query image"""
    # Get embedding for query image
    query_embedding = get_pooled_vision_embeddings(query_image_path)
    
    # Get embeddings for reference images
    reference_embeddings = []
    for image_path in reference_image_paths:
        embedding = get_pooled_vision_embeddings(image_path)
        reference_embeddings.append(embedding)
    
    # Compute cosine similarity between query and references
    similarities = []
    for embedding in reference_embeddings:
        similarity = cosine_similarity(query_embedding, embedding)[0][0]
        similarities.append(similarity)
    
    # Find the index of the most similar image
    most_similar_idx = np.argmax(similarities)
    
    return most_similar_idx, similarities

if __name__ == "__main__":
    # Example usage - replace with your own images
    image_folder = "your_image_folder"  # Change this to your image folder
    
    # Look for image files in the folder
    image_files = [f for f in os.listdir(image_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(image_files) < 2:
        print(f"Need at least 2 images in {image_folder}. Found {len(image_files)}.")
        exit(1)
    
    # Take at most 6 images for the example
    image_files = image_files[:min(6, len(image_files))]
    
    # Get full paths
    image_paths = [os.path.join(image_folder, f) for f in image_files]
    image_uris = [f"file://{os.path.abspath(path)}" for path in image_paths]
    
    print(f"Comparing {len(image_paths)} images...")
    
    # Compare all images
    similarity_matrix = compare_images(image_uris)
    
    # Visualize results
    visualize_similarity(image_uris, similarity_matrix)
    
    # Find the most similar pair
    max_similarity = 0
    most_similar_pair = (0, 0)
    
    for i in range(len(image_paths)):
        for j in range(i+1, len(image_paths)):
            if similarity_matrix[i][j] > max_similarity:
                max_similarity = similarity_matrix[i][j]
                most_similar_pair = (i, j)
    
    print(f"Most similar pair: {os.path.basename(image_paths[most_similar_pair[0]])} and "
          f"{os.path.basename(image_paths[most_similar_pair[1]])} "
          f"with similarity: {max_similarity:.4f}")
    
    # Example of finding similar images to a query
    query_idx = 0  # Use the first image as query
    reference_indices = list(range(1, len(image_paths)))  # Use the rest as references
    
    query_path = image_uris[query_idx]
    reference_paths = [image_uris[i] for i in reference_indices]
    
    most_similar_idx, similarities = find_most_similar(query_path, reference_paths)
    
    print(f"Most similar to {os.path.basename(image_paths[query_idx])}: "
          f"{os.path.basename(image_paths[reference_indices[most_similar_idx]])}"
          f" with similarity: {similarities[most_similar_idx]:.4f}")
    
    # Print all similarities for reference
    for i, ref_idx in enumerate(reference_indices):
        print(f"Similarity with {os.path.basename(image_paths[ref_idx])}: {similarities[i]:.4f}") 