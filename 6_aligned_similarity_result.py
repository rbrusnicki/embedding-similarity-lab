import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
from scipy.spatial.distance import cdist

# Choose which data to process
use_vision_tower_data = True  # Set to False to use original image embeddings

# Configuration
if use_vision_tower_data:
    similarities_csv = "embeddings/vt_aligned_similarities.csv"
    image_dir = "frames_covla_1k"
    output_dir = "vt_aligned_similarity_images"
    embeddings = np.load('embeddings/vision_tower_embeddings.npy', allow_pickle=True)
    metadata = pd.read_csv('embeddings/vision_tower_metadata.csv')
    n_pairs = 10  # Number of most similar/different pairs to visualize
else:
    similarities_csv = "embeddings/aligned_similarities.csv"
    image_dir = "frames_covla_1k"
    output_dir = "aligned_similarity_images"
    embeddings = np.load('embeddings/image_embeddings.npy', allow_pickle=True)
    metadata = pd.read_csv('embeddings/image_metadata.csv')
    n_pairs = 10  # Number of most similar/different pairs to visualize

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def calculate_pairwise_similarities(embedding1, embedding2):
    """Calculate pairwise cosine similarities between corresponding vectors in two embeddings."""
    # Make sure embeddings are in correct format
    embedding1 = np.asarray(embedding1, dtype=np.float32)
    embedding2 = np.asarray(embedding2, dtype=np.float32)
    
    # Calculate cosine distance between each corresponding pair
    similarities = np.array([
        1 - cdist(embedding1[i:i+1], embedding2[i:i+1], 'cosine')[0][0]
        for i in range(embedding1.shape[0])
    ])
    
    return similarities

def create_complete_visualization(img1_path, img2_path, similarity, pairwise_similarities, output_path, title):
    """Create a visualization with two images and a similarity grid side by side."""
    # Load images
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    
    # Resize images to a standard size while maintaining aspect ratio
    max_size = (512, 512)
    img1.thumbnail(max_size, Image.LANCZOS)
    img2.thumbnail(max_size, Image.LANCZOS)
    
    # Create figure with 3 subplots in a row
    fig = plt.figure(figsize=(18, 6))
    
    # Set up grid for subplots
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2])
    
    # Add the first image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(np.array(img1))
    ax1.set_title(os.path.basename(img1_path))
    ax1.axis('off')
    
    # Add the second image
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(np.array(img2))
    ax2.set_title(os.path.basename(img2_path))
    ax2.axis('off')
    
    # Add the grid visualization
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Process the similarities for the grid
    target_size = (11, 17)
    total_elements = target_size[0] * target_size[1]
    
    if len(pairwise_similarities) < total_elements:
        # Pad with zeros if we have fewer elements
        padded = np.zeros(total_elements)
        padded[:len(pairwise_similarities)] = pairwise_similarities
        pairwise_similarities = padded
    elif len(pairwise_similarities) > total_elements:
        # Truncate if we have more elements
        pairwise_similarities = pairwise_similarities[:total_elements]
    
    # Reshape to 11x17 grid
    grid = pairwise_similarities.reshape(target_size)
    
    # Create heatmap with swapped colors (blue to red)
    cmap = plt.cm.RdBu  # Blue to Red colormap (swapped as requested)
    im = ax3.imshow(grid, cmap=cmap, vmin=0, vmax=1)
    
    # Remove ticks from grid axes
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    # Add text annotations to each cell
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid[i, j]
            # Format value as specified: two decimal places, no leading zero
            text_value = f".{int(value * 100):02d}"
            
            # Determine text color based on value
            if 0.25 <= value <= 0.75:
                text_color = 'black'
            else:
                text_color = 'white'
            
            # Add the text to the cell with increased font size
            ax3.text(j, i, text_value, ha='center', va='center', 
                    color=text_color, fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Similarity (0=Low, 1=High)')
    
    ax3.set_title('Embedding Similarity Grid')
    
    # Add overall title with similarity information
    embedding_type = "Vision Tower" if use_vision_tower_data else "Standard"
    plt.suptitle(f"{title}\n{embedding_type} Aligned Similarity: {similarity:.4f}", fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved complete visualization: {output_path}")

def get_image_embeddings(image_name):
    """Get the embeddings for a specific image."""
    image_index = metadata[metadata['image_path'].str.contains(image_name)].index[0]
    return embeddings[image_index].astype(np.float32)

# Main function
def main():
    # Load similarities CSV
    print(f"Loading aligned similarities from {similarities_csv}")
    try:
        similarities_df = pd.read_csv(similarities_csv)
    except FileNotFoundError:
        print(f"Error: File {similarities_csv} not found!")
        return
    
    print(f"Found {len(similarities_df)} similarity pairs")
    
    # Get most similar pairs (largest similarities)
    similar_pairs = similarities_df.nlargest(n_pairs, 'similarity')
    embedding_type = "Vision Tower" if use_vision_tower_data else "Standard"
    print(f"\nProcessing {n_pairs} most similar image pairs ({embedding_type} aligned similarity)...")
    
    for i, (_, row) in enumerate(similar_pairs.iterrows()):
        img1_path = os.path.join(image_dir, row['image_1'])
        img2_path = os.path.join(image_dir, row['image_2'])
        similarity = row['similarity']
        
        # Get embeddings for the two images
        embedding1 = get_image_embeddings(row['image_1'])
        embedding2 = get_image_embeddings(row['image_2'])
        
        # Calculate pairwise similarities
        pairwise_similarities = calculate_pairwise_similarities(embedding1, embedding2)
        
        # Create the complete visualization
        output_path = os.path.join(output_dir, f"similar_pair_{i+1}_complete.png")
        create_complete_visualization(
            img1_path, 
            img2_path, 
            similarity,
            pairwise_similarities,
            output_path, 
            f"Most Similar Pair #{i+1} ({embedding_type} Aligned)"
        )
    
    # Get most different pairs (smallest similarities)
    different_pairs = similarities_df.nsmallest(n_pairs, 'similarity')
    print(f"\nProcessing {n_pairs} most different image pairs ({embedding_type} aligned similarity)...")
    
    for i, (_, row) in enumerate(different_pairs.iterrows()):
        img1_path = os.path.join(image_dir, row['image_1'])
        img2_path = os.path.join(image_dir, row['image_2'])
        similarity = row['similarity']
        
        # Get embeddings for the two images
        embedding1 = get_image_embeddings(row['image_1'])
        embedding2 = get_image_embeddings(row['image_2'])
        
        # Calculate pairwise similarities
        pairwise_similarities = calculate_pairwise_similarities(embedding1, embedding2)
        
        # Create the complete visualization
        output_path = os.path.join(output_dir, f"different_pair_{i+1}_complete.png")
        create_complete_visualization(
            img1_path, 
            img2_path, 
            similarity,
            pairwise_similarities,
            output_path, 
            f"Most Different Pair #{i+1} ({embedding_type} Aligned)"
        )
    
    print(f"\nAll {embedding_type} aligned similarity visualizations saved to {output_dir}!")

if __name__ == "__main__":
    main()