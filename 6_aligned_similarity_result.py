import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np

# Configuration
similarities_csv = "embeddings/aligned_similarities.csv"
image_dir = "frames_covla_1k"
output_dir = "aligned_similarity_images"
n_pairs = 10  # Number of most similar/different pairs to visualize

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def create_comparison_image(img1_path, img2_path, similarity, output_path, title):
    """Create a side-by-side comparison of two images with similarity info."""
    # Load images
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    
    # Resize images to a standard size while maintaining aspect ratio
    max_size = (512, 512)
    img1.thumbnail(max_size, Image.LANCZOS)
    img2.thumbnail(max_size, Image.LANCZOS)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display images
    ax1.imshow(np.array(img1))
    ax1.set_title(os.path.basename(img1_path))
    ax1.axis('off')
    
    ax2.imshow(np.array(img2))
    ax2.set_title(os.path.basename(img2_path))
    ax2.axis('off')
    
    # Add overall title with similarity information
    plt.suptitle(f"{title}\nAligned Similarity: {similarity:.4f}", fontsize=16)
    
    # Add spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")

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
    print(f"\nProcessing {n_pairs} most similar image pairs (aligned similarity)...")
    
    for i, (_, row) in enumerate(similar_pairs.iterrows()):
        img1_path = os.path.join(image_dir, row['image_1'])
        img2_path = os.path.join(image_dir, row['image_2'])
        similarity = row['similarity']
        
        output_path = os.path.join(output_dir, f"similar_pair_{i+1}.png")
        create_comparison_image(
            img1_path, 
            img2_path, 
            similarity, 
            output_path, 
            f"Most Similar Pair #{i+1} (Aligned)"
        )
    
    # Get most different pairs (smallest similarities)
    different_pairs = similarities_df.nsmallest(n_pairs, 'similarity')
    print(f"\nProcessing {n_pairs} most different image pairs (aligned similarity)...")
    
    for i, (_, row) in enumerate(different_pairs.iterrows()):
        img1_path = os.path.join(image_dir, row['image_1'])
        img2_path = os.path.join(image_dir, row['image_2'])
        similarity = row['similarity']
        
        output_path = os.path.join(output_dir, f"different_pair_{i+1}.png")
        create_comparison_image(
            img1_path, 
            img2_path, 
            similarity, 
            output_path, 
            f"Most Different Pair #{i+1} (Aligned)"
        )
    
    print(f"\nAll aligned similarity images saved to {output_dir}!")

if __name__ == "__main__":
    main()