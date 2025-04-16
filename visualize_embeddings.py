import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize vision tower embeddings using t-SNE")
    parser.add_argument("--embeddings_file", required=True, help="Path to the embeddings .npy file")
    parser.add_argument("--metadata_file", required=True, help="Path to the metadata .csv file")
    parser.add_argument("--output_dir", default="tsne_plots", help="Directory to save visualization results")
    parser.add_argument("--perplexity", type=int, default=30, help="Perplexity parameter for t-SNE")
    parser.add_argument("--n_components", type=int, default=2, help="Number of components for t-SNE (2 or 3)")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    parser.add_argument("--show_images", action="store_true", help="Show thumbnail images in the plot")
    return parser.parse_args()

def load_embeddings(embeddings_file, metadata_file):
    """Load embeddings and metadata"""
    # Load embeddings (potentially with different shapes)
    embeddings_data = np.load(embeddings_file, allow_pickle=True)
    
    # Handle different data types
    if isinstance(embeddings_data, np.ndarray):
        embeddings = embeddings_data
    else:
        # If it's a .npz file
        if hasattr(embeddings_data, 'files'):
            # Assuming the first array contains the embeddings
            key = embeddings_data.files[0]
            embeddings = embeddings_data[key]
        else:
            raise ValueError("Unsupported embeddings file format")
    
    # For the case where embeddings are stored as object arrays of different shapes
    if embeddings.dtype == np.dtype('O'):
        # Try to stack arrays after pooling them if necessary
        processed_embeddings = []
        for emb in embeddings:
            # If it's 3D (batch, seq, hidden), pool to 2D
            if len(emb.shape) == 3:
                emb = emb.mean(axis=1)  # Mean pooling over sequence dimension
            # If it's still not 2D, we have a problem
            if len(emb.shape) != 2:
                raise ValueError(f"Unexpected embedding shape: {emb.shape}")
            processed_embeddings.append(emb.reshape(1, -1))  # Reshape to (1, hidden_dim)
        
        embeddings = np.vstack(processed_embeddings)
    
    # Load metadata
    metadata = pd.read_csv(metadata_file)
    
    return embeddings, metadata

def prepare_embeddings_for_tsne(embeddings):
    """Prepare embeddings for t-SNE by ensuring they're 2D"""
    # If the embeddings are 3D (batch, seq, hidden), pool them to 2D
    if len(embeddings.shape) == 3:
        embeddings = embeddings.mean(axis=1)
    
    # Get 2D shape (batch, features)
    if len(embeddings.shape) != 2:
        raise ValueError(f"Embeddings must be 2D after processing, got shape {embeddings.shape}")
    
    return embeddings

def perform_tsne(embeddings, n_components=2, perplexity=30, random_state=42):
    """Perform t-SNE dimensionality reduction"""
    print(f"Performing t-SNE with {n_components} components and perplexity {perplexity}...")
    
    # Check if the number of samples is less than perplexity*3
    if embeddings.shape[0] < perplexity * 3:
        print(f"Warning: Number of samples ({embeddings.shape[0]}) is less than 3*perplexity ({perplexity*3}).")
        print(f"Reducing perplexity to {embeddings.shape[0] // 3}.")
        perplexity = max(5, embeddings.shape[0] // 3)
    
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    embeddings_tsne = tsne.fit_transform(embeddings)
    
    return embeddings_tsne

def get_image_for_plot(path, zoom=0.1):
    """Load and prepare an image for embedding in a plot"""
    try:
        # Handle file:// URIs
        if path.startswith("file://"):
            path = path[7:]
        
        img = Image.open(path)
        img = img.convert("RGB")  # Ensure RGB format
        
        return OffsetImage(img, zoom=zoom)
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None

def plot_2d_tsne(embeddings_tsne, metadata, output_path, show_images=False):
    """Create a 2D t-SNE plot"""
    plt.figure(figsize=(12, 10))
    
    # Create scatter plot
    plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], alpha=0.7, s=30)
    
    # If image thumbnails are requested
    if show_images:
        for i, (x, y) in enumerate(embeddings_tsne):
            img_path = metadata.iloc[i]['image_path']
            img = get_image_for_plot(img_path, zoom=0.1)
            if img is not None:
                ab = AnnotationBbox(img, (x, y), frameon=False, pad=0)
                plt.gca().add_artist(ab)
    else:
        # Add image indices as annotations
        for i, (x, y) in enumerate(embeddings_tsne):
            plt.annotate(str(i), xy=(x, y), xytext=(3, 3), textcoords="offset points")
    
    plt.title('t-SNE Visualization of Vision Tower Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_3d_tsne(embeddings_tsne, metadata, output_path):
    """Create a 3D t-SNE plot"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create scatter plot
    scatter = ax.scatter(
        embeddings_tsne[:, 0], 
        embeddings_tsne[:, 1], 
        embeddings_tsne[:, 2],
        alpha=0.7, 
        s=30
    )
    
    # Add image indices as annotations
    for i, (x, y, z) in enumerate(embeddings_tsne):
        ax.text(x, y, z, str(i), size=8)
    
    ax.set_title('3D t-SNE Visualization of Vision Tower Embeddings')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def create_image_grid(metadata, output_path, max_images=100, grid_size=10):
    """Create a grid of images with their indices for reference"""
    n_images = min(len(metadata), max_images)
    n_cols = min(grid_size, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    for i in range(n_images):
        img_path = metadata.iloc[i]['image_path']
        try:
            # Handle file:// URIs
            if img_path.startswith("file://"):
                img_path = img_path[7:]
                
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].set_title(f"Index: {i}")
            axes[i].axis('off')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            axes[i].text(0.5, 0.5, f"Error loading\nIndex: {i}", ha='center', va='center')
            axes[i].axis('off')
    
    # Turn off remaining axes
    for j in range(n_images, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load embeddings and metadata
    embeddings, metadata = load_embeddings(args.embeddings_file, args.metadata_file)
    print(f"Loaded {len(embeddings)} embeddings")
    
    # Prepare embeddings for t-SNE
    embeddings = prepare_embeddings_for_tsne(embeddings)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Perform t-SNE
    embeddings_tsne = perform_tsne(
        embeddings, 
        n_components=args.n_components, 
        perplexity=args.perplexity,
        random_state=args.random_state
    )
    
    # Create reference image grid
    image_grid_path = os.path.join(args.output_dir, "image_grid.png")
    create_image_grid(metadata, image_grid_path)
    print(f"Created image grid: {image_grid_path}")
    
    # Generate t-SNE visualization
    if args.n_components == 2:
        tsne_plot_path = os.path.join(args.output_dir, "tsne_2d.png")
        plot_2d_tsne(embeddings_tsne, metadata, tsne_plot_path, args.show_images)
        print(f"Created 2D t-SNE plot: {tsne_plot_path}")
    else:
        tsne_plot_path = os.path.join(args.output_dir, "tsne_3d.png")
        plot_3d_tsne(embeddings_tsne, metadata, tsne_plot_path)
        print(f"Created 3D t-SNE plot: {tsne_plot_path}")
    
    # Save t-SNE coordinates
    tsne_coords_path = os.path.join(args.output_dir, "tsne_coordinates.csv")
    tsne_df = pd.DataFrame(embeddings_tsne, columns=[f"tsne_{i+1}" for i in range(args.n_components)])
    tsne_df['image_path'] = metadata['image_path']
    tsne_df.to_csv(tsne_coords_path, index=False)
    print(f"Saved t-SNE coordinates: {tsne_coords_path}")

if __name__ == "__main__":
    main() 