from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import os
from itertools import combinations
from tqdm import tqdm  # Optional: for progress bar

folder = "frames_covla_1k"
output_csv = "embeddings/aligned_distances.csv"  # Different filename to avoid overwriting

embeddings = np.load('embeddings/image_embeddings.npy', allow_pickle=True)
metadata = pd.read_csv('embeddings/image_metadata.csv')

def aligned_distance(set1, set2, metric='euclidean'):
    """
    Compute the distance between two sets of vectors assuming they're already aligned.
    This computes the distance between corresponding vectors and sums them.
    
    Args:
        set1: numpy array of shape (N, D) where N is the number of vectors and D is the vector dimension
        set2: numpy array of shape (N, D) with the same number of vectors as set1
        metric: distance metric to use ('euclidean', 'sqeuclidean', 'cosine', etc.)
        
    Returns:
        float: the sum of distances between corresponding vector pairs
    """
    # Convert arrays to float type if they aren't already
    set1 = np.asarray(set1, dtype=np.float32)
    set2 = np.asarray(set2, dtype=np.float32)
    
    # Check if the sets have the same number of vectors
    if set1.shape[0] != set2.shape[0]:
        raise ValueError(f"Sets must have the same number of vectors. Got {set1.shape[0]} and {set2.shape[0]}")
    
    # Compute distances between corresponding vectors
    # For each i, calculate distance between set1[i] and set2[i]
    distances = np.array([
        cdist(set1[i:i+1], set2[i:i+1], metric)[0][0]
        for i in range(set1.shape[0])
    ])
    
    # Sum all distances
    total_distance = np.sum(distances)
    
    return total_distance

# Get list of image files
#image_files = os.listdir(folder)  # Adjust the slice as needed

image_dir = "frames_covla_1k"
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:300]

# Prepare DataFrame to store results
results = []

# Use combinations to avoid computing the same pair twice
total_comparisons = len(list(combinations(image_files, 2)))
print(f"Computing {total_comparisons} unique pairwise distances...")

for image_name_1, image_name_2 in tqdm(combinations(image_files, 2), total=total_comparisons):
    # Get embeddings for image 1
    image_index_1 = metadata[metadata['image_path'].str.contains(image_name_1)].index[0]
    image_embedding_1 = embeddings[image_index_1].astype(np.float32)
    
    # Get embeddings for image 2
    image_index_2 = metadata[metadata['image_path'].str.contains(image_name_2)].index[0]
    image_embedding_2 = embeddings[image_index_2].astype(np.float32)
    
    try:
        # Compute the aligned distance
        distance = aligned_distance(image_embedding_1, image_embedding_2)
        
        # Store result
        results.append({
            'image_1': image_name_1,
            'image_2': image_name_2,
            'distance': distance
        })
    except ValueError as e:
        # Skip if embeddings have different numbers of vectors
        print(f"Skipping {image_name_1} and {image_name_2}: {e}")

# Create DataFrame from results
distances_df = pd.DataFrame(results)

# Save to CSV
distances_df.to_csv(output_csv, index=False)
print(f"Distances saved to {output_csv}")

# Display some statistics
print("\nImages with smallest distance (most similar):")
most_similar = distances_df.loc[distances_df['distance'].idxmin()]
print(f"  {most_similar['image_1']} and {most_similar['image_2']}: {most_similar['distance']:.4f}")

print("\nImages with largest distance (most different):")
most_different = distances_df.loc[distances_df['distance'].idxmax()]
print(f"  {most_different['image_1']} and {most_different['image_2']}: {most_different['distance']:.4f}")

# Optional: If you want to find the N most similar/different pairs
n = 5  # Number of pairs to show
print(f"\nTop {n} most similar image pairs:")
for _, row in distances_df.nsmallest(n, 'distance').iterrows():
    print(f"  {row['image_1']} and {row['image_2']}: {row['distance']:.4f}")

print(f"\nTop {n} most different image pairs:")
for _, row in distances_df.nlargest(n, 'distance').iterrows():
    print(f"  {row['image_1']} and {row['image_2']}: {row['distance']:.4f}")

