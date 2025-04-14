from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import os
from itertools import combinations
from tqdm import tqdm  # Optional: for progress bar

folder = "frames_covla_1k"
output_csv = "embeddings/chamfer_distances.csv"

embeddings = np.load('embeddings/image_embeddings.npy', allow_pickle=True)
metadata = pd.read_csv('embeddings/image_metadata.csv')

def chamfer_distance_vectorized(set1, set2):
    """
    Compute the Chamfer distance between two sets of vectors using vectorized operations.
    
    Args:
        set1: numpy array of shape (N, D) where N is the number of vectors and D is the vector dimension
        set2: numpy array of shape (M, D) where M is the number of vectors and D is the vector dimension
        
    Returns:
        float: the Chamfer distance between the two sets
    """
    
    # Convert arrays to float type if they aren't already
    set1 = np.asarray(set1, dtype=np.float32)
    set2 = np.asarray(set2, dtype=np.float32)
    
    # Compute pairwise squared Euclidean distances between all points
    dist_matrix = cdist(set1, set2, 'sqeuclidean')
    
    # For each point in set1, find the minimum distance to any point in set2
    min_dist_1_to_2 = np.min(dist_matrix, axis=1)
    
    # For each point in set2, find the minimum distance to any point in set1
    min_dist_2_to_1 = np.min(dist_matrix, axis=0)
    
    # Compute the average in both directions and sum them
    chamfer_dist = np.mean(min_dist_1_to_2) + np.mean(min_dist_2_to_1)
    
    return chamfer_dist

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
    
    # Compute distance
    distance = chamfer_distance_vectorized(image_embedding_1, image_embedding_2)
    
    # Store result
    results.append({
        'image_1': image_name_1,
        'image_2': image_name_2,
        'distance': distance
    })

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

