from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import os
from itertools import combinations
from tqdm import tqdm  # Optional: for progress bar

# Choose which data to process
use_vision_tower_data = False  # Set to False to use original image embeddings

if use_vision_tower_data:
    # New data from 7_emb_after_vt.py
    folder = "frames_covla_1k"
    output_csv = "embeddings/vt_aligned_similarities.csv"
    
    embeddings = np.load('embeddings/vision_tower_embeddings.npy', allow_pickle=True)
    metadata = pd.read_csv('embeddings/vision_tower_metadata.csv')
else:
    # Original data from 0_compute_emb_distances.py
    folder = "frames_covla_1k"
    output_csv = "embeddings/aligned_similarities.csv"
    
    embeddings = np.load('embeddings/image_embeddings.npy', allow_pickle=True)
    metadata = pd.read_csv('embeddings/image_metadata.csv')

def aligned_similarity(set1, set2):
    """
    Compute the cosine similarity between two sets of vectors assuming they're already aligned.
    This computes the similarity between corresponding vectors and averages them.
    
    Args:
        set1: numpy array of shape (N, D) where N is the number of vectors and D is the vector dimension
        set2: numpy array of shape (N, D) with the same number of vectors as set1
        
    Returns:
        float: the average cosine similarity between corresponding vector pairs
    """
    # Convert arrays to float type if they aren't already
    set1 = np.asarray(set1, dtype=np.float32)
    set2 = np.asarray(set2, dtype=np.float32)
    
    # Check if the sets have the same number of vectors
    if set1.shape[0] != set2.shape[0]:
        raise ValueError(f"Sets must have the same number of vectors. Got {set1.shape[0]} and {set2.shape[0]}")
    
    # Compute cosine similarities between corresponding vectors
    # For each i, calculate similarity between set1[i] and set2[i]
    similarities = np.array([
        cdist(set1[i:i+1], set2[i:i+1], 'cosine')[0][0]
        for i in range(set1.shape[0])
    ])
    
    # Convert cosine distance to similarity (1 - distance)
    similarities = 1 - similarities
    
    # Average all similarities
    avg_similarity = np.mean(similarities)
    
    return avg_similarity

# Get list of image files
image_dir = "frames_covla_1k"
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Prepare DataFrame to store results
results = []

# Use combinations to avoid computing the same pair twice
total_comparisons = len(list(combinations(image_files, 2)))
print(f"Computing {total_comparisons} unique pairwise similarities...")

for image_name_1, image_name_2 in tqdm(combinations(image_files, 2), total=total_comparisons):
    # Get embeddings for image 1
    if use_vision_tower_data:
        image_index_1 = metadata[metadata['image_path'].str.contains(image_name_1)].index[0]
        image_embedding_1 = embeddings[image_index_1].astype(np.float32)
    else:
        image_index_1 = metadata[metadata['image_path'].str.contains(image_name_1)].index[0]
        image_embedding_1 = embeddings[image_index_1].astype(np.float32)
    
    # Get embeddings for image 2
    if use_vision_tower_data:
        image_index_2 = metadata[metadata['image_path'].str.contains(image_name_2)].index[0]
        image_embedding_2 = embeddings[image_index_2].astype(np.float32)
    else:
        image_index_2 = metadata[metadata['image_path'].str.contains(image_name_2)].index[0]
        image_embedding_2 = embeddings[image_index_2].astype(np.float32)
    
    try:
        # Compute the aligned similarity
        similarity = aligned_similarity(image_embedding_1, image_embedding_2)
        
        # Store result
        results.append({
            'image_1': image_name_1,
            'image_2': image_name_2,
            'similarity': similarity
        })
    except ValueError as e:
        # Skip if embeddings have different numbers of vectors
        print(f"Skipping {image_name_1} and {image_name_2}: {e}")

# Create DataFrame from results
similarities_df = pd.DataFrame(results)

# Save to CSV
similarities_df.to_csv(output_csv, index=False)
print(f"Similarities saved to {output_csv}")

# Display some statistics
embedding_type = "Vision Tower" if use_vision_tower_data else "Standard"
print(f"\nImages with highest {embedding_type} similarity (most similar):")
most_similar = similarities_df.loc[similarities_df['similarity'].idxmax()]
print(f"  {most_similar['image_1']} and {most_similar['image_2']}: {most_similar['similarity']:.4f}")

print(f"\nImages with lowest {embedding_type} similarity (most different):")
most_different = similarities_df.loc[similarities_df['similarity'].idxmin()]
print(f"  {most_different['image_1']} and {most_different['image_2']}: {most_different['similarity']:.4f}")

# Optional: If you want to find the N most similar/different pairs
n = 5  # Number of pairs to show
print(f"\nTop {n} most similar image pairs ({embedding_type}):")
for _, row in similarities_df.nlargest(n, 'similarity').iterrows():
    print(f"  {row['image_1']} and {row['image_2']}: {row['similarity']:.4f}")

print(f"\nTop {n} most different image pairs ({embedding_type}):")
for _, row in similarities_df.nsmallest(n, 'similarity').iterrows():
    print(f"  {row['image_1']} and {row['image_2']}: {row['similarity']:.4f}")