# Vision Tower Embeddings for Qwen2.5-VL

This repository contains utilities to extract and use vision tower embeddings from the Qwen2.5-VL multimodal model. These embeddings are useful for a variety of computer vision tasks, including image similarity, image retrieval, and clustering.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/qwen2.5-vl-embeddings.git
cd qwen2.5-vl-embeddings
```

2. Install the required dependencies:
```bash
pip install transformers accelerate torch torchvision pandas numpy matplotlib scikit-learn tqdm
```

## Usage

### 1. Extracting Vision Tower Embeddings

The `0_compute_emb_distances.py` script extracts vision tower embeddings from images in a given directory.

```bash
python 0_compute_emb_distances.py
```

Before running, you should update the following in the script:
- `image_dir`: Path to your images folder
- `output_dir`: Where to save the embeddings

The script includes two key functions:
- `get_vision_tower_embeddings`: Extracts raw embeddings from the vision tower
- `get_pooled_vision_embeddings`: Gets mean-pooled embeddings (more useful for similarity comparisons)

### 2. Comparing Images using Vision Tower Embeddings

The `compare_images.py` script demonstrates how to compare images using vision tower embeddings.

```bash
python compare_images.py
```

Before running, update the `image_folder` variable in the script to point to your folder of images.

The script:
1. Loads a set of images
2. Extracts vision tower embeddings for each image
3. Computes a similarity matrix between all pairs of images
4. Visualizes the similarity matrix and images
5. Finds the most similar pair of images

### 3. Image Search with Vision Tower Embeddings

The `image_search.py` script implements a simple image search system using vision tower embeddings.

```bash
python image_search.py --image_dir /path/to/image/collection --query_image /path/to/query/image.jpg --k 5
```

Arguments:
- `--image_dir`: Directory containing images to search through
- `--query_image`: Path to the query image
- `--k`: Number of results to return (default: 5)
- `--build_index`: Force rebuilding the search index
- `--output_dir`: Directory to save results (default: "search_results")

The script:
1. Builds a search index for all images in the specified directory (if not already built)
2. Extracts the vision tower embedding for the query image
3. Computes the similarity between the query and all indexed images
4. Returns and visualizes the top-k most similar images

## How It Works

### Vision Tower Architecture

The Qwen2.5-VL model contains a vision tower that processes images into embeddings. According to the model config:

```json
"vision_config": {
  "depth": 32,
  "hidden_act": "silu",
  "hidden_size": 1280,
  "intermediate_size": 3420,
  "num_heads": 16,
  "in_chans": 3,
  "out_hidden_size": 2048,
  "patch_size": 14,
  "spatial_merge_size": 2,
  "spatial_patch_size": 14,
  "window_size": 112,
  "fullatt_block_indexes": [
    7, 15, 23, 31
  ],
  "tokens_per_second": 2,
  "temporal_patch_size": 2
}
```

### Extraction Process

1. Images are processed through the Qwen2.5-VL processor to prepare them for the model
2. The processed images are passed through the model's vision tower
3. The last hidden state from the vision outputs is extracted
4. For simplicity in comparison tasks, we mean-pool these embeddings across the sequence dimension

## Applications

The vision tower embeddings can be used for:

1. **Image Similarity**: Find similar images in a collection
2. **Image Retrieval**: Search for images similar to a query image
3. **Clustering**: Group similar images together
4. **Transfer Learning**: Use these embeddings as features for downstream tasks

## Notes

- The model uses a dynamic visual resolution approach, which automatically resizes images to balance quality and computational efficiency.
- For efficient memory usage, the scripts use `torch.no_grad()` during inference.
- Embeddings are saved in NumPy format for easy storage and reuse.

## Customization

You can customize the code to:
- Use different pooling strategies (max pooling, attention pooling, etc.)
- Extract embeddings from specific layers of the vision tower
- Combine vision tower embeddings with other features
- Implement more sophisticated search algorithms 