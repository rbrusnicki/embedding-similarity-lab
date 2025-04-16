import torch
from PIL import Image
import requests
from io import BytesIO
from huggingface_hub import login
import os
import numpy as np

def load_image_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

def load_token_from_file(file_path=".token"):
    """Load Hugging Face token from a file."""
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            token = f.read().strip()
        return token
    else:
        print(f"Warning: Token file '{file_path}' not found.")
        return None

def process_vision_info(messages):
    """Process vision information from messages."""
    image_inputs = []
    video_inputs = []
    
    for message in messages:
        if message["role"] != "user":
            continue
        
        for content in message["content"]:
            if content["type"] == "image":
                if isinstance(content["image"], str) and content["image"].startswith("http"):
                    # Load image from URL
                    image = load_image_from_url(content["image"])
                    image_inputs.append(image)
                elif isinstance(content["image"], Image.Image):
                    # Use the provided PIL Image directly
                    image_inputs.append(content["image"])
                else:
                    # Assume it's a local file path
                    image = Image.open(content["image"])
                    image_inputs.append(image)
            elif content["type"] == "video":
                video_inputs.append(content["video"])
    
    return image_inputs, video_inputs

def extract_image_embeddings(model, inputs):
    """Extract the image embeddings from the model's vision encoder."""
    # Get the image embeddings from the vision tower
    vision_tower = model.get_vision_tower()
    if vision_tower is not None:
        # Move to the same device as inputs
        device = inputs["pixel_values"].device
        vision_tower = vision_tower.to(device)
        
        # Disable gradient computation for efficiency
        with torch.no_grad():
            # Get image embeddings from the vision tower
            image_embeddings = vision_tower(inputs["pixel_values"])
            
            if isinstance(image_embeddings, tuple):
                # Some models return a tuple of embeddings, take the last one
                image_embeddings = image_embeddings[0]
                
            # Print shape information
            print(f"\nImage embeddings shape: {image_embeddings.shape}")
            print(f"This means we have {image_embeddings.shape[1]} image tokens per image")
            
            # Print a sample of the embeddings
            flat_embeddings = image_embeddings.reshape(-1, image_embeddings.shape[-1])
            sample_size = min(5, flat_embeddings.shape[0])
            print(f"\nSample of {sample_size} image token embeddings (first 10 dimensions of each):")
            sample_indices = np.linspace(0, flat_embeddings.shape[0]-1, sample_size, dtype=int)
            for i, idx in enumerate(sample_indices):
                print(f"Token {i+1}: {flat_embeddings[idx, :10].cpu().numpy()}")
            
            return image_embeddings
    else:
        print("Vision tower not found in the model")
        return None

def main():
    # Model and tokenizer names
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    # Load token from file and login to Hugging Face
    token = load_token_from_file()
    if token:
        print("Authenticating with Hugging Face using token from .token file...")
        login(token=token)
    else:
        print("No token found. Authentication may fail.")
    
    print(f"Loading {model_name}...")
    
    # Configure resolution parameters for optimal performance
    # Each 28×28 pixel patch corresponds to one image token
    # We set the min and max token count to control resolution
    min_tokens = 256  # Minimum number of image tokens (patches)
    max_tokens = 1280  # Maximum number of image tokens (patches)
    min_pixels = min_tokens * 28 * 28  # Convert to pixels
    max_pixels = max_tokens * 28 * 28  # Convert to pixels
    
    print(f"Using resolution range: {min_tokens}-{max_tokens} tokens (patches of 28×28 pixels)")
    
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
        
        # Load model and processor with resolution parameters
        processor = AutoProcessor.from_pretrained(
            model_name, 
            min_pixels=min_pixels, 
            max_pixels=max_pixels,
            trust_remote_code=True
        )
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto", 
            device_map="auto",
            trust_remote_code=True
        )
        
        print("Successfully loaded Qwen2.5-VL model and processor")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have the latest transformers library installed:")
        print("pip install --upgrade transformers")
        raise
    
    # Test with a sample image
    image_url = "https://images.unsplash.com/photo-1575936123452-b67c3203c357?q=80&w=1000&auto=format&fit=crop"
    print(f"Loading image from {image_url}")
    
    # Create messages with image
    query = "What do you see in this image? Describe it in detail."
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_url},
                {"type": "text", "text": query}
            ]
        }
    ]
    
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Process vision info (images and videos)
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Process inputs
    inputs = processor(
        text=[text],
        images=image_inputs,
        #videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    
    # Print the keys in the inputs dictionary
    print("\nInputs dictionary contains the following keys:")
    for key in inputs:
        if isinstance(inputs[key], torch.Tensor):
            print(f"- {key}: tensor of shape {inputs[key].shape}")
        else:
            print(f"- {key}: {type(inputs[key])}")
    
    # Print information about image grid
    if "image_grid_thw" in inputs:
        print("\nImage grid shape (time, height, width):", inputs["image_grid_thw"].cpu().numpy())
    
    # Move inputs to GPU if available
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
        print("Using CUDA for inference")
    else:
        print("Using CPU for inference (CUDA not available)")
    
    # Extract and print the image embeddings
    print("\nExtracting image embeddings...")
    image_embeddings = extract_image_embeddings(model, inputs)
    
    # If you want to specifically see how each 28x28 patch is encoded
    if "pixel_values" in inputs:
        # Calculate how many 28x28 patches the image has been divided into
        pixel_values = inputs["pixel_values"]
        img_height, img_width = pixel_values.shape[2], pixel_values.shape[3]
        patch_size = 28
        num_patches_h = img_height // patch_size
        num_patches_w = img_width // patch_size
        total_patches = num_patches_h * num_patches_w
        
        print(f"\nThe image of size {img_height}x{img_width} is divided into {total_patches} patches")
        print(f"({num_patches_h} patches vertically × {num_patches_w} patches horizontally)")
        print(f"Each patch is {patch_size}×{patch_size} pixels and corresponds to one image token")
    
    # Generate response
    print("\nGenerating response...")
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    
    # Extract only the generated text (not the input)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    # Decode the generated tokens to text
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    print("\nQuery:", query)
    print("\nResponse:", output_text[0])

if __name__ == "__main__":
    main() 