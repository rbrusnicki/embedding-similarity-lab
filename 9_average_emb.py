from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import torch
from PIL import Image
import requests
from io import BytesIO
import re

# Set CUDA device 1 as the default
torch.cuda.set_device(1)

# Load the model specifically on CUDA device 1
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="cuda:1"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-3B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

# Function to process a single image and get its embedding
def get_image_embedding(image_path):
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

    # Get the image embedding
    image_embeds = inputs.pixel_values
    return image_embeds


# Directory containing the images
image_dir = "frames_covla_1k"

image_file_1 = "0c044f8d27806478_455.jpg"       # no car in this image
image_file_2 = "0cb8061c03acad29_160.jpg"       # car in this image is gray

# Process the first image
image_path_1 = os.path.join(image_dir, image_file_1)
# Get full path
absolute_path_1 = os.path.abspath(image_path_1)
# Convert to file:// format
file_uri_1 = f"file://{absolute_path_1}"
# Get embedding
embedding_1 = get_image_embedding(file_uri_1)

# Process the second image  
image_path_2 = os.path.join(image_dir, image_file_2)
# Get full path
absolute_path_2 = os.path.abspath(image_path_2)
# Convert to file:// format
file_uri_2 = f"file://{absolute_path_2}"
# Get embedding
embedding_2 = get_image_embedding(file_uri_2)

# Average the embeddings
embedding_avg = (embedding_1 + embedding_2) / 2

# Create a function that will process the input but allow us to substitute the embedding
def get_model_response_approach1(image_path, replacement_embedding, question):
    # First, get a complete set of inputs using the original image
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Replace pixel_values and ensure we're not carrying other image information
    inputs.pixel_values = replacement_embedding
    
    # Important: Check if there are other image-related fields in the inputs
    # These could include "image_embeds", "vision_embeds", or similar fields
    for key in inputs.keys():
        if "image" in key or "vision" in key or "visual" in key:
            if key != "pixel_values":
                print(f"Also replacing: {key}")
                # You might need to decide what to do with these fields
                # For now, let's try to set them to None
                setattr(inputs, key, None)
    
    inputs = inputs.to("cuda:1")
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

# Test with original image 1
print("Response for image 1:")
response1 = get_model_response_approach1(file_uri_1, embedding_1, "What color is the car in front?")
print(response1)
print("\n")

# Test with original image 2
print("Response for image 2:")
response2 = get_model_response_approach1(file_uri_2, embedding_2, "What color is the car in front?")
print(response2)
print("\n")

# Test with averaged embedding
print("Response for averaged embedding:")
response_avg = get_model_response_approach1(file_uri_1, embedding_avg, "What color is the car in front?")
print(response_avg)
print("\n")

# Test with swapped embedding (image 1 path but image 2 embedding)
print("Response for swapped embedding (image 1 path, image 2 embedding):")
response_swapped = get_model_response_approach1(file_uri_1, embedding_2, "What color is the car in front?")
print(response_swapped)

def get_model_response_approach2(replacement_embedding, question):
    # Create a dummy message without any real image
    # We'll bypass the normal image processing completely
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Create minimal inputs with just text
    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt",
    )
    
    # Manually add the pixel_values
    inputs.pixel_values = replacement_embedding
    
    # Move to device
    inputs = inputs.to("cuda:1")
    
    # Generate response
    try:
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
    except Exception as e:
        return f"Error: {str(e)}"

def load_image(image_path):
    """Helper function to load an image from a URL or file path"""
    if image_path.startswith('http'):
        # Handle URL
        response = requests.get(image_path)
        return Image.open(BytesIO(response.content))
    elif image_path.startswith('file://'):
        # Handle file:// URI by removing the prefix
        local_path = re.sub(r'^file://', '', image_path)
        return Image.open(local_path)
    else:
        # Handle regular file path
        return Image.open(image_path)

def direct_embedding_inference_fixed(embedding, question, model, processor):
    # Prepare text prompt without images
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
            ],
        }
    ]
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    text_inputs = processor.tokenizer(
        text, 
        return_tensors="pt"
    ).to("cuda:1")
    
    # The embedding should be on the device
    embedding = embedding.to("cuda:1")
    
    # We need to load a sample image properly
    try:
        # Load the image as a PIL Image
        sample_image = load_image(file_uri_1)
        
        # Get sample inputs with the properly loaded image
        sample_inputs = processor(
            text=["Sample text"],
            images=[sample_image],  # Pass the PIL Image directly
            padding=True,
            return_tensors="pt",
        )
        
        print("Available keys in sample inputs:", sample_inputs.keys())
        
        # Use the simpler approach that we already know works
        return get_model_response_approach1(file_uri_1, embedding, question)
    except Exception as e:
        print(f"Error in direct_embedding_inference_fixed: {str(e)}")
        # Fallback to approach 1
        return get_model_response_approach1(file_uri_1, embedding, question)

# Add test calls for Approach 2
print("\n")
print("=== TESTING APPROACH 2 (Bypassing image processing) ===")
print("\n")

# Test with original image 1
print("Response for image 1 (Approach 2):")
response1_app2 = get_model_response_approach2(embedding_1, "What color is the car in front?")
print(response1_app2)
print("\n")

# Test with original image 2
print("Response for image 2 (Approach 2):")
response2_app2 = get_model_response_approach2(embedding_2, "What color is the car in front?")
print(response2_app2)
print("\n")

# Test with averaged embedding
print("Response for averaged embedding (Approach 2):")
response_avg_app2 = get_model_response_approach2(embedding_avg, "What color is the car in front?")
print(response_avg_app2)
print("\n")

# Add test calls for Direct Embedding Inference
print("\n")
print("=== TESTING DIRECT EMBEDDING INFERENCE ===")
print("\n")

# Test with original image 1
print("Response for image 1 (Direct):")
response1_direct = direct_embedding_inference_fixed(embedding_1, "What color is the car in front?", model, processor)
print(response1_direct)
print("\n")

# Test with original image 2
print("Response for image 2 (Direct):")
response2_direct = direct_embedding_inference_fixed(embedding_2, "What color is the car in front?", model, processor)
print(response2_direct)
print("\n")

# Test with averaged embedding
print("Response for averaged embedding (Direct):")
response_avg_direct = direct_embedding_inference_fixed(embedding_avg, "What color is the car in front?", model, processor)
print(response_avg_direct)
print("\n")

# Summary of results
print("\n")
print("=== SUMMARY OF RESULTS ===")
print("Image 1 description: No car in this image")
print("Image 2 description: Car in this image is gray")
print("\n")
print("Approach 1 (Original method with pixel_values replacement):")
print(f"  Image 1: {response1}")
print(f"  Image 2: {response2}")
print(f"  Averaged: {response_avg}")
print(f"  Swapped (Image 1 path, Image 2 embedding): {response_swapped}")
print("\n")
print("Approach 2 (Bypassing image processing):")
print(f"  Image 1: {response1_app2}")
print(f"  Image 2: {response2_app2}")
print(f"  Averaged: {response_avg_app2}")
print("\n")
print("Direct Embedding Inference:")
print(f"  Image 1: {response1_direct}")
print(f"  Image 2: {response2_direct}")
print(f"  Averaged: {response_avg_direct}")

def get_model_response_debug(image_path, replacement_embedding, question):
    """This is a debugging version of Approach 1 to see exactly what's being processed"""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    print("Type of image_inputs:", type(image_inputs))
    if image_inputs:
        print("Number of image inputs:", len(image_inputs))
        print("Type of first image input:", type(image_inputs[0]))
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    print("Keys in processed inputs:", inputs.keys())
    
    # Print information about all image-related keys
    for key in inputs.keys():
        if hasattr(inputs, key):
            value = getattr(inputs, key)
            if isinstance(value, torch.Tensor):
                print(f"Key: {key}, Shape: {value.shape}, Type: {value.dtype}")
            else:
                print(f"Key: {key}, Type: {type(value)}")
    
    # Save the original values
    original_pixel_values = inputs.pixel_values.clone()
    
    # Replace pixel_values
    inputs.pixel_values = replacement_embedding
    
    print(f"Original pixel values shape: {original_pixel_values.shape}")
    print(f"Replacement embedding shape: {replacement_embedding.shape}")
    
    # Make sure shapes match
    if original_pixel_values.shape != replacement_embedding.shape:
        print("WARNING: Shape mismatch between original and replacement embeddings!")

    inputs = inputs.to("cuda:1")

    # Generate response
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

# Test our debug function
print("\n")
print("=== DEBUGGING APPROACH 1 (Original method with pixel_values replacement) ===")
print("\n")
debug_response = get_model_response_debug(file_uri_1, embedding_2, "What color is the car in front?")
print("Debug response:", debug_response)
print("\n")

# Update the Direct Embedding approach based on what we learn
print("\n")
print("=== TESTING FIXED DIRECT EMBEDDING INFERENCE ===")
print("\n")
print("Response for image 2 (Direct Fixed):")
response2_direct_fixed = direct_embedding_inference_fixed(embedding_2, "What color is the car in front?", model, processor)
print(response2_direct_fixed)
print("\n")

# Try a completely different approach - modifying the vision_info directly
print("\n")
print("=== TESTING VISION INFO MODIFICATION APPROACH ===")
print("\n")

def vision_info_modification_approach(source_image_path, target_embedding, question):
    # First get the vision_info from the source image
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": source_image_path,
                },
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Now modify the image input
    # We need to check what format this is in
    print("Type of image_inputs:", type(image_inputs))
    if isinstance(image_inputs, list) and len(image_inputs) > 0:
        # If it's a PIL Image or similar
        print("Image input info:", type(image_inputs[0]))
        # For now, let's continue with the regular processing and print more debug info
        
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    # Now we have the properly formatted inputs
    # Let's examine what's in them
    print("Keys in processor output:", inputs.keys())
    
    # Replace the pixel values with our target embedding
    original_pixel_values = inputs.pixel_values.clone()
    inputs.pixel_values = target_embedding
    
    # Check if shapes match
    print(f"Original shape: {original_pixel_values.shape}")
    print(f"Replacement shape: {target_embedding.shape}")
    
    # Move to device
    inputs = inputs.to("cuda:1")
    
    # Generate
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

print("Testing vision info modification:")
vision_mod_response = vision_info_modification_approach(file_uri_1, embedding_2, "What color is the car in front?")
print("Vision Info Modification Response:", vision_mod_response)

# First, let's only test the most promising approach based on our results
print("\n")
print("=== TESTING WITH THE MOST CONSISTENT APPROACH ===")
print("\n")

# Test with original image 1
print("Response for image 1 (Approach 1):")
response1_final = get_model_response_approach1(file_uri_1, embedding_1, "What color is the car in front?")
print(response1_final)
print("\n")

# Test with original image 2
print("Response for image 2 (Approach 1):")
response2_final = get_model_response_approach1(file_uri_2, embedding_2, "What color is the car in front?")
print(response2_final)
print("\n")

# Test with averaged embedding
print("Response for averaged embedding (Approach 1):")
response_avg_final = get_model_response_approach1(file_uri_1, embedding_avg, "What color is the car in front?")
print(response_avg_final)
print("\n")

# Now let's try direct modification of the embedding
print("=== TESTING EMBEDDING MODIFICATIONS ===")
print("\n")

# Test with 25% image1, 75% image2
blend_75_25 = (embedding_1 * 0.25 + embedding_2 * 0.75)
print("Response for 25% image1, 75% image2:")
response_75_25 = get_model_response_approach1(file_uri_1, blend_75_25, "What color is the car in front?")
print(response_75_25)
print("\n")

# Test with 75% image1, 25% image2
blend_25_75 = (embedding_1 * 0.75 + embedding_2 * 0.25) 
print("Response for 75% image1, 25% image2:")
response_25_75 = get_model_response_approach1(file_uri_1, blend_25_75, "What color is the car in front?")
print(response_25_75)
print("\n")

# Final summary
print("=== FINAL SUMMARY ===")
print("Image 1: No car in this image")
print("Image 2: Car in this image is gray")
print("\n")
print("Original Images:")
print(f"  Image 1 response: {response1_final}")
print(f"  Image 2 response: {response2_final}")
print("\n")
print("Blended Embeddings:")
print(f"  50/50 Average: {response_avg_final}")
print(f"  25/75 Blend: {response_75_25}")
print(f"  75/25 Blend: {response_25_75}")

