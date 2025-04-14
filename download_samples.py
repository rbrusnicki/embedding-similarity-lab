import os
import requests
from io import BytesIO
from PIL import Image

def download_sample_images():
    """Download some sample images for testing with Qwen2.5VL."""
    os.makedirs("sample_images", exist_ok=True)
    
    sample_images = {
        "cat.jpg": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?q=80&w=1000&auto=format&fit=crop",
        "landscape.jpg": "https://images.unsplash.com/photo-1506744038136-46273834b3fb?q=80&w=1000&auto=format&fit=crop",
        "food.jpg": "https://images.unsplash.com/photo-1504674900247-0877df9cc836?q=80&w=1000&auto=format&fit=crop"
    }
    
    for filename, url in sample_images.items():
        filepath = os.path.join("sample_images", filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            response = requests.get(url)
            
            # Save the image as a file
            with open(filepath, "wb") as f:
                f.write(response.content)
            
            # Verify we can load it with PIL
            try:
                Image.open(filepath).verify()
                print(f"Successfully saved to {filepath}")
            except Exception as e:
                print(f"Error verifying image {filepath}: {e}")
        else:
            print(f"{filepath} already exists")
    
    return [os.path.join("sample_images", filename) for filename in sample_images.keys()]

if __name__ == "__main__":
    print("Downloading sample images...")
    image_paths = download_sample_images()
    print(f"Downloaded {len(image_paths)} images to the sample_images directory.")
    print("You can now use these images with Qwen2.5VL for testing.")
    print("\nMake sure you have your Hugging Face token saved in a file named '.token'")
    print("\nExample usage:")
    print("  1. Basic test with URL image:")
    print("     python test_qwen2_5vl.py")
    print("  2. Test with local images:")
    print("     python test_qwen2_5vl_local.py --image sample_images/cat.jpg")
    print("\nAvailable sample images:")
    for path in image_paths:
        print(f"  - {path}") 