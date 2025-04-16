import base64
import copy
import math
import os
import requests
import logging
from io import BytesIO
from typing import List, Dict, Tuple, Union, Optional

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# Constants
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

logger = logging.getLogger(__name__)

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )

    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    return h_bar, w_bar

def to_rgb(pil_image: Image.Image) -> Image.Image:
    """Convert image to RGB format, handling RGBA images with a white background"""
    if pil_image.mode == 'RGBA':
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
        return white_background
    else:
        return pil_image.convert("RGB")

def fetch_image(ele: Dict, size_factor: int = IMAGE_FACTOR) -> Image.Image:
    """Fetch and process an image from various sources (URL, file, base64)"""
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]

    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            # fix memory leak issue while using BytesIO
            with requests.get(image, stream=True) as response:
                response.raise_for_status()
                with BytesIO(response.content) as bio:
                    image_obj = copy.deepcopy(Image.open(bio))
        elif image.startswith("file://"):
            image_obj = Image.open(image[7:])
        elif image.startswith("data:image"):
            if "base64," in image:
                _, base64_data = image.split("base64,", 1)
                data = base64.b64decode(base64_data)
                # fix memory leak issue while using BytesIO
                with BytesIO(data) as bio:
                    image_obj = copy.deepcopy(Image.open(bio))
        else:
            image_obj = Image.open(image)
    
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    
    image = to_rgb(image_obj)

    # resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    image = image.resize((resized_width, resized_height))
    return image

def extract_vision_info(conversations: List[Dict] or List[List[Dict]]) -> List[Dict]:
    """Extract vision info (images, videos) from conversation messages"""
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    
    for conversation in conversations:
        for message in conversation:
            if isinstance(message.get("content", ""), list):
                for ele in message["content"]:
                    if isinstance(ele, dict) and (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele.get("type","") in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    
    return vision_infos

def process_vision_info(
    conversations: List[Dict] or List[List[Dict]],
    return_video_kwargs: bool = False,
) -> Tuple[List[Image.Image] or None, List[torch.Tensor or List[Image.Image]] or None, Optional[Dict]]:
    """Process vision information (images, videos) from conversations"""
    vision_infos = extract_vision_info(conversations)
    
    # Read images
    image_inputs = []
    video_inputs = []
    
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info or vision_info.get("type") == "image":
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info or vision_info.get("type") == "video":
            # For this implementation, we'll just handle videos as a placeholder
            # In a full implementation, this would call fetch_video
            logger.warning("Video processing is not fully implemented in this simplified utility")
            # If videos were in the conversation, we'd populate this with the actual video data
            # video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True)
            # video_inputs.append(video_input)
    
    if len(image_inputs) == 0:
        image_inputs = None
    
    if len(video_inputs) == 0:
        video_inputs = None
    
    if return_video_kwargs:
        # In a full implementation, this would return actual video parameters
        return image_inputs, video_inputs, {'fps': []}
    
    return image_inputs, video_inputs 