# Qwen2.5VL Testing

This repository contains a simple script to test the Qwen2.5VL 3B multimodal model from Alibaba.

## About Qwen2.5VL

Qwen2.5VL is a multimodal large language model that can process both text and images. It's part of the Qwen (Qwen2.5) family of models developed by Alibaba. The model can:

- Understand and describe image content in detail
- Answer questions about images
- Perform visual reasoning tasks
- Generate text based on image inputs

## Quick Start

1. Install dependencies:
```bash
python install_dependencies.py
```

2. Create a file named `.token` in the root directory with your Hugging Face token

3. Download sample images:
```bash
python download_samples.py
```

4. Run the test:
```bash
python test_qwen2_5vl_local.py --image sample_images/cat.jpg
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers (version 4.40.0 or higher)
- Pillow
- Huggingface Hub
- Accelerate
- Safetensors

Install the required packages with:

```bash
# Option 1: Use the installation script (recommended)
python install_dependencies.py

# Option 2: Install via pip
pip install torch transformers>=4.40.0 pillow huggingface_hub accelerate safetensors

# Option 3: Install via setup.py
pip install -e .

# Optional but recommended: Install Qwen-VL package
pip install git+https://github.com/QwenLM/Qwen-VL.git
```

## Authentication

Qwen2.5VL models are gated and require authentication to access. You'll need to:

1. Create a Hugging Face account if you don't have one
2. Request access to the model at [Qwen2.5VL on Hugging Face](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
3. Generate a Hugging Face token in your account settings
4. Save your token in a file named `.token` in the root directory of this project

The scripts will automatically read your token from the `.token` file for authentication.

## Important Note on Model Loading

Due to the multimodal nature of Qwen2.5VL, it requires special handling when loading through the Transformers library. The scripts provided handle this by using multiple approaches:

1. First trying to import a specific module for Qwen VL models (if you installed the Qwen-VL package)
2. Trying to use `AutoModelForVision` with `trust_remote_code=True`
3. Falling back to other methods with `trust_remote_code`

This avoids the error: `ValueError: Unrecognized configuration class for this kind of AutoModel: AutoModelForCausalLM`.

## Using the Test Scripts

### Basic URL Image Test

The repository includes a test script (`test_qwen2_5vl.py`) that demonstrates how to:

1. Load the Qwen2.5VL 3B model
2. Process an image from a URL
3. Generate a text description of the image

To run the test:

```bash
python test_qwen2_5vl.py
```

### Local Image Test

To test with local images, use the `test_qwen2_5vl_local.py` script:

```bash
python test_qwen2_5vl_local.py --image sample_images/cat.jpg
```

You can also customize the prompt:

```bash
python test_qwen2_5vl_local.py --image sample_images/cat.jpg --prompt "What kind of animal is this and what is it doing?"
```

## Sample Images

You can download sample images for testing using:

```bash
python download_samples.py
```

This will create a `sample_images` directory with a few test images.

## Troubleshooting

### Model Loading Issues

If you encounter model loading issues, try:

1. Install the Qwen-VL package directly:
```bash
pip install git+https://github.com/QwenLM/Qwen-VL.git
```

2. Make sure you're using a recent version of transformers (4.40.0 or higher):
```bash
pip install --upgrade transformers
```

3. Check that your Hugging Face token is correct and that you have access to the model

### Out of Memory Issues

If you encounter memory issues:

1. Try setting a lower precision by modifying the scripts to use `torch.float32` or `torch.bfloat16`
2. Try the 2B model version instead of the 3B version

## Hardware Requirements

For the 3B model: at least 8GB GPU VRAM (or can run on CPU but will be slow)

## Resources

- [Qwen2.5VL on Hugging Face](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [Qwen Model Family Documentation](https://qwenlm.github.io/)
- [Qwen-VL GitHub Repository](https://github.com/QwenLM/Qwen-VL) 