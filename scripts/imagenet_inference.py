#!/usr/bin/env python3
import sys
import os
import getopt
import numpy as np
from PIL import Image
import onnxruntime as ort
from datasets import load_from_disk


def get_class_names():
    """Load class names from the ImageNet dataset on disk."""
    dataset = load_from_disk("/data/CPE_487-587/imagenet-1k-arrow")
    class_names = dataset['train'].features['label'].names
    return class_names


def preprocess_image(image_path):
    """Load and preprocess image to match val_transform used during training."""
    image = Image.open(image_path).convert('RGB')

    # Resize to 256 keeping aspect ratio then center crop to 224
    image = image.resize((256, 256), Image.BILINEAR)

    # Center crop 224x224
    left   = (256 - 224) // 2
    top    = (256 - 224) // 2
    right  = left + 224
    bottom = top  + 224
    image  = image.crop((left, top, right, bottom))

    # To numpy, normalize same as training (mean=0.5, std=0.5)
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = (img_array - 0.5) / 0.5

    # HWC -> CHW -> NCHW
    img_array = img_array.transpose(2, 0, 1)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict(model_path, image_path, top_k=5):
    """Run inference and print top-k predictions."""

    # ── Load ONNX model ───────────────────────────────────────────────────────
    print(f"Loading model from: {model_path}")
    session = ort.InferenceSession(model_path)
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # ── Preprocess image ──────────────────────────────────────────────────────
    print(f"Loading image from: {image_path}")
    img_array = preprocess_image(image_path)

    # ── Run inference ─────────────────────────────────────────────────────────
    outputs = session.run([output_name], {input_name: img_array})
    logits  = outputs[0][0]  # shape: (1000,)

    # Softmax to get probabilities
    exp_logits   = np.exp(logits - np.max(logits))
    probabilities = exp_logits / exp_logits.sum()

    # ── Load class names ──────────────────────────────────────────────────────
    print("Loading class names...")
    class_names = get_class_names()

    # ── Top-k results ─────────────────────────────────────────────────────────
    top_k_indices = np.argsort(probabilities)[::-1][:top_k]

    print(f"\n{'='*50}")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"{'='*50}")
    print(f"Top-{top_k} Predictions:\n")
    for rank, idx in enumerate(top_k_indices, 1):
        primary_name = class_names[idx].split(',')[0].strip()
        print(f"  {rank}. {primary_name:<30} {probabilities[idx]*100:.2f}%")
    print(f"{'='*50}")


def main(argv):
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_imagenet_cnn","imagenet_cnn.onnx")
    image_path = None
    top_k      = 5

    try:
        opts, args = getopt.getopt(argv, "hm:i:k:",
                                   ["help", "model=", "image=", "topk="])
    except getopt.GetoptError:
        print('Usage: imagenet_inference.py -i <image_path> [-m <model_path>] [-k <top_k>]')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('Usage: imagenet_inference.py -i <image_path> [-m <model_path>] [-k <top_k>]')
            sys.exit()
        elif opt in ('-m', '--model'):
            model_path = arg
        elif opt in ('-i', '--image'):
            image_path = arg
        elif opt in ('-k', '--topk'):
            top_k = int(arg)

    if image_path is None:
        print("Error: image path is required. Use -i <image_path>")
        print('Usage: imagenet_inference.py -i <image_path> [-m <model_path>] [-k <top_k>]')
        sys.exit(2)

    if not os.path.exists(image_path):
        print(f"Error: image file not found: {image_path}")
        sys.exit(2)

    if not os.path.exists(model_path):
        print(f"Error: model file not found: {model_path}")
        print("Make sure imagenet_cnn.onnx is in the scripts/ folder or pass -m <path>")
        sys.exit(2)

    predict(model_path, image_path, top_k=top_k)


if __name__ == "__main__":
    main(sys.argv[1:])
