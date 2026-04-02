# Shallow Learning

**Author:** Vojin Dzeletovic  
**Date:** January 24, 2026
**Revised:** April 02, 2026 

## Setup
```bash
git clone https://github.com/triplethreadinggh/shallow_learning.git
cd shallow_learning
git checkout <commit-hash-of-the-specific-HW>
uv sync
```

## Usage

Run the binary classification script:
```bash
nohup ./binary_class.sh > training_log.out 2>&1 &
```

## HW02Q7

```bash
cd shallow_learning/scripts/
./animated_binary_class.sh

# Go here for outputs
cd /shallow_learning/scripts/media/videos/1080p30/
```

## HW02Q8

```bash
cd shallow_learning
./malwaredatadownload.sh
./scripts/multiclass_impl.sh
```

The boxplot output will be in data as .png

## Output

Loss function plot saved as PDF in `scripts/output/`

## HW03Q6 - ImageNet CNN

### Setup
```bash
uv sync
uv add onnxruntime
```

### Training
```bash
cd shallow_learning
bash scripts/imagenet_impl.sh
```
Monitor training progress:
```bash
tail -f logs/imagenet_training.log
```

### Inference with trained ONNX model
```bash
python3 scripts/imagenet_inference.py -i <path_to_image>
```
Example with sample image and redirection. This is how I ran my submitted solution:
```bash
python3 scripts/imagenet_inference.py -i scripts/output_imagenet_cnn/sample_train.png > scripts/output_imagenet_cnn/inference_result.txt
```
Top-3 predictions:
```bash
python3 scripts/imagenet_inference.py -i scripts/output_imagenet_cnn/sample_train.png -k 3
```
Custom model path:
```bash
python3 scripts/imagenet_inference.py -i <path_to_image> -m <path_to_model.onnx>
```

### Output
All outputs are saved in `scripts/output_imagenet_cnn/`:
- `imagenet_cnn.onnx` — trained model
- `training_plot.png` — loss and accuracy curves
- `sample_train.png` — sample training image
- `sample_val.png` — sample validation image

Training logs saved in `logs/imagenet_training.log`

### Inference path note
Default model path for inference is `scripts/output_imagenet_cnn/imagenet_cnn.onnx`.
To use a custom path:
```bash
python3 scripts/imagenet_inference.py -i <image> -m scripts/output_imagenet_cnn/imagenet_cnn.onnx
```

## HW03Q7 - ACC Classifier
### Setup
```bash
uv sync
```
### Training
```bash
cd shallow_learning
bash scripts/acc_impl.sh
```

### Output
All outputs are saved in `scripts/output_acc/`:
- `accnet_acc.onnx` — trained model
- `accnet_best.pt` — best checkpoint during training
- `accnet_acc.scaler.json` — scaler used for normalization
- `metrics_acc_<timestamp>.csv` — per epoch metrics (loss, accuracy, precision, recall, F1)
