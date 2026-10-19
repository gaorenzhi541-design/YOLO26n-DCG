# YOLO26n-DCG: A Lightweight Dorsal Hand Acupoint Detection Model Tailored for Edge Devices

Official PyTorch implementation and pre-trained weights for the paper **"A lightweight dorsal hand acupoint detection model YOLO26n-DCG tailored for edge devices"**.

## 1. Overview
**YOLO26n-DCG** is a lightweight keypoint detection architecture designed for automatic localization of 15 dorsal hand acupoints in traditional Chinese medicine (TCM) edge-based auxiliary devices:
- **D-LKA (`ultralytics_modules/block.py`)**: Deformable Large Kernel Attention integrated into the backbone to capture non-rigid hand geometric deformations.
- **CIB / C2fCIB (`ultralytics_modules/block.py`)**: Compact Inverted Block in the neck network for lossless cross-channel feature aggregation with minimal parameter overhead.
- **KBC Loss (`ultralytics_modules/loss.py`)**: Keypoint Bounding-box Constraint loss function (`PoseLoss26`) to suppress out-of-bounds coordinate drift.

## 2. Repository Structure
- `configs/yolo26n-dcg.yaml`: Network architecture configuration for 15 dorsal hand acupoints (`kpt_shape: [15, 3]`).
- `ultralytics_modules/__init__.py` & `block.py`: Implementation of `D_LKA`, `CIB`, and `C2fCIB` modules.
- `ultralytics_modules/loss.py`: Implementation of the `KBC` loss constraint in `PoseLoss26`.
- `ultralytics_modules/tasks.py`: Model parser and task registration for `YOLO26n-DCG`.
- `weights/yolo26n-dcg.pt` & `weights/yolo26n-dcg.onnx`: Trained PyTorch and ONNX weights for desktop and NVIDIA Jetson Orin Nano edge inference.

## 3. Usage
1. Install dependencies (`Python >= 3.9`, `PyTorch >= 2.0`, `ultralytics`, `torchvision`).
2. Replace the corresponding files in your local `ultralytics` package (`ultralytics/nn/modules/__init__.py`, `ultralytics/nn/modules/block.py`, `ultralytics/nn/tasks.py`, and `ultralytics/utils/loss.py`) with the files in `ultralytics_modules/`.
3. Run inference or training:
```python
from ultralytics import YOLO

# Load pre-trained YOLO26n-DCG weights for inference
model = YOLO("weights/yolo26n-dcg.pt")
results = model.predict(source="your_hand_image.jpg", imgsz=256)
