# MAD-YOLO: A Tiny Mosquito Detection Algorithm Based on Multi-Scale Feature Fusion

Official implementation of **MAD-YOLO**, an improved YOLOv8 for extremely small object detection in household mosquito-eradication scenarios. This repository is built on [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (v8.0.202).

Mosquito targets typically occupy **less than 1%** of the image area and appear against low-contrast, high-interference backgrounds (walls, screen windows), which causes three coupled difficulties for standard detectors: *feature annihilation*, *multi-scale degradation*, and *noise dilution*. MAD-YOLO addresses them with three synergistic modules:

| Module | Core idea | Replaces |
|--------|-----------|----------|
| **SLM** (Spatial-scale Link Module) | Large Separable Kernel Attention (LSKA) for spatial re-weighting | `SPPF` |
| **AFPN** + P2 head | Asymptotic feature fusion with an extra P2 high-resolution head | `PAN-FPN` |
| **C2SC** | Self-Calibrated Convolutions (SCConv) for signal purification | `C2f` |

On a custom dataset of 2,078 real household mosquito images, MAD-YOLO reaches **94.6 FPS** and improves Precision / Recall / mAP50 / mAP50-95 by **+10.2 / +5.5 / +6.4 / +5.2** percentage points over the YOLOv8 baseline (mAP50-95 = 33.0%).

## Model configuration

The MAD-YOLO architecture is defined in:

```
ultralytics/cfg/models/v8/yolov8-AFPN-P2345-Custom.yaml
```

Custom modules are implemented under `ultralytics/nn/extra_modules/` and registered in `ultralytics/nn/tasks.py`.

## Environment

The code is based on Ultralytics 8.0.202 and was tested with:

- Python 3.8
- PyTorch 1.13.1
- TorchVision 0.14.1

Install the extra dependencies:

```bash
pip install timm thop efficientnet_pytorch einops grad-cam
# dyhead-related modules additionally require (optional)
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

## Usage

### Train

Edit `train.py` to point to your model config and dataset, then:

```bash
python train.py
```

For MAD-YOLO, set the model to:

```python
model = YOLO('ultralytics/cfg/models/v8/yolov8-AFPN-P2345-Custom.yaml')
```

### Validate

```bash
python val.py
```

### Inference

```bash
python detect.py
```

## Dataset

Datasets are expected in Ultralytics YOLO format (an `images/` + `labels/` directory layout described by a `*.yaml`). The custom mosquito dataset used in the paper is available from the corresponding author on reasonable request.

## Results

Reported on an RTX 3080 at 640×640 input:

- Inference speed: **94.6 FPS**
- vs. YOLOv8 baseline: Precision +10.2, Recall +5.5, mAP50 +6.4, mAP50-95 +5.2 (percentage points)
- Cross-domain generalization validated on the URPC2020 underwater small-object dataset

## License

This project inherits the **AGPL-3.0** license of Ultralytics YOLOv8. You are free to use and modify it under the terms of AGPL-3.0.

## Citation

If you use MAD-YOLO in your research, please cite the corresponding paper.
