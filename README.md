# CISRNet: Multi-modal Complementary Integration and Spectral Refinement for Optical-SAR Flood Detection

This repository contains the PyTorch implementation of **CISRNet**, a dual-stream encoder-single decoder architecture for optical-SAR fusion flood detection.

## Architecture Overview

CISRNet adopts a dual-stream encoder and single-stream decoder architecture, designed to effectively compress multi-modal information into a unified representation for robust flood detection via a hierarchical feature fusion strategy. Specifically, we utilize two parallel ResNet-50 backbones to extract multi-scale features from Optical and SAR images respectively, deploying a Residual Dilated Block (RS_Dblock) at the deepest encoder stage to capture long-range context. To bridge the modality gap, we design a hierarchical fusion mechanism at each encoder stage: the extracted dual-stream features first interact via the **Feature Complementary Module (FCM)** and are immediately merged into a single enhanced feature tensor by the **Refined Feature Integration (RFI)** module. The decoder progressively restores spatial resolution, and the final output is structurally optimized in the frequency domain via the **Spectral Refinement Block (SRB)**.

### Key Modules

- **FCM (Feature Complementary Module)**: Mitigates the inherent distributional shifts between Optical and SAR modalities through dual-domain (channel + spatial) cross-modal recalibration. Uses a zero-initialized residual scaling factor to ensure identity mapping at training onset.

- **RFI (Refined Feature Integration)**: Synthesizes complementary information from refined optical and SAR streams via a dual-branch strategy: (1) a spatially-variant gating mechanism for pixel-level fusion, and (2) a content-adaptive dynamic branch that calibrates fusion weights based on global scene representation.

- **SRB (Spectral Refinement Block)**: Counteracts the spectral bias of deep fusion architectures by leveraging the orthogonal decomposition property of the Discrete Wavelet Transform (DWT). Disentangles features into frequency-specific components for targeted restoration of high-frequency boundary details via learnable Frequency MLP.

## Requirements

- Python >= 3.8
- PyTorch >= 1.10
- segmentation-models-pytorch
- albumentations
- scikit-learn
- numpy
- matplotlib
- opencv-python
- Pillow
- tqdm
- pandas
- thop

Install dependencies:

```bash
pip install torch torchvision segmentation-models-pytorch albumentations scikit-learn numpy matplotlib opencv-python Pillow tqdm pandas thop
```

## Dataset Preparation

We evaluate on two benchmarks:

1. **CAU-Flood**: A large-scale multi-modal dataset with 15,231 training and 3,071 testing samples.
2. **Wuhan**: A small-sample dataset with 552 training and 112 testing samples.

The dataset should be organized with matched optical, SAR, and label images sharing the same filenames:

```
data/
  train/
    opt/          # Optical images (4-channel, e.g., B-G-R-NIR)
    vv/           # SAR images (1-channel, VV polarization)
    flood_vv/     # Binary flood labels (0: background, 1: flood)
  test/
    opt/
    vv/
    flood_vv/
```

Before training, update the dataset paths in `config.py`:

```python
OPTICAL_DIR = "/path/to/train/opt"
RADAR_DIR = "/path/to/train/vv"
LABEL_DIR = "/path/to/train/flood_vv"
```

And the test set paths in `train.py` and `test.py`:

```python
TEST_OPT_DIR = "/path/to/test/opt"
TEST_SAR_DIR = "/path/to/test/vv"
TEST_LBL_DIR = "/path/to/test/flood_vv"
```

## Training

```bash
python train.py
```

Training configuration can be modified in `config.py`:

| Parameter | Default | Description |
|:---|:---|:---|
| `EPOCHS` | 50 | Number of training epochs |
| `BATCH_SIZE` | 16 | Training batch size |
| `LR` | 0.01 | Initial learning rate |
| `height, width` | 256, 256 | Input image size |
| `ratio` | 0.5 | Binary prediction threshold |

## Testing / Evaluation

```bash
python test.py
```

Before running, update `CHECKPOINT_PATH` in `test.py` to point to your trained model weights:

```python
CHECKPOINT_PATH = "./CISRNet/checkpoints/resnet101_imagenet_CISRNet_best.pt"
```

The script outputs:
- Evaluation metrics (mIoU, F1, Accuracy, Precision, Recall, Kappa) printed to console
- Metrics saved to `test_results_standalone.csv`
- Sample visualization saved to `test_result_sample.jpg`

## Evaluation Metrics

We adopt five standard metrics for evaluation:

| Metric | Description |
|:---|:---|
| mIoU | Mean Intersection over Union (flood + background) |
| Accuracy | Overall pixel-level accuracy |
| Precision | Positive predictive value for flood class |
| Recall | Sensitivity / true positive rate for flood class |
| F1 | Harmonic mean of Precision and Recall |
| Kappa | Cohen's Kappa coefficient |

## 