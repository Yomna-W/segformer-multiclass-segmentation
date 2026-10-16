# AI/DL Multiclass Semantic Segmentation with Custom SegFormer

This repository contains an end-to-end deep learning pipeline for multi-class semantic segmentation and distance map estimation using a custom architecture built on top of Hugging Face's **SegFormer** (`nvidia/segformer-b5-finetuned-ade-640-640`). 

The pipeline is implemented in PyTorch and incorporates custom data cropping, run-length encoding (RLE) mask decoding, heavy data augmentations via Albumentations, and integrated evaluation metrics via Hugging Face `evaluate`.

---

## 🚀 Key Features

- **Custom SegFormer Architecture (`CustomSegFormer2`)**: Extends Hugging Face's `SegformerPreTrainedModel` to simultaneously output multi-class semantic segmentation logits and a continuous distance map.
- **Run-Length Encoding (RLE) Mask Handling**: Robust parsing utilities to reconstruct multi-class segmentation masks from competition-style RLE annotations.
- **Advanced Data Pipeline**: Automated bounding box cropping (`crop`), resizing, normalization, and tensor conversions.
- **Loss Functions & Optimization**: Supports weighted `CrossEntropyLoss` (to combat class imbalance) and experimental `FocalTverskyLoss`. Optimized using `AdamW` with step-wise learning rate decay (`StepLR`).
- **Comprehensive Monitoring**: Tracks overall accuracy, per-category accuracy, Mean Intersection over Union (mIoU), and loss using a custom `AverageMeter` logging loop.

---

## 🛠️ Project Structure & Notebook Outline

The workflow in the provided notebook is organized into the following sequential stages:

1. **Environment Setup & Dependencies**: Installation of `evaluate`, `albumentations`, `torchmetrics`, and core deep learning libraries.
2. **Utility Functions**:
   - `crop(...)`: Resizes or crops images based on mask density and region-of-interest coordinates.
   - `decode_rle_to_mask(...)`: Converts flattened RLE strings into multi-channel NumPy arrays.
3. **Model & Loss Definitions**:
   - `CustomSegFormer2`: Dual-head model returning `upsampled_logits` and `upsampled_distancemaps`.
   - `FocalTverskyLoss`: Alternative loss module for handling severe class imbalance.
   - `AverageMeter`: Helper class for tracking loss and evaluation statistics.
4. **Data Preparation & Loading**:
   - Reads training metadata from `train.csv`.
   - Iterates through unique image IDs, decodes RLE masks per class, aggregates them, and applies cropping.
5. **Data Loaders**:
   - Uses `Albumentations` (`Normalize`, `ToTensorV2`).
   - Splits data into training (`[:1600]`) and test/validation (`[1600:]`) subsets using PyTorch `DataLoader` with multi-worker support.
6. **Training Loop (`train_model`)**:
   - Manages forward/backward passes, optimizer step, scheduler updates, and dynamic logging of mIoU and accuracy via Hugging Face evaluation metrics.
   - Automatically saves the best-performing model weights to `/kaggle/working/modelPerformance/`.

---

## 📋 Requirements & Dependencies

Make sure your environment has the following packages installed:

```bash
pip install torch torchvision transformers albumentations evaluate torchmetrics scikit-learn pandas opencv-python tqdm
```

---

## ⚙️ Model Architecture Overview

```python
class CustomSegFormer2(SegformerPreTrainedModel):
    def __init__(self, config, in_channel, n_classes=5):
        super().__init__(config)
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b5-finetuned-ade-640-640", 
            ignore_mismatched_sizes=True
        )
        self.logits_outputs = nn.Conv2d(in_channel, out_channels=n_classes, kernel_size=1, stride=1, padding=0)
        self.distance_outputs = nn.Conv2d(in_channel, out_channels=1, kernel_size=1, stride=1, padding=0)
```

---

## 🚀 Getting Started & Training

To run the training script:

1. Place your dataset directory structure under `/kaggle/input/ai-dl-multiclass-segmentation/` containing `TrainImages/`, `TestImages/`, and `train.csv`.
2. Run the notebook cells sequentially to initialize the model, configure the optimizer, and start the training loop:

```python
configuration = SegformerConfig(
    hidden_dropout_prob=0.3, 
    attention_probs_dropout_prob=0.3, 
    classifier_dropout_prob=0.5
)
model = CustomSegFormer2(configuration, in_channel=150)
model.to(device)

# Train the model
trained_model = train_model(
    model=model,
    dataloaders=dataloaders,
    criterion=criterion,
    optimizer=optimizer_ft,
    scheduler=exp_lr_scheduler,
    name="segformer_b5_multiclass",
    num_epochs=25,
    metric=evaluate.load("mean_iou")
)
```

---

## 📈 Evaluation Metrics

The pipeline evaluates model predictions using Hugging Face's `mean_iou` metric, tracking:
- **Overall Accuracy**
- **Mean Accuracy**
- **Mean IoU (mIoU)**
- **Per-Category IoU & Accuracy**