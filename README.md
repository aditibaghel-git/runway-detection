# ✈️ Runway Detection Using Multi-Task Deep Learning

This repository contains code for detecting runways in images using a **U-Net-style convolutional neural network** combined with anchor point prediction.

---

## ⚙️ Features

- **Image Segmentation:** Predicts runway masks from high-resolution images.
- **Anchor Point Regression:** Predicts key runway coordinates (anchors) for precise localization.
- **Custom Data Generator:** Efficient loading and augmentation of images, masks, and coordinates.
- **Evaluation Metrics:**
  - IOU (Intersection over Union) for mask accuracy
  - Anchor Score (points within predicted polygon)
  - Boolean score for centerline containment
- **Visualization:** Inspect predictions alongside ground truth masks and anchor points.

---


## 🗂 Dataset

**FS2020 Runway Dataset** by [relufrank](https://www.kaggle.com/relufrank/fs2020-runway-dataset) (approx. 10GB):

- `1920x1080/train` & `1920x1080/test`: Image directories
- `labels/lines`: JSON files with runway anchor points
- `labels/areas`: JSON files with segmentation masks

**Note:** Due to dataset size, it is recommended to run training on Kaggle notebooks or mount Google Drive in Colab.
> ⚠️ Make sure to have the following Python packages installed: tensorflow, opencv-python, kagglehub, numpy, matplotlib, pandas
---

---

## 🚀 Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set dataset paths in:

```python
config/config.py
```

3. Train the model:

```bash
python src/train.py
```

Model saved to: `outputs/models/runway_model.h5`

Visualize predictions using:

```bash
python src/visualize.py
```

---

## 📊 Evaluation

Metrics include:

* **IoU** – mask overlap
* **Anchor Coverage** – predicted polygon vs mask
* **Boolean Centerline Score** – geometric consistency

---

## 🛠 Future Improvements

* Improve overall model accuracy
* Refine predicted mask outlines
* Better segmentation losses (Dice / Focal)
* Pretrained backbone (ResNet, EfficientNet)
* Anchor ordering constraints
* Advanced data augmentation for robustness

---

## 🤝 Contributors

Contributions are welcome to **improve model accuracy and refine predicted mask outlines**, as well as bug fixes, new features, or visualization tools.

**How to contribute:** Fork → branch → commit → push → open PR. Everyone is welcome!

---

## 📄 License

This project uses the **MIT License**. 

---
