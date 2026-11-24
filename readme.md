# 🐦 AvianVision
### AI-Powered Bird Species Classification and Biodiversity Monitoring

AvianVision is a deep-learning-based bird species classification and biodiversity monitoring system. It uses transfer learning and computer vision to identify bird species from images, provide visual explanations for predictions, detect potentially unknown species, and maintain bird observation records through an interactive dashboard.

The project explores multiple deep learning architectures, including EfficientNetB0, MobileNetV2, and a custom Convolutional Neural Network (CNN), to compare their performance on bird species classification.

---

## 📌 1. Project Overview

Bird species identification plays an important role in biodiversity monitoring, ecological research, and wildlife conservation. Manual identification can be time-consuming and requires domain expertise.

AvianVision aims to simplify this process using artificial intelligence.

### Key Objectives

- Classify images into 20 selected bird species.
- Implement transfer learning using EfficientNetB0.
- Compare EfficientNetB0 with MobileNetV2 and a custom CNN.
- Provide model explainability using Grad-CAM.
- Implement unknown-species detection for images that do not confidently belong to the supported classes.
- Develop an interactive Streamlit dashboard.
- Store bird observations and classification records using SQLite.
- Provide useful biodiversity monitoring statistics.

## 🛠️ 2. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow / Keras |
| Computer Vision | OpenCV |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Deep Learning Models | EfficientNetB0, MobileNetV2, Custom CNN |
| Explainable AI | Grad-CAM |
| Frontend / Dashboard | Streamlit |
| Database | SQLite |
| Version Control | Git and GitHub |

## 📂 3. Dataset

**Dataset:** [CUB-200-2011 (Caltech-UCSD Birds-200-2011)](https://www.vision.caltech.edu/datasets/cub_200_2011/)

CUB-200-2011 is a publicly available fine-grained bird image dataset containing 200 bird categories.

For AvianVision, 20 bird species have been selected from the original dataset.

### Dataset Distribution

The selected dataset contains 1,200 images.

| Split | Images | Images per Class |
|---|---:|---:|
| Training | 480 | 24 |
| Validation | 120 | 6 |
| Testing | 600 | 30 |
| **Total** | **1,200** | **60** |

The official test split is preserved and has not been used for model training or hyperparameter tuning.

### Dataset Access

The complete dataset is not included in this GitHub repository because of its size.

To run the complete training and evaluation pipeline:

1. Download CUB-200-2011 from the official dataset source.
2. Obtain the selected 20-species dataset and preprocessing metadata from the project team.
3. Place the original images in the expected directory.
4. Run the preprocessing and training scripts.

**Expected directory structure:**

```text
AvianVision/
│
├── dataset/
│   ├── CUB_200_2011/
│   │   └── images/
│   │       ├── 001.Black_footed_Albatross/
│   │       ├── 002.Laysan_Albatross/
│   │       └── ...
│   │
│   └── processed/
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       └── class_names.json
```

The species directory names above are illustrative. The actual selected 20 classes are defined by the project metadata.

**Note:** The original dataset and its images are not committed to GitHub. Follow the original dataset's usage and redistribution terms when sharing copies.

## ⚙️ 4. Data Preprocessing

File: `src/preprocessing.py`

The preprocessing pipeline prepares the bird images for deep learning.

### Implemented Operations

1. Image loading and decoding.
2. Resizing images to 224 × 224 pixels.
3. Mapping original class IDs to zero-based labels (0–19).
4. Training-only data augmentation.
5. TensorFlow dataset creation.
6. Batching and prefetching for efficient data loading.

### Data Augmentation

The following augmentation techniques are applied to training images:

- Random horizontal flipping.
- Random rotation.
- Random zoom.
- Random contrast adjustment.

Validation and testing images are not augmented.

### Verified Preprocessing Output

| Parameter | Value |
|---|---|
| Training images | 480 |
| Validation images | 120 |
| Testing images | 600 |
| Image dimensions | 224 × 224 |
| Number of channels | 3 |
| Batch size | 16 |
| Batch shape | `(16, 224, 224, 3)` |
| Pixel range | 0–255 |
| Number of classes | 20 |

The preprocessing pipeline also generates `class_names.json`, which maintains the mapping between numerical labels and bird species names.

## 🧠 5. Deep Learning Models

AvianVision explores three architectures for bird species classification.

### 5.1 EfficientNetB0 — Primary Model

EfficientNetB0 is the primary classification architecture.

It uses ImageNet-pretrained weights and transfer learning to extract useful visual features from bird images.

**Architecture:**

```text
Input Image (224 × 224 × 3)
          |
          v
   EfficientNetB0
   (Pretrained Backbone)
          |
          v
Global Average Pooling
          |
          v
       Dropout
          |
          v
    Dense (20)
          |
          v
      Softmax
          |
          v
Predicted Bird Species
```

#### Initial Training Configuration

| Parameter | Value |
|---|---|
| Architecture | EfficientNetB0 |
| Image size | 224 × 224 |
| Batch size | 16 |
| Epochs | 20 |
| Backbone | Frozen |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | Sparse Categorical Crossentropy |

#### Initial Training Results

| Metric | Result |
|---|---:|
| Final training accuracy | 98.75% |
| Best validation accuracy | 90.00% |
| Final validation loss | 0.4523 |

#### Fine-Tuning

After initial training, the pretrained backbone was partially unfrozen to adapt its deeper feature representations to the selected bird species.

Fine-tuning configuration:

| Parameter | Value |
|---|---|
| Unfrozen backbone layers | Last 30 |
| Trainable backbone layers | 23 |
| BatchNormalization | Frozen |
| Optimizer | Adam |
| Learning rate | 0.00001 |
| Maximum epochs | 15 |
| Early stopping patience | 4 |
| Learning rate reduction | ReduceLROnPlateau |

#### Fine-Tuning Results

| Metric | Result |
|---|---:|
| Final training accuracy | 97.08% |
| Best validation accuracy | 90.00% |
| Final validation loss | 0.4564 |

Fine-tuning maintained the initial model's best validation accuracy of 90%.

The validation loss decreased from 0.5439 in the first fine-tuning epoch to 0.4564 by the final epoch.

**Current primary model:** `efficientnet_finetuned_best.keras`

### 5.2 MobileNetV2

MobileNetV2 will be implemented as a lightweight alternative to EfficientNetB0.

The objective is to compare:

- Classification accuracy.
- Precision, recall, and F1-score.
- Model size.
- Training time.
- Inference performance.

### 5.3 Custom CNN

A custom Convolutional Neural Network will be developed as a baseline model.

It will help evaluate how a CNN trained specifically on the selected dataset compares with pretrained transfer-learning architectures.

## 🔍 6. Model Explainability — Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) will be implemented to visualize which image regions influence the model's predictions.

The objective is to make the classification system more interpretable.

For example, when the model predicts a particular bird species, Grad-CAM should highlight relevant visual regions such as:

- Beak structure.
- Head and facial patterns.
- Feather coloration.
- Wing patterns.

The resulting heatmap will be overlaid on the original image to provide a visual explanation.

## ❓ 7. Unknown-Species Detection

A classification model trained on 20 species will always produce a probability distribution over those 20 classes, even when given an image of an entirely different bird.

AvianVision will implement an unknown-species detection mechanism to identify predictions that should not be treated as reliable known-species classifications.

The initial approach will investigate confidence-based thresholding.

If the prediction confidence is below a calibrated threshold, the system will flag the image as:

**Unknown / Low-Confidence Species**

The threshold will be determined using validation data rather than the test set.

This is a confidence-based rejection mechanism, not a guarantee of detecting every unseen species.

## 📊 8. Streamlit Dashboard

An interactive Streamlit dashboard will provide a user-friendly interface for the classification system.

### Planned Features

- Upload bird images.
- Display predicted bird species.
- Show prediction confidence.
- Display Grad-CAM visualizations.
- Flag unknown or low-confidence predictions.
- Maintain bird observation records.
- Display biodiversity monitoring statistics.
- Visualize species-wise observation counts.
- Display historical observation information.

## 🗄️ 9. Database — SQLite

SQLite will be used to maintain bird observation records.

### Planned Observation Fields

| Field | Description |
|---|---|
| Observation ID | Unique record identifier |
| Image Path | Stored image location |
| Predicted Species | Classified bird species |
| Confidence | Model prediction confidence |
| Observation Date | Date of observation |
| Location | Observation location, when provided |
| Classification Status | Known / Unknown |

The database will support historical record management and biodiversity monitoring analytics.

## 📁 10. Project Structure

```text
AvianVision/
│
├── dataset/
│   ├── CUB_200_2011/
│   │   └── images/                 # Not tracked by Git
│   │
│   └── processed/
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       └── class_names.json
│
├── models/
│   ├── efficientnet_best.keras
│   ├── efficientnet_final.keras
│   ├── efficientnet_history.csv
│   ├── efficientnet_finetuned_best.keras
│   └── efficientnet_finetuned_final.keras
│
├── src/
│   ├── preprocessing.py
│   ├── train_efficientnet.py
│   ├── finetune_efficientnet.py
│   ├── evaluate_efficientnet.py
│   ├── train_mobilenet.py          # Planned
│   └── train_cnn.py                # Planned
│
├── app/                            # Planned Streamlit application
│
├── README.md
├── requirements.txt
└── .gitignore
```

The structure will expand as the dashboard, database, evaluation, and explainability modules are implemented.

## 🚀 11. Installation and Setup

### Prerequisites

- Python 3.10 or later (use a TensorFlow-compatible version).
- Git.
- pip.
- Sufficient storage for the CUB-200-2011 dataset.
- CPU or compatible GPU.

### Step 1: Clone the Repository

```bash
git clone https://github.com/<YOUR_USERNAME>/AvianVision.git

cd AvianVision
```

### Step 2: Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv venv

.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download and Prepare the Dataset

Download CUB-200-2011 and arrange the selected bird images according to the dataset structure described above.

Obtain the processed CSV files and class mapping from the project team, or regenerate them using the preprocessing pipeline.

### Step 5: Run Preprocessing

```bash
python src/preprocessing.py
```

### Step 6: Train EfficientNetB0

```bash
python src/train_efficientnet.py
```

### Step 7: Fine-Tune EfficientNetB0

```bash
python src/finetune_efficientnet.py
```

Additional evaluation, comparison, and dashboard commands will be documented as those modules are completed.

## 📈 12. Evaluation Strategy

The models will be evaluated using a consistent test dataset.

The following metrics will be considered:

| Metric | Purpose |
|---|---|
| Accuracy | Overall classification correctness |
| Precision | Correctness of positive predictions |
| Recall | Ability to identify actual instances |
| F1-score | Balance between precision and recall |
| Confusion Matrix | Identifies frequently confused species |
| Model Size | Storage requirements |
| Inference Time | Prediction speed |

The official test set will remain untouched during training and model selection.

It will be used for final evaluation after model development and validation-based decisions are complete.



### Team Collaboration

All team members are expected to:

- Maintain their work in separate Git branches.
- Commit code with meaningful messages.
- Document dependencies and implementation details.
- Test their modules before merging.
- Coordinate integration through GitHub pull requests.

## 📌 14. Current Development Status

| Component | Status |
|---|---|
| Dataset download and selection | Completed |
| Train/validation/test splitting | Completed |
| Image preprocessing | Completed |
| Data augmentation | Completed |
| EfficientNetB0 initial training | Completed |
| Training and validation curves | Completed |
| EfficientNetB0 fine-tuning | Completed |
| Final test-set evaluation | Pending |
| MobileNetV2 implementation | Pending |
| Custom CNN implementation | Pending |
| Model comparison | Pending |
| Grad-CAM implementation | Pending |
| Unknown-species detection | Pending |
| Streamlit dashboard | Pending |
| SQLite integration | Pending |
| End-to-end system integration | Pending |

## 🔮 15. Future Enhancements

Potential future improvements include:

- Expanding classification to more bird species.
- Improving unknown-species detection beyond simple confidence thresholding.
- Adding geographical observation mapping.
- Supporting real-time camera-based bird identification.
- Maintaining long-term species observation statistics.
- Exploring lightweight models for mobile deployment.
- Adding location-aware biodiversity analysis.

## 📜 16. Dataset Reference

Wah, C., Branson, S., Welinder, P., Perona, P., and Belongie, S.

**The Caltech-UCSD Birds-200-2011 Dataset.**

California Institute of Technology, 2011.

Dataset: https://www.vision.caltech.edu/datasets/cub_200_2011/

## 📄 17. License and Usage

This repository contains the source code and implementation developed for AvianVision.

The CUB-200-2011 dataset is maintained by its original authors and is subject to its own terms of use. Refer to the original dataset source for applicable usage and redistribution conditions.

Third-party libraries and pretrained model weights are subject to their respective licenses.

---

**AvianVision — Bringing AI and Biodiversity Monitoring Together.** 🐦🌿
