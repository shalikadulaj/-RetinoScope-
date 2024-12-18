
# Final Project - Deep Learning Fall 2024
### Group  RetinoScope ([Kavinda Rathnayaka](https://github.com/Kavi91) 2305410, Shalika Kande Vidanalage, Ashika Ruwanthi)

This repository contains the implementation of tasks A, B, and C, which aim to detect and classify retinopathy using advanced deep learning techniques. The project leverages datasets such as DeepDRiD and APTOS for training, evaluation, and testing. Key features include custom augmentation pipelines, self-attention mechanisms, and visualization tools.

## Table of Contents
- [Overview](#overview)
- [Project Tasks](#project-tasks)
  - [Task A: Classification](#task-a-classification)
  - [Task B: Transfer Learning](#task-b-transfer-learning)
  - [Task C: Attention Mechanisms](#task-c-attention-mechanisms)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Visualizations](#visualizations)
- [Contribution](#contributing)
- [License](#license)

---

## Overview
Retinopathy detection is a crucial task in medical imaging. This project demonstrates how deep learning models can achieve high accuracy and robustness in diagnosing retinopathy across multiple datasets. The implementation uses PyTorch, torchvision, and various augmentation and attention techniques.

### Features
- **Custom Augmentation**: Augmentation techniques including random crop, rotation, color jitter, and self-designed `CutOut`.
- **Transfer Learning**: Utilize pre-trained models for cross-dataset learning.
- **Attention Mechanisms**: Implement self-attention for feature enhancement.
- **Visualization**: Training loss, kappa scores, confusion matrices, and more.

---

## Project Tasks

### Task A: Classification
This task involves training a model on the **DeepDRiD dataset** to classify retinopathy into 5 levels. Key highlights:
- Custom augmentations for better generalization.
- Training and validation pipeline with kappa score evaluation.

### Task B: Transfer Learning
In this task, we use a pre-trained model (trained on the **APTOS-2019 dataset**) to transfer knowledge for classification on the **DeepDRiD dataset**. The approach involves fine-tuning to adapt to new domain-specific features.

### Task C: Attention Mechanisms
This task introduces self-attention mechanisms in the classification model. The addition of attention layers improves feature extraction and enhances model performance.

---

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/retinopathy-detection.git
   cd retinopathy-detection

2. **Set up a Virtual Environment
   ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate

3. **Install Dependencies
   ```bash
     pip install -r requirements.txt

4. **Download the Datasets

   [APTOS-2019 dataset](https://www.kaggle.com/datasets/mariaherrerot/aptos2019)

   [DeepDRiD](https://www.kaggle.com/competitions/521153S-3005-final-project/data)

---   

## Usage

1. **Google Colab

  Upload the [.ipynb](https://github.com/shalikadulaj/-RetinoScope-/tree/50079a5755f5edb0d4186eab163a8e8d0a2b45fc/Notebooks) file to the Google Colab environment and run it using a 
  hosted GPU or a local runtime with GPU availability.
   
3. **Python [.py](https://github.com/shalikadulaj/-RetinoScope-/tree/01446ec2e9111f38d0a06b73aaf948c47e12a9c2/src) file Execution
   ```bash
   python File_name.py

  
---

## Results

| Model Name                      | Batch Size | Learning Rate | Epochs | Best Epoch | Train Kappa | Train Accuracy | Train Precision | Train Recall | Train Loss | Val Kappa | Val Accuracy | Val Precision | Val Recall | Val Loss |
|---------------------------------|------------|---------------|--------|------------|-------------|----------------|-----------------|--------------|------------|-----------|--------------|---------------|------------|----------|
| model_dual_run_taskA            | 32         | 0.0001        | 20     | 15         | 0.91        | 0.85           | 0.84            | 0.86         | 0.3        | 0.88      | 0.81         | 0.82          | 0.79       | 0.2      |
| model_dual_run_taskB2_DeepDRid  | 32         | 0.0001        | 20     | 12         | 0.87        | 0.83           | 0.82            | 0.84         | 0.35       | 0.86      | 0.8          | 0.81          | 0.78       | 0.25     |
| model_dual_run_taskB_APTOS      | 32         | 0.0001        | 20     | 14         | 0.89        | 0.86           | 0.85            | 0.87         | 0.28       | 0.87      | 0.82         | 0.83          | 0.8        | 0.23     |
| model_dual_run_taskC_Self_Attention | 32      | 0.0001        | 20     | 13         | 0.92        | 0.88           | 0.87            | 0.89         | 0.26       | 0.9       | 0.85         | 0.86          | 0.84       | 0.18     |
| model_dual_run_taskC_Spatial_Attention | 32   | 0.0001        | 20     | 11         | 0.93        | 0.89           | 0.88            | 0.9          | 0.22       | 0.91      | 0.86         | 0.87          | 0.85       | 0.17     |

---

## Visualizations

![W B Chart 12_18_2024, 10_03_01 PM](https://github.com/user-attachments/assets/57a2db70-d9bd-4f3e-b758-eaefeae7fd89)

![W B Chart 12_18_2024, 10_03_28 PM](https://github.com/user-attachments/assets/e15ac323-3a01-4338-8b64-e4e8076dcad9)

![W B Chart 12_18_2024, 10_03_37 PM](https://github.com/user-attachments/assets/4b61631e-0564-464b-b91b-b70dfab64e89)

![W B Chart 12_18_2024, 10_04_18 PM](https://github.com/user-attachments/assets/3b50c595-e90f-4989-9cc2-edb5ba021ade)

![W B Chart 12_18_2024, 10_05_02 PM](https://github.com/user-attachments/assets/14aacaf7-cc39-4e44-b173-6f58adbfe97d)

[For more Results here](https://github.com/shalikadulaj/-RetinoScope-/tree/0a27866e80b76c1e250e832cd956dc1247dfcb93/Results)

---

## Individual Contribution

1. Task a b , d-2, e-1 - Kavinda Rathnayaka
2. Task c, e-2 - Shalika Kande Vidanage
3. Task d1 - Ashika Ruwanthi

---

License
This project is licensed under the MIT License. See the LICENSE file for details.
