
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
- [Contributing](#contributing)
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
   
3. **Python file Execution
   ```bash
   python File_name.py

  
---

## Results


---

## Visualizations


---

License
This project is licensed under the MIT License. See the LICENSE file for details.
