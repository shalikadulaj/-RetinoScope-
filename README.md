
# Final Project - Deep Learning Fall 2024
### Group Name:  RetinoScope 
#### Authors :([Kavinda Rathnayaka](https://github.com/Kavi91) 2305410, [Shalika Kande Widanalage](https://github.com/shalikadulaj) 2305411, Ashika Ruwanthi)
#### University of Oulu

This repository contains the implementation of the Deep Learning 521153S-3005 course Final Project, which aims to train deep learning models for retinopathy detection using advanced techniques such as Dual Mode, Two-Stage Training, Attention (Spatial and Self), and Ensemble Learning, along with state-of-the-art Explainable AI methods like GradCAM. The project leverages datasets such as DeepDRiD and APTOS for training, evaluation, and testing. Key features include custom augmentation pipelines, self-attention mechanisms, and visualization tools.

## Table of Contents
- [Overview](#overview)
- [Project Tasks](#project-tasks)
  - Task A: Fine-tune a pretrained model using the DeepDRiD dataset
  - Task B: Two stage training with additional dataset(s)
  - Task C: Incorporate attention mechanisms in the model
  - Task D: Ensemble Learning - Compare the performance of different models and strategies
  - Task E: Creating Visualizations and Explainable AI
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Visualizations](#visualizations)
- [Contribution](#contributing)
- [License](#license)

---

## Overview
Retinopathy detection is a crucial task in medical imaging. This project demonstrates how deep learning models can achieve high accuracy and robustness in diagnosing retinopathy across multiple datasets. 

### Features
- **Custom Augmentation**: Augmentation techniques include random crop, rotation, color jitter, and self-design.
- **Transfer Learning**: Utilize pre-trained models for cross-dataset learning.
- **Attention Mechanisms**: Implement self-attention and Spatial Attention for feature enhancement.
- **Visualization**: Training loss, kappa scores, confusion matrices, and more.
- **Ensemble Learning** - Stacking, Boosting, Weighted Average, Max Voting, Bagging
- **Explainable AI for model description** - GradCAM 

---

## Project Tasks

### Task A: 
This task involves training a model on the **DeepDRiD dataset** to classify retinopathy into 5 levels. Key highlights:
- Custom augmentations for better generalization.
- Training and validation pipeline with kappa score evaluation.
- Architecture - Dual Mode

<img width="715" alt="Screenshot 2024-12-18 at 22 48 24" src="https://github.com/user-attachments/assets/3c56c02a-c3cf-42b3-b3da-0779fb142a32" />

  

### Task B: Transfer Learning
In this task, we use a pre-trained model (trained on the **APTOS-2019 dataset**) to transfer knowledge for classification on the **DeepDRiD dataset**. The approach involves fine-tuning to adapt to new domain-specific features.

- Reference Architecture
  
<img width="734" alt="Screenshot 2024-12-18 at 22 48 58" src="https://github.com/user-attachments/assets/33c60f8d-f784-4544-806a-ecf85e2ffdac" />

### Task C: Attention Mechanisms
This task introduces self-attention mechanisms in the classification model. The addition of attention layers improves feature extraction and enhances model performance.

---

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/shalikadulaj/-RetinoScope-.git
   cd -RetinoScope-

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

5. **Download the Pre-Trained Models

   [Models](https://unioulu-my.sharepoint.com/:f:/g/personal/krathnay23_student_oulu_fi/EheAw6y0KYxBupax2qY7EsoB5tTuadoT2ybtIjImDHzMyQ?e=mZpWEK )  

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

| Model Name                              | Batch Size | Learning Rate | Epochs | Best Epoch | Train Kappa | Train Accuracy | Train Precision | Train Recall | Train Loss | Val Kappa | Val Accuracy | Val Precision | Val Recall | Val Loss |
|-----------------------------------------|------------|---------------|--------|------------|-------------|----------------|-----------------|-------------|------------|-----------|--------------|---------------|------------|----------|
| ./model_dual_run_taskA.pth              | 32         | 0.0001        | 20     | 20         | 0.873660    | 0.750000       | 0.739545        | 0.750000    | 0.667789   | 0.866975  | 0.725000     | 0.716115      | 0.725000   | 0.800250 |
| ./model_dual_run_taskB2_DeepDRid.pth    | 32         | 0.0001        | 20     | 6          | 0.636656    | 0.480000       | 0.446347        | 0.480000    | 1.218990   | 0.665407  | 0.535000     | 0.583123      | 0.535000   | 1.191679 |
| ./model_dual_run_taskB_APTOS.pth        | 32         | 0.0001        | 20     | 19         | 0.857790    | 0.812969       | 0.771506        | 0.812969    | 0.527223   | 0.882813  | 0.830601     | 0.813678      | 0.830601   | 0.467242 |
| ./model_dual_run_taskC_Self_Attention.pth | 32        | 0.0001        | 20     | 14         | 0.937014    | 0.855000       | 0.857930        | 0.855000    | 0.459556   | 0.839827  | 0.695000     | 0.686422      | 0.695000   | 0.879889 |
| ./model_dual_run_taskC_Spatial_Attention.pth | 32      | 0.0001        | 20     | 7          | 0.914923    | 0.838333       | 0.836203        | 0.838333    | 0.471291   | 0.873583  | 0.700000     | 0.681686      | 0.700000   | 0.856975 |


---

## Visualizations

![W B Chart 12_18_2024, 10_03_01 PM](https://github.com/user-attachments/assets/57a2db70-d9bd-4f3e-b758-eaefeae7fd89)

![W B Chart 12_18_2024, 10_03_28 PM](https://github.com/user-attachments/assets/e15ac323-3a01-4338-8b64-e4e8076dcad9)

![W B Chart 12_18_2024, 10_03_37 PM](https://github.com/user-attachments/assets/4b61631e-0564-464b-b91b-b70dfab64e89)

![W B Chart 12_18_2024, 10_04_18 PM](https://github.com/user-attachments/assets/3b50c595-e90f-4989-9cc2-edb5ba021ade)

![W B Chart 12_18_2024, 10_05_02 PM](https://github.com/user-attachments/assets/14aacaf7-cc39-4e44-b173-6f58adbfe97d)

[For more Results](https://github.com/shalikadulaj/-RetinoScope-/tree/0a27866e80b76c1e250e832cd956dc1247dfcb93/Results)

---

## Individual Contribution

1. Task a, b , d-2, e-1 - Kavinda Rathnayaka
2. Task c, e-2 - Shalika Kande Widanalage
3. Task d1 - Ashika Ruwanthi

---

License
This project is licensed under the MIT License. See the LICENSE file for details.
