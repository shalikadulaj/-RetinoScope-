
# Final Project - Deep Learning Fall 2024
### Group Name:  RetinoScope 
#### Authors :([Kavinda Rathnayaka](https://github.com/Kavi91) 2305410, [Shalika Kande Widanalage](https://github.com/shalikadulaj) 2305411, [Ashika Ruwanthi](https://github.com/ashikapamodyaruwanthi) 2302767)
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
- [Installation and Codes](#installation-and-codes)
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

<div align="center">
  <img width="715" alt="Screenshot 2024-12-18 at 22 48 24" src="https://github.com/user-attachments/assets/3c56c02a-c3cf-42b3-b3da-0779fb142a32" />
</div>
  

### Task B: Transfer Learning
In this task, we use a pre-trained model (trained on the **APTOS-2019 dataset**) to transfer knowledge for classification on the **DeepDRiD dataset**. The approach involves fine-tuning to adapt to new domain-specific features.

- Reference Architecture

<div align="center">  
<img width="734" alt="Screenshot 2024-12-18 at 22 48 58" src="https://github.com/user-attachments/assets/33c60f8d-f784-4544-806a-ecf85e2ffdac" />
</div>

### Task C: Attention Mechanisms
This task introduces self-attention mechanisms in the classification model. The addition of attention layers improves feature extraction and enhances model performance.


### Task D: Ensemble Learning 

In this task, Different ensemble learning techniques were experimented over base models. The validation data of **DeepDRiD dataset** set was used to obtain predictions from base models (ResNet18, ResNet34 and VGG16) because the training data set has already been used for training the base models. Results indicates that the performance was enhanced with ensemble learning techniques.
Further, several pre-processing techniques were examined over enseble learning techniques. The performance was further enhanced with pre-processing techniques. 

### Task E: Explainable AI 


In this task, we use Explainable AI techniques such as GradCAM (Gradient-weighted Class Activation Mapping) to highlight the important regions in an input image that a convolutional neural network focuses on when making a predictions.















---

## Installation and Codes

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

5. Download the codes

    - [Task A](https://github.com/shalikadulaj/-RetinoScope-/blob/f588a6fbc60c4f01cc57de8a8c8727e5861712c7/Notebooks/DL_FP_Task_A.ipynb) 
    - Task B
      
       - [APTOS-2019-resnet18](https://github.com/shalikadulaj/-RetinoScope-/blob/01e0bd47d27f3ee8f096bec3c777d602eca8e472/Notebooks/DL_FP_Task_B_APTOS-RESNET18.ipynb)
       - [APTOS-2019-resnet34](https://github.com/shalikadulaj/-RetinoScope-/blob/01e0bd47d27f3ee8f096bec3c777d602eca8e472/Notebooks/DL_FP_Task_B_APTOS-RESNET34.ipynb)
       - [APTOS-2019-VGG16](https://github.com/shalikadulaj/-RetinoScope-/blob/01e0bd47d27f3ee8f096bec3c777d602eca8e472/Notebooks/DL_FP_Task_B_APTOS-VGG16.ipynb)
         
       - [Two-Stage-DeepDRid-resnet18](https://github.com/shalikadulaj/-RetinoScope-/blob/01e0bd47d27f3ee8f096bec3c777d602eca8e472/Notebooks/DL_FP_Task_B_II_DeepDRid-RESNET18.ipynb)
       - [Two-Stage-DeepDRid-resnet34](https://github.com/shalikadulaj/-RetinoScope-/blob/01e0bd47d27f3ee8f096bec3c777d602eca8e472/Notebooks/DL_FP_Task_B_II_DeepDRid-RESNET34.ipynb)
       - [Two-Stage-DeepDRid-VGG16](https://github.com/shalikadulaj/-RetinoScope-/blob/01e0bd47d27f3ee8f096bec3c777d602eca8e472/Notebooks/DL_FP_Task_B_II_DeepDRid-VGG16.ipynb)
         
    - Task C
      - [Self-attention](https://github.com/shalikadulaj/-RetinoScope-/blob/b0313c0334d639cb0cfc23e06fc18701f8e9f49d/src/dl_fp_task_c_with_self_attention.py)
      - [Spatial-attention](https://github.com/shalikadulaj/-RetinoScope-/blob/8248de7ee00ee51a8db491eee650f5fd28046ac9/src/dl_fp_task_c_with_spatial_attention.py) 
    - Task D
      -   [Ensemble Learning Techniques](https://github.com/shalikadulaj/-RetinoScope-/blob/main/Notebooks/DL_FP_Task_D_Ensemble_Learning.ipynb)
      -   [Ensemble Learning with Pre-processing Methods](https://github.com/shalikadulaj/-RetinoScope-/blob/main/Notebooks/DL_FP_Task_D_Ensemble_Learning_With_Pre_Processing.ipynb)
      
    - Task E - Explainable AI
      -   [Single Model](https://github.com/shalikadulaj/-RetinoScope-/blob/8248de7ee00ee51a8db491eee650f5fd28046ac9/src/task_E_GradCAM_for_Single_Model.py)
      -   [Dual Model](https://github.com/shalikadulaj/-RetinoScope-/blob/7841136543d06e9d795ab05e1efe406819e56da8/src/task_E_GradCAM_for_Dual_Model.py)


 
6. **Download the Pre-Trained Models

   - [Task A Models](https://unioulu-my.sharepoint.com/:f:/g/personal/krathnay23_student_oulu_fi/EhMNbtO60-ZDkwTHsVNGwFUBiWDKcvfNgUTqHX6t0d8fuw?e=GIeyXS)
   - [Task B-APTOS Models](https://unioulu-my.sharepoint.com/:f:/g/personal/krathnay23_student_oulu_fi/Esh4AlfsBD9Hi2hJMaMx33ABOda2D6WULGekl60lhsjwGw?e=deGEch)
   - [Task B-DeepDRid Models](https://unioulu-my.sharepoint.com/:f:/g/personal/krathnay23_student_oulu_fi/ErxfjYqYDSRKqX5L8rG7hEoBt_CWjySc-aHWfTyTCk3Llg?e=fqXJBv)
   - [Task C Models](https://unioulu-my.sharepoint.com/:f:/g/personal/krathnay23_student_oulu_fi/EiUSmN9jtyNOuKtv2w9ni4sBJHjVjFoBJw5ZUlTewNXm1A?e=Qm8XLC)
   - [Task D Models](https://unioulu-my.sharepoint.com/:f:/r/personal/krathnay23_student_oulu_fi/Documents/Deep%20Learning%202024%20-%20Final%20Project%20Models%20-%20Group%20%23Retinoscope/Task%20D%20-%20Modals?csf=1&web=1&e=aIJz05)          

---   

## Usage

> **Note:**  
> Kindly ensure that the dataset paths, model paths (for GRADCM),  and test image paths(for GRADCM) are correctly set in the main file before running the code.


1. **Google Colab

  Upload the [.ipynb](https://github.com/shalikadulaj/-RetinoScope-/tree/50079a5755f5edb0d4186eab163a8e8d0a2b45fc/Notebooks) file to the Google Colab environment and run it using a 
  hosted GPU or a local runtime with GPU availability.
   
2. **Python [.py](https://github.com/shalikadulaj/-RetinoScope-/tree/01446ec2e9111f38d0a06b73aaf948c47e12a9c2/src) file Execution
   ```bash
   python File_name.py


---

## Results

# Model Performance Metrics

| Model Name                                              | Batch Size | Learning Rate | Epochs | Best Epoch | Train Kappa | Train Accuracy | Train Precision | Train Recall | Train Loss | Val Kappa | Val Accuracy | Val Precision | Val Recall | Val Loss |
|---------------------------------------------------------|------------|---------------|--------|------------|-------------|----------------|-----------------|--------------|------------|-----------|--------------|---------------|------------|----------|
| `./model_dual_run_taskA.pth`                            | 32         | 0.0001        | 20     | 20         | 0.873660    | 0.750000       | 0.739545        | 0.750000     | 0.667789   | 0.866975  | 0.725        | 0.716115      | 0.725      | 0.800250 |
| `./model_dual_run_Task B-resnet18 - Two Stage Training` | 32         | 0.0001        | 20     | 6          | 0.859581    | 0.706667       | 0.706436        | 0.706667     | 0.753298   | 0.831091  | 0.675        | 0.604035      | 0.675      | 0.883531 |
| `./model_dual_run_Task B-resnet34 - Two Stage Training` | 32         | 0.0001        | 20     | 5          | 0.861138    | 0.710000       | 0.685231        | 0.710000     | 0.717953   | 0.832273  | 0.680        | 0.619312      | 0.680      | 0.899415 |
| `./model_dual_run_Task B-vgg16 - Two Stage Training`    | 32         | 0.0001        | 20     | 20         | 0.800237    | 0.631667       | 0.591996        | 0.631667     | 0.931427   | 0.820438  | 0.675        | 0.606760      | 0.675      | 0.933223 |
| `./model_dual_run_taskC_Self_Attention.pth`             | 32         | 0.0001        | 20     | 14         | 0.937014    | 0.855000       | 0.857930        | 0.855000     | 0.459556   | 0.839827  | 0.695        | 0.686422      | 0.695      | 0.879889 |
| `./model_dual_run_taskC_Spatial_Attention.pth`          | 32         | 0.0001        | 20     | 12         | 0.928560    | 0.845000       | 0.850215        | 0.845000     | 0.482145   | 0.835215  | 0.690        | 0.680320      | 0.690      | 0.884215 |





---

## Visualizations
![W B Chart 12_23_2024, 8_25_47 PM](https://github.com/user-attachments/assets/84cba572-ee0d-45e4-8641-9312a85249da)

![W B Chart 12_23_2024, 8_25_24 PM](https://github.com/user-attachments/assets/483f38ac-c858-49f9-b8c1-47891b3d8304)

![W B Chart 12_23_2024, 8_25_08 PM](https://github.com/user-attachments/assets/6849a36f-f48d-4631-9885-212f795f9a3a)

![W B Chart 12_23_2024, 8_24_48 PM](https://github.com/user-attachments/assets/112ecdf3-e0f2-4bd8-a1e3-5a87cd8266c7)

[For more Results](https://github.com/shalikadulaj/-RetinoScope-/tree/0a27866e80b76c1e250e832cd956dc1247dfcb93/Results)

#### Results of GradCAM for Single Model

![Results of GradCAM for Single Model](https://github.com/shalikadulaj/-RetinoScope-/blob/adefe0b8231cd233fdb216e9c06f193ae800e96e/Results/task_E_GradCAM_Single_Model.png)

#### Results of GradCAM for Dual Model

![Results of GradCAM for Dual Model](https://github.com/shalikadulaj/-RetinoScope-/blob/edc5f8d2f9b5876ebc5122c2a6bb139f2ddc70ea/Results/task_E_GradCAM_Dual_Model.png)
---

## Individual Contribution

1. Task a, b , d-2, e-1 - Kavinda Rathnayaka
2. Task c, e-2 - Shalika Kande Widanalage
3. Task d1 - Ashika Ruwanthi

---

License
This project is licensed under the MIT License. See the LICENSE file for details.
