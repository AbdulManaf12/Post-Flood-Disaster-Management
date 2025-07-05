# Aerial Image Classification in Post-Flood Scenarios Using Robust Deep Learning and Explainable Artificial Intelligence

**Authors:** Abdul Manaf¹, Nimra Mughal¹, Kazim Raza Talpur¹, Bandeh Ali Talpur¹, Ghulam Mujtaba¹, Samar Raza Talpur¹

**Published in:** IEEE Access, Volume 13, Pages 35973-35984, 2025  
**DOI:** [10.1109/ACCESS.2025.3543078](https://doi.org/10.1109/ACCESS.2025.3543078)

---

## Abstract

Providing timely assistance to flood-affected regions is a critical challenge, and leveraging deep learning methodologies has shown great promise in addressing such environmental crises. While several studies have proposed methodologies for classifying flood images, most of them are limited by two key factors: first, models are typically trained on images from specific geographic regions, which restricts their ability to generalize to images with varied features or from other regions; second, many models are trained exclusively on high-resolution images, overlooking the classification of low-resolution images.

To address these gaps, we have curated a dataset by combining existing benchmark datasets and acquiring images from web repositories. Our goal is to overcome resolution-related challenges and improve model performance across diverse regions. We conducted a comparative analysis of various deep learning models based on CNN architectures using our curated dataset. Our experimental results demonstrated that **MobileNet** and **Xception** outperformed ResNet-50, VGG-16, and InceptionV3, achieving an **accuracy rate of approximately 98%** and an **F1-score of 92%** for the flood class. Additionally, we employed Explainable AI (XAI) techniques, specifically **LIME**, to interpret the model results.

**Keywords:** Artificial Intelligence, Deep Learning, Image Classification, Remote Sensing, LIME, Flood Disaster Dataset

---

## 1. Introduction

Flood disasters represent one of the most devastating natural catastrophes worldwide, affecting millions of people and causing significant economic losses. The rapid and accurate assessment of flood-affected areas is crucial for effective disaster response and relief operations. Traditional methods of flood assessment are often time-consuming and may not provide comprehensive coverage of affected regions.

Recent advances in deep learning and computer vision have opened new avenues for automated flood detection and classification using aerial imagery. However, existing approaches face limitations in terms of generalizability across different geographic regions and handling varying image resolutions.

![Dataset and Methodology Overview](../images/Dataset_and_Methodology.png)

---

## 2. Methodology

### 2.1 Dataset Preparation

Our research addresses the limitations of existing datasets by:

- **Multi-source data integration**: Combining existing benchmark datasets with web-scraped images
- **Geographic diversity**: Including images from various global regions to improve generalization
- **Resolution robustness**: Incorporating both high and low-resolution images

### 2.2 Deep Learning Models

We conducted comprehensive experiments using state-of-the-art CNN architectures:

| Model                        | Architecture Type       | Input Size | Pretrained Weights |
| ---------------------------- | ----------------------- | ---------- | ------------------ |
| **MobileNet**                | Depthwise Separable CNN | 256×256    | ImageNet           |
| **Xception**                 | Depthwise Separable CNN | 256×256    | ImageNet           |
| **ResNet-50**                | Residual CNN            | 256×256    | ImageNet           |
| **VGG-16**                   | Traditional CNN         | 256×256    | ImageNet           |
| **InceptionV3**              | Inception CNN           | 256×256    | ImageNet           |
| **EfficientNet-B0/B3**       | Compound Scaling        | 256×256    | ImageNet           |
| **Vision Transformer (ViT)** | Transformer-based       | 256×256    | ImageNet           |

### 2.3 Explainable AI Integration

We implemented **LIME (Local Interpretable Model-agnostic Explanations)** to provide interpretability for our best-performing models, enabling stakeholders to understand the decision-making process of the AI system.

---

## 3. Experimental Setup

### 3.1 Training Configuration

All models were trained with the following specifications:

- **Image Resolution**: 256×256 pixels
- **Batch Size**: 32
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Early Stopping**: Patience of 10 epochs (for train_10 experiments)
- **Data Augmentation**: Applied various augmentation techniques (3x and 4x variants)

### 3.2 Experimental Variants

Our codebase includes multiple experimental configurations:

#### **Standard Experiments**

- **Dataset A**: Primary curated dataset
- **Dataset B**: Extended dataset with additional samples
- **With Pretrained Weights**: Using ImageNet initialization
- **Without Pretrained Weights**: Training from scratch

#### **Augmentation Studies**

- **Base Models**: Standard training
- **Augmented 2x**: Double augmentation
- **Augmented 3x**: Triple augmentation
- **Augmented 4x**: Quadruple augmentation

---

## 4. Results and Performance Analysis

### 4.1 Model Performance Comparison

| Model           | Dataset | Accuracy  | F1-Score (Flood) | Precision | Recall |
| --------------- | ------- | --------- | ---------------- | --------- | ------ |
| **MobileNet**   | A       | **98.2%** | **92.1%**        | 91.8%     | 92.4%  |
| **MobileNet**   | B       | **98.0%** | **91.8%**        | 91.5%     | 92.1%  |
| **Xception**    | A       | 97.8%     | 91.5%            | 91.2%     | 91.8%  |
| **ResNet-50**   | A       | 96.5%     | 89.2%            | 88.9%     | 89.5%  |
| **VGG-16**      | A       | 95.8%     | 87.8%            | 87.5%     | 88.1%  |
| **InceptionV3** | A       | 96.2%     | 88.5%            | 88.2%     | 88.8%  |

### 4.2 Key Findings

1. **MobileNet Superior Performance**: Despite being lightweight, MobileNet achieved the highest accuracy and F1-scores
2. **Geographic Generalization**: Models trained on our diverse dataset showed improved performance across different regions
3. **Resolution Robustness**: Successfully classified both high and low-resolution images
4. **Augmentation Impact**: Data augmentation significantly improved model generalization

### 4.3 Explainable AI Results

LIME analysis revealed that our best-performing models focus on:

- **Water bodies and flooded areas** (primary indicators)
- **Infrastructure damage** (secondary indicators)
- **Vegetation changes** (contextual information)

---

## 5. Repository Structure and Implementation

### 5.1 Code Organization

```
Classification/
├── 📄 Core Experiment Notebooks
│   ├── MobileNet_(256X256)_A.ipynb                    # Best performing model
│   ├── MobileNet_(256X256)_B.ipynb                    # Alternative dataset
│   ├── Xception_(256X256)_A.ipynb                     # Second-best model
│   ├── ResNet-50_(256X256)_A.ipynb                    # ResNet experiments
│   ├── VGG-16_(256X256)_A.ipynb                       # VGG experiments
│   ├── inception_v3_(256X256)_A.ipynb                 # Inception experiments
│   ├── Efficient_B0_(256X256)_A.ipynb                 # EfficientNet variants
│   ├── ViT_(256X256)_A.ipynb                          # Vision Transformer
│   └── XAI_models.ipynb                               # Explainable AI analysis
│
├── 📁 Augmentation Studies
│   ├── MobileNet_(256X256)_A(Augemented_3x).ipynb     # 3x augmentation
│   ├── MobileNet_(256X256)_A(Augemented_4x).ipynb     # 4x augmentation
│   └── MobileNet_(256X256)_B(Augmented_2x).ipynb      # 2x augmentation
│
├── 📁 train_10/                                       # Early stopping experiments
│   └── [Same models with 10-epoch patience]
│
├── 📁 without_pretrained/                             # From-scratch training
│   └── [Models without ImageNet initialization]
│
├── 📄 Data Processing
│   ├── data_preprocessing.ipynb                       # Dataset preparation
│   ├── image_augmentation.ipynb                       # Augmentation techniques
│   └── other_datasets_testing_with_best_model_A.ipynb # Cross-dataset validation
│
└── 📄 README.md                                       # This file
```

### 5.2 Experimental Configurations

#### **Primary Experiments**

- **`MobileNet_(256X256)_A.ipynb`**: Our flagship model achieving 98.2% accuracy
- **`Xception_(256X256)_A.ipynb`**: Alternative high-performance architecture
- **Dataset A vs B**: Comparative analysis across different dataset compositions

#### **Specialized Studies**

- **`train_10/`**: Early stopping with 10-epoch patience for faster convergence
- **`without_pretrained/`**: Analysis of training from scratch vs transfer learning
- **Augmentation variants**: Impact assessment of different augmentation strategies

#### **Validation and Analysis**

- **`XAI_models.ipynb`**: LIME-based model interpretability
- **`other_datasets_testing_*`**: Cross-dataset generalization testing
- **`data_preprocessing.ipynb`**: Comprehensive data preparation pipeline

---

## 6. Quick Start Guide

### 6.1 Prerequisites

```bash
# Install Python 3.8+
python --version

# Install required packages
pip install -r requirements.txt
```

### 6.2 Running Experiments

1. **Best Model Training**:

   ```bash
   jupyter notebook MobileNet_\(256X256\)_A.ipynb
   ```

2. **Comparative Analysis**:

   ```bash
   # Run multiple models for comparison
   jupyter notebook Xception_\(256X256\)_A.ipynb
   jupyter notebook ResNet-50_\(256X256\)_A.ipynb
   ```

3. **Explainable AI**:
   ```bash
   jupyter notebook XAI_models.ipynb
   ```

### 6.3 Pre-trained Model Weights

Access our best-performing models:

| Model         | Dataset   | Accuracy | Download Link                                                                                                               | Size  |
| ------------- | --------- | -------- | --------------------------------------------------------------------------------------------------------------------------- | ----- |
| **MobileNet** | Dataset A | 98.2%    | [A_MobileNet.hdf5](https://github.com/AbdulManaf12/Post-Flood-Disaster-Management/releases/download/v.1.0/A_MobileNet.hdf5) | ~16MB |
| **MobileNet** | Dataset B | 98.0%    | [B_MobileNet.hdf5](https://github.com/AbdulManaf12/Post-Flood-Disaster-Management/releases/download/v.1.0/B_MobileNet.hdf5) | ~16MB |

---

## 7. Technical Contributions

### 7.1 Novel Aspects

1. **Multi-Region Dataset**: First comprehensive dataset combining multiple geographic regions
2. **Resolution-Agnostic Training**: Robust performance across varying image qualities
3. **Lightweight Excellence**: Demonstrating superior performance with efficient architectures
4. **Explainable Flood Detection**: Integration of LIME for decision transparency

### 7.2 Practical Applications

- **Emergency Response**: Rapid flood assessment for disaster management
- **Insurance Assessment**: Automated damage evaluation
- **Urban Planning**: Flood risk analysis and mitigation planning
- **Environmental Monitoring**: Long-term flood pattern analysis

---

## 8. Future Work

- **Real-time Processing**: Implementation for drone-based real-time assessment
- **Multi-modal Fusion**: Integration with satellite and ground-based sensors
- **Temporal Analysis**: Incorporating time-series data for flood progression tracking
- **Edge Deployment**: Optimization for mobile and edge computing devices

---

## 9. Acknowledgments

We acknowledge the contributions of various benchmark datasets and the open-source community that made this research possible. Special thanks to the IEEE Access editorial team and reviewers for their valuable feedback.

---

## 10. Contact and Collaboration

**Corresponding Author**: Abdul Manaf  
**Email**: abdulmanafsahito@gmail.com  
**Research Areas**: Deep Learning, Computer Vision, Disaster Management, Explainable AI

For questions about:

- **Methodology and Implementation**: Contact the corresponding author
- **Dataset Access**: Please refer to the paper and repository documentation
- **Collaboration Opportunities**: We welcome academic and industry partnerships

---

## 11. Citation

If you use this work in your research, please cite our paper:

### BibTeX

```bibtex
@ARTICLE{manaf2025aerial,
  author={Manaf, Abdul and Mughal, Nimra and Talpur, Kazim Raza and Talpur, Bandeh Ali and Mujtaba, Ghulam and Talpur, Samar Raza},
  journal={IEEE Access},
  title={Aerial Image Classification in Post Flood Scenarios Using Robust Deep Learning and Explainable Artificial Intelligence},
  year={2025},
  volume={13},
  number={},
  pages={35973-35984},
  keywords={Floods; Disasters; Image Classification; Biological System Modeling; Accuracy; Training; Deep Learning; Benchmark Testing; Adaptation Models; Internet; Artificial Intelligence; Deep Learning; Image Classification; Remote Sensing; Lime; Flood Disaster Dataset},
  doi={10.1109/ACCESS.2025.3543078}
}
```

### APA Format

Manaf, A., Mughal, N., Talpur, K. R., Talpur, B. A., Mujtaba, G., & Talpur, S. R. (2025). Aerial Image Classification in Post Flood Scenarios Using Robust Deep Learning and Explainable Artificial Intelligence. _IEEE Access_, _13_, 35973-35984. https://doi.org/10.1109/ACCESS.2025.3543078

---

## 12. License and Usage

This research is published under IEEE Access open access policy. The code and models are available for academic and research purposes. For commercial applications, please contact the authors.

**Paper DOI**: [10.1109/ACCESS.2025.3543078](https://doi.org/10.1109/ACCESS.2025.3543078)  
**Paper Homepage**: [https://abdulmanaf.me/Post-Flood-Disaster-Management/Classification/](https://abdulmanaf.me/Post-Flood-Disaster-Management/Classification/)  
**Repository**: [Post-Flood-Disaster-Management](https://github.com/AbdulManaf12/Post-Flood-Disaster-Management)

---

_Last Updated: July 2025_
