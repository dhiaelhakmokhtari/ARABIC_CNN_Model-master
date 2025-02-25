# Arabic Handwritten Letter Recognition 📝🔠🤖

## Project Overview 🚀📈🔍

This project focuses on **handwritten Arabic letter recognition** using **Convolutional Neural Networks (CNNs)**. It aims to develop models that can accurately classify Arabic letters based on a dataset of handwritten samples. The research was inspired by methodologies applied to Bangla handwritten character recognition and adapted to Arabic script.

## Dataset 📂📊🔎

- **23,327 images** of handwritten Arabic letters.
- **62 classes**, representing different Arabic letters.
- Images were **resized (128x128 to 32x32)** for computational efficiency.
- Data split into **training (70%), validation (15%), and test (15%) sets**.

## Methodology 🏗️🧪📊

### Data Preprocessing ✂️🖼️📌
- **Grayscale conversion** to simplify image processing.
- **Binarization and inversion** for enhanced contrast.
- **Normalization and resizing** to optimize model input.

### CNN Model Architectures 🤖🛠️📡

#### Model 1:
- **Convolutional layers** with ReLU activation.
- **Max-pooling layers** to reduce spatial dimensions.
- **Fully connected layers** for feature integration.
- **Softmax activation** for classification.
- **170,580 trainable parameters**.
- Achieved **90.23% accuracy** on test data.

#### Model 2:
- **Deeper architecture** with more convolutional layers.
- **Increased feature complexity handling**.
- **Softmax activation** for multi-class classification.
- **8,157,438 trainable parameters**.
- Achieved **92.58% accuracy** on test data.

## Training Setup ⚙️📉📈

- **Framework:** TensorFlow
- **Optimizer:** Adam
- **Loss Function:** Categorical Cross-Entropy
- **Metrics:** Accuracy
- **Batch size:** 32
- **Model 1:** Trained for **50 epochs**
- **Model 2:** Trained for **25 epochs** (best performance at 20 epochs)

## Challenges & Limitations ⚠️🛑🤔

- **Data Imbalance**: Some letter classes had more samples than others.
- **Computational Constraints**: Model 2 required higher resources.
- **Handwriting Variability**: Different styles and noise in images affected classification.

## Future Improvements 🚀🔬💡

- **Data Augmentation**: Enhance dataset diversity with transformations.
- **More Balanced Dataset**: Increase sample size for underrepresented letters.
- **Hyperparameter Tuning**: Optimize learning rate and architectures.
- **GUI Development**: Implement a real-time handwriting recognition interface.
