# Handwritten Digit Recognition Using CNN

## Project Overview

Handwritten Digit Recognition Using CNN is a Deep Learning project developed to recognize handwritten numerical digits (0–9) using a Convolutional Neural Network (CNN).

The project uses the MNIST dataset, which contains 70,000 grayscale images of handwritten digits. A CNN model is developed using TensorFlow and Keras to automatically learn image features and classify handwritten digits.

The model achieved **99.04% test accuracy** after 5 epochs of training.

## Project Information

| Details | Description |
|---|---|
| Project Name | Handwritten Digit Recognition with CNN |
| Project ID | DL-001 |
| Program | Deep Learning |
| Group ID | BT_DL_01 |
| Course | B.Tech – Computer Science and Engineering |
| Semester | V |
| Section | A |
| Mentor | Mr. Yashraj |
| Institution | Invertis University, Bareilly |

## Objectives

- To understand the working of Convolutional Neural Networks.
- To develop a deep learning model for handwritten digit recognition.
- To explore and preprocess the MNIST dataset.
- To train and evaluate a CNN model using TensorFlow and Keras.
- To achieve high accuracy in digit classification.
- To understand the practical applications of deep learning in computer vision.

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Google Colab
- MNIST Dataset
- Streamlit

## Dataset Description

The project uses the MNIST handwritten digit dataset.

| Dataset Details | Values |
|---|---|
| Total Images | 70,000 |
| Training Images | 60,000 |
| Testing Images | 10,000 |
| Image Dimensions | 28 × 28 |
| Image Format | Grayscale |
| Classes | 0–9 |

## Model Architecture

The Convolutional Neural Network consists of:

1. Input Layer
2. Convolutional Layer (32 Filters)
3. Max Pooling Layer
4. Convolutional Layer (64 Filters)
5. Max Pooling Layer
6. Flatten Layer
7. Dense Layer (64 Neurons)
8. Dropout Layer (0.3)
9. Output Layer (10 Neurons with Softmax)

## Methodology

The project follows these steps:

1. Loading the MNIST dataset.
2. Visualizing sample handwritten digit images.
3. Preprocessing and normalizing image data.
4. Designing the CNN architecture.
5. Compiling the model using Adam optimizer.
6. Training the model for 5 epochs.
7. Evaluating the model on test data.
8. Predicting handwritten digits with confidence scores.
9. Saving the trained model for future use.

## Training Configuration

| Parameter | Value |
|---|---|
| Framework | TensorFlow and Keras |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Epochs | 5 |
| Batch Size | 64 |
| Validation Split | 10% |
| Dropout | 0.3 |
| Total Parameters | 121,930 |

## Results

The trained CNN model achieved the following performance:

| Metric | Result |
|---|---:|
| Training Accuracy | 98.66% |
| Validation Accuracy | 99.03% |
| Test Accuracy | **99.04%** |
| Test Loss | 0.0299 |

The results demonstrate that the CNN model can recognize handwritten digits with high accuracy.

## Project Workflow

Dataset Loading → Data Preprocessing → CNN Model Building → Model Training → Model Evaluation → Digit Prediction → Model Saving

## Future Scope

- Real-time handwritten digit recognition.
- Integration with a web-based application.
- Mobile application development.
- Recognition of handwritten letters and numbers.
- Improving model performance using data augmentation.
- Extending the system to recognize multiple handwritten digits.

## Conclusion

This project successfully demonstrates the application of Convolutional Neural Networks in handwritten digit recognition. By using the MNIST dataset and TensorFlow/Keras, the model achieved 99.04% test accuracy.

The project provides practical knowledge of deep learning, image preprocessing, CNN architecture, model training, and image classification.

---

**Academic Deep Learning Project | B.Tech CSE | Semester V | 2026**
