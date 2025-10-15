# Self-Supervised Contrastive Learning for Image Classification

This repository explores the principles of **self-supervised learning** and **contrastive learning** to train a model for image classification on the CIFAR-10 dataset. The implementation uses **PyTorch** and demonstrates how to learn powerful image representations without relying on explicit labels for pre-training.

The core idea is to train a model on a "pretext" task where it learns to distinguish between similar and dissimilar images. The learned feature representations are then fine-tuned for a downstream classification task. 🖼️

---

## ✨ Core Concepts

### Self-Supervised Learning
Self-supervised learning (SSL) is a machine learning paradigm where a model learns from unlabeled data by solving automatically generated tasks (pretext tasks). For example, a model might be asked to predict a missing part of an image or identify if two augmented versions of an image come from the same source. This process forces the model to learn meaningful features about the data's underlying structure.

### Contrastive Learning
Contrastive learning is a popular SSL technique. The goal is to learn an embedding space where similar data points are brought closer together while dissimilar ones are pushed farther apart. This is achieved by:
1.  **Creating Positive Pairs**: Generating two or more augmented versions of the same image (e.g., by cropping, rotating, or changing colors).
2.  **Creating Negative Pairs**: Using augmented images from different source images.
3.  **Training**: The model is trained to maximize the similarity (e.g., cosine similarity) between positive pairs and minimize the similarity between negative pairs, often using a contrastive loss function like NT-Xent (Normalized Temperature-scaled Cross-Entropy).



---

## 📖 Dataset

This project uses the **CIFAR-10 dataset**, a widely used benchmark for image classification. It consists of 60,000 32x32 color images in 10 classes (e.g., airplane, car, bird). The data is loaded and transformed using `torchvision`, with augmentations like random cropping and color jitter applied for the contrastive learning task.

---

## 🛠️ Technologies Used

* **Python**
* **PyTorch**: The primary deep learning framework for building and training the model.
* **Torchvision**: For loading the CIFAR-10 dataset and applying image augmentations.
* **NumPy**: For numerical operations.
* **Matplotlib**: For visualizing data and results.
