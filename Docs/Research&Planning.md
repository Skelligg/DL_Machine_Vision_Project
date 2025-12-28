# Step 0: Research and Planning

## Dataset Choice

* **Dataset:** Fashion-MNIST ([https://www.kaggle.com/datasets/zalando-research/fashionmnist](https://www.kaggle.com/datasets/zalando-research/fashionmnist))
* **Images:** 28x28 grayscale images
* **Classes:** 10 clothing categories (T-shirt, Trouser, Pullover, Dress, etc.)
* **Train/Test Split:** 60,000 train, 10,000 test

**Justification:** Fashion-MNIST is a standard benchmark dataset for image classification. It is small enough for rapid experimentation but sufficiently complex for transfer learning with pretrained models. Its grayscale images will be converted to RGB to be compatible with pretrained vision models.

---

## Pretrained Vision Models

**Options considered:**

1. **ResNet18 / ResNet34**

   * Lightweight and widely used.
   * Good balance between speed and accuracy.
   * Compatible with Grad-CAM for explainability.
2. **VGG16 / VGG19**

   * Easy to interpret.
   * Slightly slower and more memory-intensive.
3. **EfficientNet-B0**

   * Efficient and high-performing.
   * Good for small to medium-sized datasets.

**Planned model selection:**

* **Baseline:** ResNet18 (pretrained on ImageNet).
* Optionally, EfficientNet-B0 as a comparison if time permits.

**Justification:** ResNet18 provides a strong, fast baseline for transfer learning, works well with Grad-CAM, and trains efficiently on GPU. EfficientNet-B0 may provide slightly better accuracy for comparison without requiring excessive GPU resources.

---

## Data Preprocessing and Augmentation

**Preprocessing Steps:**

1. Search for any duplicates that may cause leakage into validation set.
2. Convert grayscale 28x28 images to RGB.
3. Resize images to 224x224 for compatibility with pretrained models.
4. Normalize using ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.

**Data Augmentation (training set only):**

* It does not seem necessary for this dataset as we have a sufficient amount of images and variety.

**Justification:** These preprocessing steps ensure compatibility with pretrained networks.

---

## Model Training Strategy

1. **Baseline Pretrained Model:** Feature extraction (freeze backbone, train final classifier).
2. **Hyperparameter Tuning:**

   * Learning rates: 0.001, 0.0005, 0.0001
3. **Scratch CNN:** 3–4 convolutional layers + pooling + fully connected layers.
4. **Epochs:** 5–15 for meaningful comparisons.
5. **Logging:** TensorBoard for metrics and checkpoints.

**Justification:** This strategy allows clear comparisons between pretrained transfer learning and scratch models. Feature extraction is faster and suitable for smaller datasets, while hyperparameter tuning evaluates sensitivity to learning rate and batch size.

---

## Evaluation Metrics and Visualization

* **Metrics:** Accuracy (primary), confusion matrix, optional F1-score/precision/recall.
* **Visualization:**

  * Training/validation loss and accuracy curves
  * Grad-CAM heatmaps for at least 5 images (including 3 misclassified)
* **Justification:** These metrics provide quantitative and qualitative insight into model performance and interpretability, fulfilling project requirements for explanation and error analysis.

---

## Notebook Outline

0. Introduction & Plan (this section)
1. Data Exploration & Preprocessing
2. Baseline Pretrained Model
3. Hyperparameter Experiments
4. Scratch CNN
5. Grad-CAM & Error Analysis
6. Final Comparison Between Models
7. Conclusions & Future Work
