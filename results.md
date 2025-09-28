# Gesture Recognition Lab Summary

This document summarizes the results of the gesture recognition lab, where we explored three different deep learning models to classify hand gestures from images.

## Model Performance Comparison

Here is a summary of the final validation accuracy for each model:

| Model | Validation Accuracy | Key Observations |
| :--- | :--- | :--- |
| **1. Base CNN** | 34.97% | Suffered from severe **overfitting**. The model memorized the training data but failed to generalize to the validation set. |
| **2. Augmented CNN** | 33.10% | **Successfully reduced overfitting** by using data augmentation, but the simple architecture was not powerful enough to learn the complex patterns from the augmented data, resulting in slightly lower accuracy. |
| **3. Improved CNN** | 39.63% | Adding more layers and `Dropout` **improved both accuracy and generalization**. This showed that a more complex custom model could perform better but still had limitations. |
| **4. Transfer Learning (MobileNetV2)**| **56.88%** | **The most effective approach by a significant margin.** Leveraging a pre-trained model provided powerful features that, after fine-tuning, adapted very well to our specific task and achieved the highest accuracy. |

## Conclusion

This lab work clearly demonstrates the power of transfer learning for computer vision tasks. While building a custom model from scratch is an excellent learning exercise, using a pre-trained model like `MobileNetV2` is often the most effective strategy to achieve high performance, especially when working with a relatively small dataset. The pre-trained model provides a strong foundation of learned features that can be quickly adapted to a new, specific task.