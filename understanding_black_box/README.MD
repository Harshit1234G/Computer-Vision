# 🧠 CNN Feature Map Visualizer – Facial Emotion Recognition (FER2013)

A project that visualizes the **intermediate feature maps** of a VGG-16 Convolutional Neural Network trained on the **FER2013** dataset. It helps explore **how CNNs interpret and transform facial images** into features across different convolutional layers.

## 🔍 Project Overview

This tool extracts and visualizes the **first 16 feature maps** from every convolutional layer of a given CNN. In this implementation, the model used is **VGG-16 trained on FER2013**, a dataset for recognizing **7 basic human emotions**:

* 😠 Anger
* 🤢 Disgust
* 😨 Fear
* 😄 Happiness
* 😐 Neutral
* 😭 Sadness
* 😮 Surprise

The purpose of this project is to:

* Understand **how deep layers interpret images**
* Track **progressive abstraction** of features
* Visually explore **how emotions are captured** by CNNs

## 📷 Example Output

> Visualization of 16 feature maps from each convolutional layer, here `C` means Convolutional layer and `F` means number of feature map:
![angry](https://github.com/user-attachments/assets/8c3b99d4-5030-4db5-ae13-9589646f41bf)

## ✍️ How It Works

1. **Identifies all convolutional layers** in the original CNN (e.g., VGG-16).
2. **Constructs a new intermediate model** that shares the same input as the original CNN, but outputs the feature maps from selected convolutional layers.
3. **Feeds a test image** through this intermediate model to extract the feature maps corresponding to each convolutional layer.
4. **Visualizes the first 16 feature maps** from each layer using Matplotlib, showing how the image is progressively transformed by the network.

## 🎓 Learning Outcomes

* Deepen your understanding of **convolutional filters**
* Visual intuition into what CNN layers “see”
* Great educational tool for exploring **black-box behavior**

## 🧠 Future Ideas

* Compare **different architectures** (ResNet, MobileNet)
* Visualize attention maps or Grad-CAM

## **📜 License**  
This project is licensed under the **Apache 2.0 License** – see the [LICENSE](https://github.com/Harshit1234G/Comp.Vision/blob/master/LICENSE.txt) file for details.  