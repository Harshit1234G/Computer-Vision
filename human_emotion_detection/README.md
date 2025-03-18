# **Human Emotion Detection**  

## **📌 About This Project**  
This is my **first computer vision project**, where I built a deep learning model to classify human emotions from facial expressions. Emotion recognition has various real-world applications, such as:  
- **Education:** Used in classrooms via CCTV to analyze students' engagement based on their emotions.  
- **Mental Health & Well-being:** Helps detect signs of stress, anxiety, or happiness.  
- **Human-Computer Interaction:** Enhances AI assistants by understanding user emotions.  
- **Security & Surveillance:** Can identify suspicious behavior based on emotional cues.  

## **📷 Dataset**
The **FER-2013** dataset is a widely used benchmark for facial emotion recognition. It consists of **35,887 grayscale images** of size **48x48 pixels**, categorized into **seven emotions**: **Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise**. The dataset was introduced in the **ICML 2013 Challenges in Representation Learning** and is often used to train and evaluate deep learning models for emotion classification.

## **🧠 Model Development**  
I explored multiple CNN architectures to improve the model’s performance:  

| **Model** | **Performance** | **Remarks** |  
|-----------|---------------|-------------|  
| **LeNet-5** | Low accuracy | Had no expectations with this architecture, as it’s too simple for this task. |  
| **AlexNet** | Performed poorly | Designed for large-scale classification, but did not generalize well here. |  
| **VGG-16** | **64% accuracy (best model)** | Achieved this with **data augmentation**, making it the most effective model so far. |  
| **ResNet-34** | Custom implementation | Built from scratch but did not surpass VGG-16 in accuracy. |  
| **Transfer Learning** | No improvement | Tried various pre-trained models, but none outperformed VGG-16. |  

🔹 I will continue working on improving this project **after my college exams**.  

## **🛠 Pretrained Models Used**  
This project also utilizes pretrained models for specific tasks:  

1. **[deploy.prototxt](https://github.com/keyurr2/face-detection/blob/master/deploy.prototxt.txt)** – Configuration file for the deep learning face detection model.  
2. **[emotion-ferplus-12-int8.onnx](https://github.com/onnx/models/blob/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-12-int8.onnx)** – ONNX model trained on the FER+ dataset for emotion recognition.  
3. **[res10_300x300_ssd_iter_140000.caffemodel](https://github.com/keyurr2/face-detection/blob/master/res10_300x300_ssd_iter_140000.caffemodel)** – A pre-trained SSD-based face detector, I used this to detect faces. 

🔹My model was able to classify all emotions correctly, whereas the pretrained model struggled and showed a bias toward the neutral emotion. While my model lacks high confidence in its predictions, it successfully distinguishes between different emotions.

## **📜 License**  
This project is licensed under the **Apache 2.0 License** – see the [LICENSE](https://github.com/Harshit1234G/Comp.Vision/blob/master/LICENSE.txt) file for details.  
