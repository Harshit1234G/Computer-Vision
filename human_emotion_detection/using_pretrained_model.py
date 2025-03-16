import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.special import softmax


class FacialEmotionRecognition:
    def __init__(self) -> None:
        """
        # Facial Emotion Recognition 
        It is a program to determine a person's emotion. It can detect multiple faces and their emotions, using a webcam in real-time. 

        This class uses 2 different models to detect faces and their emotions. The model are as follows:
        ## 1. res10_300x300_ssd_iter_140000

        ### Overview  
        `res10_300x300_ssd_iter_140000` is a pre-trained deep learning model used for face detection. It is based on the **Single Shot MultiBox Detector (SSD)** framework and utilizes the **ResNet-10** backbone. The model is trained on the **WIDER FACE** dataset and is commonly used for real-time face detection due to its speed and accuracy.

        ### Model Details  
        - **Architecture**: ResNet-10 + SSD  
        - **Input Size**: 300 × 300 pixels  
        - **Training Iterations**: 140,000  
        - **Framework**: Caffe  
        - **Dataset Used**: WIDER FACE  

        ### Key Features  
        - **Fast and Lightweight**: Optimized for real-time applications.  
        - **Good Accuracy**: Detects faces with decent precision, even in challenging conditions.  
        - **Pre-trained Weights**: Available in OpenCV’s Deep Neural Network (DNN) module for easy deployment.  
        - **Bounding Box Output**: Provides rectangular coordinates around detected faces with confidence scores.  

        ### Use Cases  
        - **Real-time face detection**  
        - **Face tracking in videos**  
        - **Preprocessing for face recognition systems**  
        - **Surveillance and security applications**  

        ### Limitations  
        - May struggle with **extreme occlusions** and **small faces**.  
        - Performs **less accurately** on non-frontal or low-resolution faces.  
        - Not designed for **facial landmark detection** (only detects bounding boxes).  

        This model is widely used in OpenCV for face detection tasks due to its efficiency and ease of use.

        ## 2. emotion-ferplus-12-int8.onnx  

        ### Overview  
        `emotion-ferplus-12-int8.onnx` is a deep learning model designed for **facial emotion recognition**. It is an optimized version of the **FER+ (Facial Expression Recognition Plus) model**, quantized to **INT8** precision for improved efficiency. The model classifies facial expressions into multiple emotional categories.  

        ### Model Details  
        - **Architecture**: CNN-based model (optimized for FER+)  
        - **Input Size**: 64 × 64 grayscale face images  
        - **Quantization**: INT8 (reduced model size and faster inference)  
        - **Framework**: ONNX (optimized for interoperability)  
        - **Dataset Used**: FER+ (enhanced version of the original FER2013 dataset)  

        ### Key Features  
        - **Fast and Lightweight**: INT8 quantization reduces computational overhead.  
        - **Improved Accuracy**: FER+ dataset provides more accurate emotion labels compared to FER2013.  
        - **Pre-trained Weights**: Easily deployable using OpenCV’s DNN module or ONNX runtime.  
        - **Multi-Emotion Classification**: Recognizes 8 emotion categories:  
        - Neutral  
        - Happiness  
        - Surprise  
        - Sadness  
        - Anger  
        - Disgust  
        - Fear  
        - Contempt  

        ### Use Cases  
        - **Real-time emotion recognition**  
        - **Human-computer interaction (HCI)**  
        - **Behavioral analysis and sentiment detection**  
        - **AI-powered customer experience enhancement**  

        ### Limitations  
        - **Performance drops on occluded or extreme facial angles**.  
        - **Works best with clear, well-lit facial images**.  
        - **Limited to the predefined 8 emotion categories**.  

        This model is widely used in AI applications where real-time emotion detection is required, balancing accuracy and efficiency.
        """
        self.cap = cv2.VideoCapture(0)
        
        # loading models
        self.face_rec = cv2.dnn.readNetFromCaffe(
            r'E:\Python\Computer Visions\pretrained_models\deploy.prototxt',
            r'E:\Python\Computer Visions\pretrained_models\res10_300x300_ssd_iter_140000.caffemodel'
        )
        self.ferplus = cv2.dnn.readNetFromONNX(
            r'E:\Python\Computer Visions\pretrained_models\emotion-ferplus-12-int8.onnx'
        )


    def mainloop(
            self, 
            *, 
            width: int = 640, 
            height: int = 480
        ) -> None:
        """
        It runs the program and ensures safe closer of program.

        Args:
            width (int, optional): Width of webcam frame. Defaults to 640.
            height (int, optional): Height of webcam frame. Defaults to 480.

        ## Example Usage:
        ```
        fer = FacialEmotionRecognition()
        fer.mainloop()
        ```
        """
        self.cap.set(3, width)
        self.cap.set(4, height)

        while self.cap.isOpened():
            ret, self.frame = self.cap.read()

            if not ret:
                break

            detections = self.detect_faces()

            for x1, y1, x2, y2 in detections:
                face_img = self.crop_image(x1, y1, x2, y2)
                emotions = self.detect_emotion(face_img)
                self.put_bounding_box_and_text(x1, y1, x2, y2, emotions)
                

            cv2.imshow('Facial Emotion Recognition', self.frame)

            if cv2.waitKey(1) & 0xff == ord('q'):
                self.stop()
                return


    def detect_faces(self, *, confidence_threshold: float = 0.8) -> NDArray:
        """
        This function uses `res10_300x300_ssd_iter_140000` to detect faces.

        Args:
            confidence_threshold (float, optional): The detections having more confidence than `confidence_threshold` will be considered faces. Defaults to 0.8.

        Returns:
            NDArray: A NumPy array of `dtype= int` and `shape= (N, 4)`, where N is number of detections. 4 means the coordinates of bouding box (x1, y1, x2, y2).
        """
        height, width = self.frame.shape[:2]

        # preprocessing for DNN
        blob = cv2.dnn.blobFromImage(
            self.frame, 
            scalefactor= 1.0,
            size= (300, 300),
            mean= (104.0, 177.0, 123.0)
        )

        # getting detections
        self.face_rec.setInput(blob)
        detections = self.face_rec.forward()

        # selecting faces with higher confidence
        mask = (detections[0, 0, :, 2] > confidence_threshold)
        faces = detections[0, 0, mask, 3:7]

        # scaling boxes according to self.frame
        arr = np.array([width, height, width, height])
        boxes = (faces * np.tile(arr, (faces.shape[0], 1))).astype(int)

        return boxes
    

    def detect_emotion(self, face_img: NDArray) -> str:
        """
        Detects emotions using `emotion-ferplus-12-int8.onnx`.

        Args:
            face_img (NDArray): Cropped image of just the face.

        Returns:
            str: Detected emotion, one from ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt']
        """
        # classes
        emotions = np.array(['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt'])

        # preprocessing
        face_gray = cv2.cvtColor(face_img, code= cv2.COLOR_BGR2GRAY)

        blob = cv2.dnn.blobFromImage(
            face_gray,
            scalefactor= 1.0,
            size= (64, 64)
        )

        # prediction
        self.ferplus.setInput(blob)
        scores = self.ferplus.forward().squeeze()   # squeezing to reduce dimnesion
        prob = softmax(scores)                      # to get a proper probability distribution
        top_indices = np.argsort(prob)[-3:]         # top 3 emotions
        return {emotions[i]: prob[i] for i in top_indices}


    def put_bounding_box_and_text(
            self, 
            x1: int, 
            y1: int, 
            x2: int, 
            y2: int, 
            texts: dict[str, float]
        ) -> None:
        """
        Puts bounding box and text.

        Args:
            x1 (int): X-coordinate of top-left
            y1 (int): Y-coordinate of top-left
            x2 (int): X-coordinate of bottom-right
            y2 (int): Y-coordinate of bottom-right
            texts (dict[str, float]): Top emotions to display
        """
        cv2.rectangle(
            img= self.frame,
            pt1= (x1, y1),
            pt2= (x2, y2),
            color= (0, 0, 255), 
            thickness= 2
        )

        for index, (class_, value) in enumerate(texts.items()):
            # determining the position of text
            y_pos = y1 - 10 - (index * 20)
            # if not enough space above the bounding box
            if y_pos < 10:
                y_pos = y2 + 20 + (index * 20)
                
            cv2.putText(
                img= self.frame,
                text= f'{class_}: {value:.2%}',
                org= (x1, y_pos),
                fontFace= cv2.FONT_HERSHEY_COMPLEX,
                fontScale= 0.75,
                color= (0, 0, 255),
                thickness= 2
            )
    

    def crop_image(self, x1: int, y1: int, x2: int, y2: int) -> NDArray:
        """
        Crops the image according to given coordinates

        Args:
            x1 (int): X-coordinate of top-left
            y1 (int): Y-coordinate of top-left
            x2 (int): X-coordinate of bottom-right
            y2 (int): Y-coordinate of bottom-right

        Returns:
            NDArray: Cropped Image
        """
        croped_img = self.frame[y1:y2, x1:x2]
        return croped_img
    

    def stop(self) -> None:
        """
        Releases self.cap and destorys all cv2 windows.
        """
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fer = FacialEmotionRecognition()
    fer.mainloop()
