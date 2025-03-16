import cv2
import tensorflow as tf
import numpy as np
from numpy.typing import NDArray


class FacialEmotionRecognition:
    def __init__(self) -> None:
        self.cap = cv2.VideoCapture(0)
        
        # loading models
        self.my_model = self.load_json_model(
            json_path= r'kaggle/working/vgg16_model_structure.json',
            weights_path= r'kaggle/working/vgg_16_model_data_aug.weights.h5'
        )

        self.face_rec = cv2.dnn.readNetFromCaffe(
            r'E:\Python\Computer Visions\pretrained_models\deploy.prototxt',
            r'E:\Python\Computer Visions\pretrained_models\res10_300x300_ssd_iter_140000.caffemodel'
        )


    @staticmethod
    def load_json_model(json_path: str, weights_path: str) -> tf.keras.Model:
        with open(json_path) as json_file:
            model_json = json_file.read()

        model = tf.keras.models.model_from_json(model_json)
        model.load_weights(weights_path)
        return model
    

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
    

    def detect_emotion(self, face_img: NDArray) -> dict[str, float]:
        """
        Detects emotions using `emotion-ferplus-12-int8.onnx`.

        Args:
            face_img (NDArray): Cropped image of just the face.

        Returns:
            dict[str, float]: Top 3 detected emotion, from ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        """
        # classes
        emotions = np.array(['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'])

        # preprocessing
        face_gray = cv2.cvtColor(face_img, code= cv2.COLOR_BGR2GRAY)
        face_gray = cv2.resize(face_gray, (48, 48))
        face_gray = tf.expand_dims(face_gray, axis= -1)   # adding color channel
        face_gray = tf.expand_dims(face_gray, axis= 0)    # batch dimension

        # prediction
        probas = self.my_model.predict(face_gray)[0]
        # print(probas)
        top_indices = np.argsort(probas)[-3:]         # top 3 emotions
        return {emotions[i]: probas[i] for i in top_indices}


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
