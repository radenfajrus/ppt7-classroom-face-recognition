

# from keras.models import load_model

import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
import os 

# model = load_model('public/files/facenet_keras.h5')
# model._make_predict_function()    

mtcnn = MTCNN()

def detect(frame):   
    results = []
    try:
        # results = model.predict(img,batch_size=1,verbose = 2)[0]
        boxes, probs, landmarks = mtcnn.detect(frame, landmarks=True)
        print(boxes)
        print(probs)
        print(landmarks)
    except Exception as e:
        print(e)

    return results

if __name__ == "__main__":
    print("test face detection")
