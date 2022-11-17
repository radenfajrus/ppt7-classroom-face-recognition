

# from keras.models import load_model

import cv2
import torch
import numpy as np
import os 

# model = load_model('public/files/facenet_keras.h5')
# model._make_predict_function()    

from PIL import Image, ImageDraw

from facenet_pytorch import MTCNN, InceptionResnetV1
mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()

from io import BytesIO
import base64

def get_boxes(img_base64):   
    frame = Image.open(BytesIO(base64.b64decode(img_base64)))
    results = []
    try:
        # results = model.predict(img,batch_size=1,verbose = 2)[0]
        boxes, probs, landmarks = mtcnn.detect(frame, landmarks=True)
        if boxes is not None:
            results = boxes.tolist()

    except Exception as e:
        print(e)

    return results

import glob

def crop(id, img_base64):   
    frame = Image.open(BytesIO(base64.b64decode(img_base64)))
    results = []
    buffered = BytesIO()

    try:
        # results = model.predict(img,batch_size=1,verbose = 2)[0]
        boxes, probs, landmarks = mtcnn.detect(frame, landmarks=True)

        if not os.path.isdir("public/assets/photo/{}".format(id)):
            os.makedirs("public/assets/photo/{}".format(id))

        files = glob.glob('public/assets/photo/{}/*'.format(id))
        for f in files:
            os.remove(f)

        frame.save("public/assets/photo/{}/base.jpg".format(id))
        frame_boxes = frame.copy()
        draw_boxes = ImageDraw.Draw(frame_boxes)

        for idx,box in enumerate(boxes):
            frame_draw = frame.copy()
            img_cropped = frame_draw.crop(box)

            img_cropped.save(buffered, format="JPEG")
            img = base64.b64encode(buffered.getvalue())
            img_cropped.save("public/assets/photo/{}/{}.jpg".format(id,idx))


            draw_boxes.rectangle(box.tolist(), outline=(39, 230, 89), width=3)
            
            results.append("/assets/photo/{}/{}.jpg".format(id,idx))

        frame_boxes.save("public/assets/photo/{}/base-boxes.jpg".format(id))

    except Exception as e:
        print(e)

    return results


if __name__ == "__main__":
    print("img")
    image = Image.open("public/college.jpg")
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img = base64.b64encode(buffered.getvalue())

    res = crop(img)
    print(len(res))
    # img_cropped = mtcnn(img, save_path="public/collegess.jpg")

    # print("img_embedding")
    # # Calculate embedding (unsqueeze to add batch dimension)
    # img_embedding = resnet(img_cropped.unsqueeze(0))

    # # Or, if using for VGGFace2 classification
    # print("img_probs")
    # resnet.classify = True
    # img_probs = resnet(img_cropped.unsqueeze(0))

    
    # print("detect")
    # frame = img
    # # Detect faces
    # boxes, _ = mtcnn.detect(frame)
    # print(boxes)
    
    # print("draw")
    # # Draw faces
    # frame_draw = frame.copy()
    
    # print("tracked")
    # # Add to frame list
    # frames_tracked = frame_draw.resize((640, 360), Image.BILINEAR)
    # frames_tracked.save("public/college-cropped.jpg")
    # print(frames_tracked)