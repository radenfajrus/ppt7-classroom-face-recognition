import os, glob
import torch
from PIL import Image
from facenet_pytorch import MTCNN

map_location=torch.device('cpu')
mtcnn = MTCNN(margin=65, select_largest=True, post_process=True,device=map_location)

read_dir = "app/face_recognition/dataset_before_mtcnn"
save_dir = "app/face_recognition/dataset"

for dirname in (sorted(os.listdir(read_dir))):
    paths = sorted(glob.glob(os.path.join(read_dir+"/"+dirname, '*')))
    for idx,img_path in enumerate(paths):
        
        filename = os.path.basename(img_path)

        img = Image.open(img_path)
        img = img.convert("RGB")
        img = img.rotate(-90)
        boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
        if boxes is None:
            print(f'{idx} Error')
            continue

        for idx1,box in enumerate(boxes):
            id = f'{idx}_{idx1}.jpg'
        
            diff_h = box[2]-box[0]
            diff_w = box[3]-box[1]
            if diff_h>diff_w:
                diff_max = diff_h-diff_w 
                box[1] = box[1]-(diff_max/2)
                box[3] = box[3]+(diff_max/2)
            else:
                diff_max = diff_w-diff_h
                box[0] = box[0]-(diff_max/2)
                box[2] = box[2]+(diff_max/2)
            img_draw = img.copy()
            img_cropped = img_draw.crop(box)
            print(img_cropped.size)
            img_cropped = img_cropped.resize((112,112))
            print(img_cropped.size)
            
            print(save_dir+f"{dirname}/{filename}")
            img_cropped.save(save_dir+f"/{dirname}/{filename}")