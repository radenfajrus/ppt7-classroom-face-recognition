
from datetime import datetime
import gc
import glob
import pandas as pd
import torch
if __name__ == '__main__':
    from backbone import build_model
else:
    from .backbone import build_model

import os
from torchvision import transforms
from PIL import Image

model = None
# dataset = [
#     {
#         "data":{"name":'Raden Alf Fajrus Shuluh',"nim":"23519031"},
#         "path":"app/face_recognition/data_retraining/23519031-Raden_Alf_Fajrus_Shuluh/0_0.jpg",
#     }, 
# ]
    
preprocess = transforms.Compose([
    transforms.Resize(112, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor()
])

dataset = []
dataset_dirs = sorted(glob.glob(os.path.join("app/face_recognition/dataset", '*')))

for dataset_dir in dataset_dirs:
    dirname = os.path.basename(dataset_dir)
    paths = sorted(glob.glob(os.path.join(dataset_dir, '*')))
    i = 0
    for path in paths:
        nim = dirname.split("-")[0]
        name = (dirname.split("-")[1]).replace("_"," ")
        dataset.append({
            "data":{"name":name,"nim":nim},
            "path":path,
        })
        i = i + 1
dataset_targets = []

def load_model_adaface(use_cuda=False):
    global model 
    global dataset_targets 
    global dataset
    now = datetime.now()
    device = "cuda" if (torch.cuda.is_available() and use_cuda) else "cpu"
    model_name = "AdaFace"

    model_path = os.path.join(os.path.dirname(__file__),'model/adaface_ir101_ms1mv3.ckpt')
    if not os.path.exists(model_path):
        os.system(
            'wget https://github.com/radenfajrus/ppt7-classroom-face-recognition/releases/download/v1/adaface_ir101_ms1mv3.ckpt -P ' + model_path
        )

    model = build_model('ir_101')
    statedict = torch.load('app/face_recognition/model/adaface_ir101_ms1mv3.ckpt',map_location=torch.device('cpu'))['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()

    list_images = []
    for data in dataset:
        img_path = data["path"]
        img = Image.open(img_path)
        img = img.resize((112,112))
        img = preprocess(img).unsqueeze(0)
        list_images.append(img)
    tensor = torch.cat(list_images, 0)
    features, _ = model(tensor)
    dataset_targets.append(features)

    _ = gc.collect()
    print("load model SR {} ({}) : {} ms".format(model_name,device,int((datetime.now() - now).total_seconds() * 1000)))


load_models = {
    "AdaFace": load_model_adaface,
    # "SFace": load_model_sface,
    # "CircleLoss": load_model_circleloss,
}

async def predict(id,list_data):
    global base_imgs
    global model

    start = datetime.now()
    data_monitor = []
    results = []
    thresh = 0.25

    for data in list_data:
        idx = data.get("idx")
        print(idx)

        img_path = data.get("img_path")
        img = Image.open(img_path)
    
        s1 = datetime.now()
        img = img.resize((112,112))
        img = preprocess(img).unsqueeze(0)
        

        preprocessing_time = int((datetime.now() - s1).total_seconds() * 1000)
        print("preprocessing_time {} : {} ms".format(idx,preprocessing_time))

        s2 = datetime.now()
        source, _ = model(img)
        
        preprocessing_time = int((datetime.now() - s2).total_seconds() * 1000)
        print("pred_time {} : {} ms".format(idx,preprocessing_time))
        s3 = datetime.now()

        similarity_scores = torch.cat([source,*dataset_targets])  @ torch.cat([source,*dataset_targets]).T
        
        scores = similarity_scores[0]
        score, idx = torch.max(scores[1:], dim=0)  # exclude source score
        score = float(score)
        print(score)

        if score >= thresh:
            res = {
                "idx" : data.get("idx"),
                "is_detected" : True,
                "score" :   score,
                "name" : dataset[int(idx)]["data"]["name"],
                "nim"  : dataset[int(idx)]["data"]["nim"],
            }
        else:
            res = {
                "idx" : data.get("idx"),
                "is_detected" : False,
                "score" :   score,
                "name" : "Data tidak ditemukan",
                "nim" : "Data tidak ditemukan",
            }

        face_detectiontime = int((datetime.now() - s3).total_seconds() * 1000)
        print("facedetectiontime {} : {} ms".format(idx,face_detectiontime))

        data_monitor.append({
            "idx" :   data.get("idx"),
            # "id" :   id,
            "p_time" :  preprocessing_time,
            "fr_time" : face_detectiontime,
            "score" :   score,
            "index" :   idx,
            "is_detected" :  res.get("is_detected"),
            "name" :  res.get("name"),
            "nim" :  res.get("nim"),
        })
        results.append(res)
        _ = gc.collect()

    print("Finish all data {} : {} ms".format(len(list_data),int((datetime.now() - start).total_seconds() * 1000)))
    df_monitor = pd.DataFrame(data_monitor)
    print(df_monitor)
    df_monitor.to_csv("logs/fr_{}.csv".format(id),index=False,header=True)


    return results

async def main():
    load_model_adaface()
    print(model)

if __name__ == "__main__":
    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())