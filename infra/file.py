import json, os
from pydantic import BaseModel


class FileClientConfig(BaseModel):
    folder: str

class FileClient():
    def __init__(self,conf: FileClientConfig):
        self.folder = conf.folder
        print("FileClient Initialized")


    def get(self, filename):
        file = "{}/{}".format(self.folder,filename)
        if os.path.exists(file) == False:
            with open(file,'w') as outfile:
                json.dump({}, outfile)
            
        data = {}
        try:
            with open(file,'r') as infile:
                content = infile.read()
                if not content:
                    self.save(data)
                else:
                    data = json.loads(content)
        except Exception as e:
            """raise Exception("Error while reading data")"""
            raise Exception("Error while reading data")
        return data

    def save(self, filename, data):
        file = "{}/{}".format(self.folder,filename)
        if os.path.exists(file) == False:
            with open(file,'w') as outfile:
                json.dump({}, outfile)
            
        try:
            with open(file,'w') as outfile:
                json.dump(data, outfile)
        except:
            """raise Exception("Error while reading data")"""
            raise Exception("Error while reading data")

        return data



### FACTORY ###
def get_client():
    cfg = FileClientConfig(
        folder = 'public/files'
    )
    return FileClientConfig(cfg)
