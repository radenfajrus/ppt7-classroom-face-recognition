from fastapi import APIRouter, Request
router = APIRouter()
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory='templates/')

def save_to_text(content, filename):
    filepath = 'data/{}.txt'.format(filename)
    with open(filepath, 'w') as f:
        f.write(content)
    return filepath


@router.get('/camera')
def homepage(request: Request):
    result = 'Type a number'
    return templates.TemplateResponse('views/camera.html', context={'request': request, 'result': result})

from os.path import exists

l_img_path = """public/photo/{}-left.jpg""".format
f_img_path = """public/photo/{}-front.jpg""".format
r_img_path = """public/photo/{}-right.jpg""".format

@router.get('/api/images/{nim}')
def get_nim_images(request: Request, nim: int):
    result = []
    if exists(l_img_path(nim)):
        left = {
            "type":"left",
            "url": 'photo/{}-left.jpg'.format(nim)
        }
        result.append(left)
    if exists(f_img_path(nim)):
        front = {
            "type":"left",
            "url": 'photo/{}-front.jpg'.format(nim)
        }
        result.append(front)
    if exists(r_img_path(nim)):
        right = {
            "type":"left",
            "url": 'photo/{}-right.jpg'.format(nim)
        }
        result.append(right)
    return result
