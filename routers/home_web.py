from fastapi import APIRouter, Request
router = APIRouter()
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory='templates/')

def save_to_text(content, filename):
    filepath = 'data/{}.txt'.format(filename)
    with open(filepath, 'w') as f:
        f.write(content)
    return filepath


@router.get('/')
def homepage(request: Request):
    result = 'Type a number'
    return templates.TemplateResponse('views/home.html', context={'request': request, 'result': result})

@router.get('/loading')
def homepage(request: Request):
    result = 'Type a number'
    return templates.TemplateResponse('views/loading.html', context={'request': request, 'result': result})

@router.get('/error')
def homepage(request: Request):
    result = 'Type a number'
    return templates.TemplateResponse('views/error.html', context={'request': request, 'result': result})


@router.get('/pyscript')
def homepage(request: Request):
    result = 'Type a number'
    return templates.TemplateResponse('views/pyscript.html', context={'request': request, 'result': result})