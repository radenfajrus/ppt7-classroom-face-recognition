# ppt7

## Set env variable
pip install venv
python -m venv .venv


## Activate .env
. ./.venv/Scripts/Activate

## Deactivate .env
deactivate

## Set python executor in vscode to folder .venv
Install Extension Python
(Ctrl+Shift+P) -> Select Interpreter -> set to .venv/Scripts/python.exe


## Install Dependency
pip install -r requirement.txt

## Update Dependency
pip freeze > requirement.txt

## create .env if needed
