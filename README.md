# ppt7
<a name="readme-top"></a>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](#)

This project designed as a tool to support Machine Learning Model testing. Model that had been trained was placed on folder public/files. Aside from testing, this  also provide data acquisition tool on page `/camera` to collect LFR-based photo from mobile phone (for Face Recognition Model training). 

Feature (/):
* Face Detection from camera (mobile&webcam) 
* Face Detection from upload image 
* Face Recognition from each face detected
* Carousel Display processed image  

Feature (/camera):
* Capture LFR-based photo for Face Recognition training purpose
* Check the submitted photo by id (nim).


<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* FastAPI
* Facenet PyTorch
* Webcam.js
* Blueimp.js
* Jquery
* Bootstrap

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

### Installation


1. Set env variable
   ```
   pip install venv
   python -m venv .venv
   ```
2. Activate .env
   ```
   . ./.venv/Scripts/Activate
   ```
3. Deactivate .env
   ```
   #deactivate
   ```
4. Set python executor in vscode to folder .venv
   Install Extension Python
   (Ctrl+Shift+P) -> Select Interpreter -> set to .venv/Scripts/python.exe
5. Install Dependency
   ```
   pip install -r requirement.txt
   ```
6. Update Dependency
   ```
   pip freeze > requirement.txt
   ```
7. create .env 
   ```
   APP_HOST=0.0.0.0
   APP_PORT=8000
   APP_DEBUG=False
   APP_THREADED=False
   ```
8. Run App
   ```
   python main.py
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
[product-screenshot]: public/home.png
