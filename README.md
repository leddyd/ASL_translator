# American Sign Language Interpretor Project
Translator for American sign language using hand detection and image classification to translate signs into letters. Utilizes media pipe, tensorflow, and keras model to do model training and drawing landmarks for hand detection.

# Packagee Dependencies
```
CV2
Numpy 
Python 3.6
Tkinter
Tensorflow 2.6
Gui
Mediapipe
spellchecker
```
# Dataset
All photo data is found in customdata. If you want to generate your own image collection you can use photographer and take photos by pressing 'q' within the camera frame.

# Commands
To run the interface
```
python ./mediapipe_solution/app.py
```

To install dependencies
```
pip3 install -r requirements.txt
```

# Contributions
Dylan worked on fixing the gui interface and training the model which mainly involves
- gui.py 
- Main.ipynb

Sammy worked on the application and model building / testing  as well as data generation which involves 
- customdata
- photographer.py
- cnn_testing

Dylan and Sammy both worked on improving the user experience on translating process
- app.py