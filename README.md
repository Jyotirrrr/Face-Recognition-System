Face Recognition System
This project is a simple yet effective face recognition system built in Python. It uses the face_recognition library, which is a wrapper around dlib's powerful face recognition model, to identify and manipulate faces in images and real-time video streams.

✨ Features
Face Encoding: Automatically encodes known faces from a set of labeled images.

Image Recognition: Recognizes faces in a static image file, drawing a bounding box and displaying the person's name.

Real-time Webcam Recognition: Performs live face recognition using a webcam feed.

Persistence: Saves face encodings to a file (encodings.pickle) so you don't have to re-encode faces every time you run the script.

⚙️ Prerequisites
Before running the project, you need to have Python and the necessary libraries installed.

Python 3.x

face_recognition

opencv-python

numpy

To install the required libraries, use pip:

pip install face_recognition opencv-python numpy

Note: The face_recognition library has a dependency on dlib, which can sometimes be difficult to install, especially on Windows. You may need to install CMake and a C++ compiler before installing dlib.

📁 Project Structure
To use this system, you need to set up the following directory structure:

face-recognition-project/
├── app.py                  # The main Python script
├── known_faces/            # Directory to store images of known people
│   ├── Elon_Musk/          # Subfolder for person 1
│   │   ├── photo1.jpg
│   │   └── photo2.png
│   └── Bill_Gates/         # Subfolder for person 2
│       ├── photo1.jpg
│       └── photo2.jpg
└── encodings.pickle        # (Auto-generated file) Stores face encodings

🚀 Usage
Prepare your dataset: Place images of the people you want to recognize in the known_faces directory, using a separate subfolder for each person's name.

Run the script: Navigate to the project directory in your terminal and run the main Python file.

python app.py

Follow the prompts: The script will first encode the faces and then ask you to choose between recognizing faces in an image or using a live webcam feed.

💾 Sample Dataset
A great sample dataset for this project is the "Face Recognition Dataset" available on Kaggle. It is a small, easy-to-use dataset with a manageable number of classes, making it perfect for getting started.

Link: https://www.kaggle.com/datasets/vasukipatel/face-recognition-dataset

You can download this dataset and arrange the images into the known_faces directory as described above.
