# 👤 Face Recognition System

This project is a simple yet effective face recognition system built in Python. It uses the `face_recognition` library, which is a wrapper around dlib's powerful face recognition model, to identify and manipulate faces in images and real-time video streams.

---

## ✨ Features

- **Face Encoding**: Automatically encodes known faces from a set of labeled images.  
- **Image Recognition**: Recognizes faces in a static image file, drawing a bounding box and displaying the person's name.  
- **Real-time Webcam Recognition**: Performs live face recognition using a webcam feed.  
- **Persistence**: Saves face encodings to a file (`encodings.pickle`) so you don't have to re-encode faces every time you run the script.

---

## ⚙️ Prerequisites

Before running the project, ensure you have Python and the necessary libraries installed:

- Python 3.x  
- `face_recognition`  
- `opencv-python`  
- `numpy`

Install the required libraries using pip:

```bash
pip install face_recognition opencv-python numpy
