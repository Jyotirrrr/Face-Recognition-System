# -*- coding: utf-8 -*-
import os
import pickle
import time

import cv2
import face_recognition
import numpy as np

# --- Configuration ---
KNOWN_FACES_DIR = "known_faces" # Directory containing subfolders of known people's faces
ENCODINGS_FILE = "encodings.pickle" # File to save/load face encodings
TOLERANCE = 0.6 # How much distance between faces to consider a match. Lower is stricter.
FRAME_THICKNESS = 2 # Thickness of the bounding box around detected faces
FONT_THICKNESS = 2 # Thickness of the text for names
MODEL = "cnn" # or "hog" for face detection. "cnn" is more accurate but slower, "hog" is faster.

# --- Helper Functions ---

def load_encodings(filename):
    """Loads known face encodings and names from a pickle file."""
    if os.path.exists(filename):
        print(f"Loading encodings from {filename}...")
        with open(filename, "rb") as f:
            return pickle.load(f)
    return {"encodings": [], "names": []}

def save_encodings(encodings_data, filename):
    """Saves known face encodings and names to a pickle file."""
    print(f"Saving encodings to {filename}...")
    with open(filename, "wb") as f:
        pickle.dump(encodings_data, f)
    print("Encodings saved.")

def encode_known_faces(known_faces_dir, encodings_file):
    """
    Encodes faces from the known_faces_dir and saves them to encodings_file.
    Each subfolder in known_faces_dir should represent a person,
    and contain images of that person.
    """
    known_face_encodings = []
    known_face_names = []

    # Load existing encodings to avoid re-processing if they exist
    existing_encodings_data = load_encodings(encodings_file)
    known_face_encodings = existing_encodings_data["encodings"]
    known_face_names = existing_encodings_data["names"]

    processed_folders = set(known_face_names) # Keep track of already processed people

    print(f"Processing faces from '{known_faces_dir}'...")
    for name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, name)
        if not os.path.isdir(person_dir):
            continue

        if name in processed_folders:
            print(f"Skipping '{name}' (already processed).")
            continue

        print(f"Processing images for '{name}'...")
        for filename in os.listdir(person_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(person_dir, filename)
                print(f"  Loading {image_path}...")
                image = face_recognition.load_image_file(image_path)
                face_locations = face_recognition.face_locations(image, model=MODEL)

                if face_locations:
                    # Assume one face per image for known faces for simplicity
                    # If multiple faces are present, it will encode the first one found
                    encoding = face_recognition.face_encodings(image, face_locations)[0]
                    known_face_encodings.append(encoding)
                    known_face_names.append(name)
                    print(f"  Encoded face for {name} from {filename}")
                else:
                    print(f"  No face found in {filename} for {name}")
        if name not in processed_folders: # Add to processed only if new images were added
             processed_folders.add(name)

    # Save the updated encodings
    save_encodings({"encodings": known_face_encodings, "names": known_face_names}, encodings_file)
    print("Face encoding complete.")

def recognize_faces_in_image(image_path, known_encodings_data, tolerance=TOLERANCE, model=MODEL):
    """
    Recognizes faces in a given image file.
    Draws bounding boxes and names on the image and displays it.
    """
    known_face_encodings = known_encodings_data["encodings"]
    known_face_names = known_encodings_data["names"]

    print(f"Loading image for recognition: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_image = image[:, :, ::-1]

    print("Detecting faces...")
    face_locations = face_recognition.face_locations(rgb_image, model=model)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    print(f"Found {len(face_locations)} face(s).")

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), FRAME_THICKNESS)

        # Draw a label with a name below the face
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), FONT_THICKNESS)

    cv2.imshow("Face Recognition", image)
    cv2.waitKey(0) # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows()

def recognize_faces_webcam(known_encodings_data, tolerance=TOLERANCE, model=MODEL):
    """
    Performs real-time face recognition using a webcam.
    """
    known_face_encodings = known_encodings_data["encodings"]
    known_face_names = known_encodings_data["names"]

    video_capture = cv2.VideoCapture(0) # 0 for default webcam

    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting webcam recognition. Press 'q' to quit.")

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame, model=model)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance)
                name = "Unknown"

                # # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), FRAME_THICKNESS)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), FONT_THICKNESS)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # --- Step 1: Prepare your dataset ---
    # Create a directory named 'known_faces' in the same location as this script.
    # Inside 'known_faces', create subfolders for each person you want to recognize.
    # Name each subfolder after the person (e.g., 'known_faces/Elon Musk', 'known_faces/Bill Gates').
    # Place multiple clear images of each person in their respective subfolder.
    # Example structure:
    # .
    # ├── app.py
    # ├── known_faces/
    # │   ├── Person_A/
    # │   │   ├── person_a_1.jpg
    # │   │   ├── person_a_2.png
    # │   └── Person_B/
    # │       ├── person_b_1.jpg
    # │       └── person_b_2.jpeg
    # └── encodings.pickle (will be created after running encode_known_faces)

    # --- Step 2: Encode the known faces ---
    # This will process the images in 'known_faces' and save their encodings.
    # Run this function whenever you add new people or new images for existing people.
    encode_known_faces(KNOWN_FACES_DIR, ENCODINGS_FILE)

    # Load the encoded faces for recognition
    known_encodings_data = load_encodings(ENCODINGS_FILE)

    if not known_encodings_data["encodings"]:
        print("No known faces encoded. Please add images to the 'known_faces' directory and re-run the encoding step.")
    else:
        print("\n--- Face Recognition Options ---")
        print("1. Recognize faces in a specific image file")
        print("2. Recognize faces from webcam (real-time)")
        choice = input("Enter your choice (1 or 2): ")

        if choice == '1':
            image_to_recognize_path = input("Enter the path to the image file (e.g., 'test_images/unknown_person.jpg'): ")
            recognize_faces_in_image(image_to_recognize_path, known_encodings_data)
        elif choice == '2':
            recognize_faces_webcam(known_encodings_data)
        else:
            print("Invalid choice. Please run the script again and choose 1 or 2.")

