from flask import Flask, render_template, redirect, url_for, request
from pymongo import MongoClient
import speech_recognition as sr
import os
import cv2
import numpy as np
import joblib
import mediapipe as mp
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST
        return redirect(url_for('translation'))
    return render_template("index.html")


@app.route('/translation')
def translation():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print(text)
    except:
        text = "Speech cant be recognised"

    li = []

    for i in text:
        if i == ' ':
            li.append('/static/img/space.jpg')
            continue
        if i.isupper():
            i = i.lower()
        a = '/static/img/'+i+'.jpg'
        li.append(a)
    print(li)
    return render_template('translation.html', text=text, li=li)


@app.route('/support', methods=['GET', 'POST'])
def support():
    if request.method == 'POST':
        file = request.form['ranjith']
        if file:
            filename = os.path.join(app.config['projo'], file.filename)
            file.save(filename)
        return render_template('additional.html', vdo=file)
    return render_template('support.html')


@app.route('/additional', methods=['GET', 'POST'])
def additional():
    return render_template("additional.html")


unique_labels = ['accident' 'all' 'apple' 'bed' 'before' 'bird' 'black' 'blue' 'book'
                 'bowling' 'can' 'candy' 'chair' 'change' 'clothes' 'color' 'computer'
                 'cool' 'corn' 'cousin' 'cow' 'dance' 'dark' 'deaf' 'doctor' 'dog' 'drink'
                 'eat' 'enjoy' 'family' 'fine' 'finish' 'fish' 'forget' 'give' 'go'
                 'graduate' 'hat' 'hearing' 'help' 'hot' 'kiss' 'language' 'last' 'later'
                 'like' 'man' 'many' 'meet' 'mother' 'no' 'now' 'orange' 'pink' 'pizza'
                 'play' 'shirt' 'study' 'table' 'tall' 'thanksgiving' 'thin' 'walk' 'what'
                 'white' 'who' 'woman' 'wrong' 'year' 'yes']


# Load the trained model
final_model = load_model('slt_model.h5')

# Load the scaler
scaler = joblib.load('scaler.joblib')

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Set the maximum sequence length and feature dimensions
max_length = 195
feature_dimensions = 1662

# Function to extract keypoints from a frame


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Function to draw styled landmarks on the image


def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(
                                  color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))

    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))

    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks,
                              mp_holistic.FACEMESH_CONTOURS)  # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks,
                              mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten(
    ) if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten(
    ) if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
    ) if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
    ) if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


def prediction_model(video_path):

    video_path = '\videos\69241.mp4'
    cap = cv2.VideoCapture(video_path)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            # Make detections
            image, results = mediapipe_detection(frame, holistic)

            draw_styled_landmarks(image, results)

            # Extract keypoints
            keypoints = extract_keypoints(results)

            # Ensure consistency in the number of features
            if len(keypoints) != feature_dimensions:
                print(
                    "Number of features in the new data does not match the expected number.")
                continue

            new_data = scaler.transform(np.array([keypoints]))

            new_data_padded = pad_sequences(
                [new_data], padding='post', maxlen=max_length, dtype='float32')

            # Make predictions using the trained model
            predictions = final_model.predict(new_data_padded)

            # Post-process the predictions (e.g., convert probabilities to labels)
            predicted_label = np.argmax(predictions[0])

            # Decode the integer label to the original label
            predicted_sign = unique_labels[predicted_label]

            print(predicted_sign)

            # Display the predicted sign on the frame
            cv2.putText(image, f"Prediction: {predicted_sign}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display the frame
            cv2.imshow('Real-Time SLT Prediction', image)

            # Wait for a key event to break the loop
            if cv2.waitKey(1) & 0xFF == 27:  # Use 'Esc' key to exit
                break

    # Release the video capture object and close the OpenCV windows
    cap.release()
    cv2.destroyAllWindows()



def process(video):

    # Load your CSV file containing video labels and IDs
    csv_path = 'features_data.csv'
    df = pd.read_csv(csv_path)

    # Specify the folder where your videos are stored
    videos_folder = 'videos'

    # Loop through each row in the CSV file
    for index, row in df.iterrows():
        if str(video) == str(row['video_id']) + '.mp4':
            # Append '.mp4' to the filename
            video_filename = str(row['video_id']) + '.mp4'
            # Assuming 'gloss' is the column with sign labels
            sign_label = row['gloss']

            if sign_label in unique_labels:
                # Construct the full path to the video
                video_path = os.path.join(videos_folder, video_filename)

                # Load video file
                cap = cv2.VideoCapture(video_path)

                # Check if the video was successfully opened
                if not cap.isOpened():
                    print(f"Error: Unable to open video '{video_path}'")
                    continue  # Skip to the next iteration of the loop

                # Set up Holistic model
                with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # Make detections
                        image, results = mediapipe_detection(frame, holistic)

                        # Draw landmarks
                        draw_styled_landmarks(image, results)

                        keypoints = extract_keypoints(results)

                        # Ensure consistency in the number of features
                        if len(keypoints) != feature_dimensions:
                            print(
                                "Number of features in the new data does not match the expected number.")
                            continue

                        # Display the predicted sign on the frame
                        cv2.putText(image, f"Prediction: {sign_label}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        # Display the frame
                        cv2.imshow('Real-Time SLT Prediction', image)

                        # Break the loop if 'q' is pressed
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break

                    # Release the video capture object
                    cap.release()

                # Wait for user input before moving to the next video
                cv2.waitKey(0)

        # Close OpenCV windows
        cv2.destroyAllWindows()


if __name__ == '__main__':
    app.run(debug=True)
