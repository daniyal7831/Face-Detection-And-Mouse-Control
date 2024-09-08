import cv2
import dlib
import face_recognition
import numpy as np
import pickle
import pyautogui
import pyttsx3
import os

# Load the pre-trained face detector and shape predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Threshold for mouth open detection (adjusted to be more sensitive)
MOUTH_OPEN_THRESHOLD = 8  # Reduced for easier mouth opening

# Points for upper and lower lip in dlib's 68-point model
UPPER_LIP_POINTS = [62, 63, 64, 65, 66]
LOWER_LIP_POINTS = [67, 68, 69, 70]

# Disable PyAutoGUI fail-safe
pyautogui.FAILSAFE = False

def get_mouth_opening(landmarks):
    # Calculate the vertical distance between upper and lower lips
    top_lip = landmarks.part(62).y  # Center of the upper lip
    bottom_lip = landmarks.part(66).y  # Center of the lower lip
    return bottom_lip - top_lip

def detect_mouth_click(frame, gray, mouth_open):
    faces = detector(gray)
    new_mouth_open = False  # Track if the mouth is currently open

    for face in faces:
        landmarks = predictor(gray, face)

        if landmarks.num_parts < 68:
            continue

        mouth_opening = get_mouth_opening(landmarks)

        # Draw circles on upper and lower lip points for debugging
        for point in UPPER_LIP_POINTS + LOWER_LIP_POINTS:
            if point < landmarks.num_parts:
                x = landmarks.part(point).x
                y = landmarks.part(point).y
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        # Mouth click control
        if mouth_opening > MOUTH_OPEN_THRESHOLD:
            new_mouth_open = True  # Mouth is open
        else:
            new_mouth_open = False  # Mouth is closed

    # If the mouth just opened and wasn't open before, start holding the click
    if new_mouth_open and not mouth_open:
        pyautogui.mouseDown(button='left')  # Hold the left click
    # If the mouth just closed and was open before, release the click
    elif not new_mouth_open and mouth_open:
        pyautogui.mouseUp(button='left')  # Release the left click

    return frame, new_mouth_open

def register_face():
    video_capture = cv2.VideoCapture(0)
    face_encodings_list = []

    while True:
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            if face_encodings:
                face_encodings_list.append(face_encodings[0])
                print(f"Face registered! ({len(face_encodings_list)}/10)")

                if len(face_encodings_list) >= 10:
                    with open("face_data.pkl", "wb") as f:
                        pickle.dump(face_encodings_list, f)
                    print("Face registered successfully!")
                    break

        cv2.imshow("Register Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def recognize_face():
    video_capture = cv2.VideoCapture(0)
    engine = pyttsx3.init()

    with open("face_data.pkl", "rb") as f:
        registered_face_encodings = pickle.load(f)

    if len(np.array(registered_face_encodings).shape) == 1:
        registered_face_encodings = [registered_face_encodings]

    face_recognized = False

    while True:
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(registered_face_encodings, face_encoding, tolerance=0.4)
            face_distances = face_recognition.face_distance(registered_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index] and face_distances[best_match_index] < 0.4:
                face_recognized = True
                break

        if face_recognized:
            print("Face recognized!")
            engine.say("Face recognized!")
            engine.runAndWait()
            video_capture.release()
            cv2.destroyAllWindows()
            control_mouse_with_finger()
            break
        else:
            print("Face not recognized.")
            engine.say("Face not recognized. Exiting program.")
            engine.runAndWait()
            video_capture.release()
            cv2.destroyAllWindows()
            break

def control_mouse_with_finger():
    import mediapipe as mp

    video_capture = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    screen_width, screen_height = pyautogui.size()
    prev_x, prev_y = None, None
    prev_mouse_x, prev_mouse_y = pyautogui.position()

    speed_factor = 10
    smoothing_factor = 10

    positions = []
    mouth_open = False  # Track mouth status

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        result = hands.process(rgb_frame)

        # Mouth click detection
        frame, mouth_open = detect_mouth_click(frame, gray_frame, mouth_open)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
            y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y

            h, w, _ = frame.shape
            finger_x = int(x * w)
            finger_y = int(y * h)

            cv2.circle(frame, (finger_x, finger_y), 5, (255, 255, 0), -1)

            positions.append((x, y))

            if len(positions) > smoothing_factor:
                positions.pop(0)

            avg_x = np.mean([pos[0] for pos in positions])
            avg_y = np.mean([pos[1] for pos in positions])

            if prev_x is not None and prev_y is not None:
                rel_x = (avg_x - prev_x) * w * speed_factor
                rel_y = (prev_y - avg_y) * h * speed_factor
                current_mouse_x, current_mouse_y = pyautogui.position()
                pyautogui.moveTo(current_mouse_x + rel_x, current_mouse_y - rel_y)

            prev_x, prev_y = avg_x, avg_y

        cv2.imshow("Control Mouse with Finger", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    hands.close()
    video_capture.release()
    cv2.destroyAllWindows()

def main():
    if not os.path.exists("face_data.pkl"):
        print("No face data file found, please register your face.")
        register_face()
    recognize_face()

if __name__ == "__main__":
    main()