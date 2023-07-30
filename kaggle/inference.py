import cv2
import json
import numpy as np
import mediapipe as mp
import tensorflow as tf

from utils import FRAME_LENGTH, FONT, COLOR
from utils import draw_landmarks, calc_landmark_list, draw_info_text

# Debug Mode
DEBUG = False

# Initialize Mediapipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False, max_num_hands=1
)


# Draw bounding box and style hand landmark
def hand_detection(frame, results, hand_landmarks, sequence_of_landmarks):
    multi_handedness = results.multi_handedness
    handness = multi_handedness[0].classification[0].label
    h, w, _ = frame.shape

    # mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
    landmark_list = calc_landmark_list(frame, hand_landmarks)
    frame = draw_landmarks(frame, landmark_list)

    # Get (x, y) coordinates of hand landmarks
    x_values = [lm.x for lm in hand_landmarks.landmark]
    y_values = [lm.y for lm in hand_landmarks.landmark]

    # Get Minimum and Maximum Values
    min_x = int(min(x_values) * w)
    max_x = int(max(x_values) * w)
    min_y = int(min(y_values) * h)
    max_y = int(max(y_values) * h)

    # Get the 3D hand landmarks
    hand_coordinates = []
    for landmark in hand_landmarks.landmark:
        hand_coordinates.extend([landmark.x, landmark.y, landmark.z])

    # Pad the hand_coordinates list with zeros if less than 21 landmarks detected
    hand_coordinates += [0.0] * (63 - len(hand_coordinates))
    sequence_of_landmarks.append(hand_coordinates)

    # Draw Bounding Box and Text Info
    frame = draw_info_text(frame, [min_x - 20, min_y - 10, max_x + 20, max_y + 10], f"{handness} Hand")
    cv2.rectangle(frame, (min_x - 20, min_y - 10), (max_x + 20, max_y + 10), COLOR['BLACK'], 4)
    return frame


# Create a VideoCapture object to access the camera (you can also load a video file)
def generate_text_fingerspelling(video_path=0):
    video_capture = cv2.VideoCapture(video_path)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    video_capture.set(cv2.CAP_PROP_FPS, 30)

    # List to store the sequence of landmarks
    global output_text
    output_text = ""
    sequence_of_landmarks = []

    while video_capture.isOpened():
        success, frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        if not success:
            break

        # Convert the BGR frame to RGB (mediapipe requires RGB input)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the RGB frame with Mediapipe Hands to obtain the landmarks
        results = mp_hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                frame = hand_detection(frame, results, hand_landmarks, sequence_of_landmarks)

        # Get number of sequence landmark
        landmark_size = len(sequence_of_landmarks)
        
        # Display Text if reach frame length limit
        if landmark_size == FRAME_LENGTH:
            sequence_input = np.asarray(sequence_of_landmarks, dtype=np.float32)
            output_text += inference_tflite(sequence_input) + ' '
            sequence_of_landmarks.clear()

        # Display the current frame
        landmark_text = f"Landmark Size: {landmark_size}"
        landmark_size = cv2.getTextSize(landmark_text, FONT, 0.5, 2)[0]
        cv2.rectangle(frame, (5, 0), (10 + landmark_size[0], 10 + landmark_size[1]), COLOR['YELLOW'], -1)
        cv2.putText(frame, landmark_text, (10, 15), FONT, 0.5, COLOR['BLACK'], 2)

        output_size = cv2.getTextSize(output_text[:-1], FONT, 0.5, 2)[0]
        cv2.rectangle(frame, (5, 30), (10 + output_size[0], 40 + output_size[1]), COLOR['YELLOW'], -1)
        cv2.putText(frame, output_text, (10, 45), FONT, 0.5, COLOR['BLACK'], 2)
        cv2.imshow('Fingerspelling Detection', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture and destroy any OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()


# TFLite Inference
def inference_tflite(sequence_of_landmarks):
    interpreter = tf.lite.Interpreter("./model/model.tflite")
    interpreter.allocate_tensors()

    REQUIRED_SIGNATURE = "serving_default"
    REQUIRED_OUTPUT = "outputs"

    with open ("./character_to_prediction_index.json", "r") as f:
        character_map = json.load(f)

    rev_character_map = {j:i for i, j in character_map.items()}
    found_signatures = list(interpreter.get_signature_list().keys())

    if REQUIRED_SIGNATURE not in found_signatures:
        raise Exception('Required input signature not found.')

    # Prediction is batch[0] shape (128, 63)
    prediction_fn = interpreter.get_signature_runner("serving_default")
    output = prediction_fn(inputs=sequence_of_landmarks)
    characters_idx = [char_idx for char_idx in np.argmax(output[REQUIRED_OUTPUT], axis=1)]
    prediction_str = "".join([rev_character_map.get(s, "") for s in characters_idx])

    # Print the shape of sequence_of_landmarks for debugging
    if DEBUG == True:
        print("Input shape:", sequence_of_landmarks.shape)
        print(f"Prediction: {prediction_str}")

    return prediction_str


if __name__ == '__main__':
    generate_text_fingerspelling(video_path=0)
    print(f"Recognition: {output_text}")