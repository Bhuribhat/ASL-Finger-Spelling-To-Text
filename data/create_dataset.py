import os
import cv2
import pickle
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=True, 
    min_detection_confidence=0.5,
    max_num_hands=1
)


def generate_keypoints(hands, image_path):
    data_aux = []
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_values = [lm.x for lm in hand_landmarks.landmark]
            y_values = [lm.y for lm in hand_landmarks.landmark]

            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(x_values[i] - min(x_values))
                data_aux.append(y_values[i] - min(y_values))

    return data_aux
    

def build_dataset(hands, data_dir, letter=True):
    data = []
    labels = []

    for label_dir in os.listdir(data_dir):
        if letter and (not label_dir.isalpha() and label_dir != "UNKNOWN_LETTER"):
            continue
        if not letter and (not label_dir.isnumeric() and label_dir != "UNKNOWN_NUMBER"):
            continue
        for image_path in os.listdir(os.path.join(data_dir, label_dir)):
            image_path = os.path.join(data_dir, label_dir, image_path)
            data_aux = generate_keypoints(hands, image_path)

            # Save "Ascii" label or "Unknown" label
            if data_aux:
                data.append(data_aux)
                labels.append(label_dir)

    # Save Dataset to pickle file
    saved_name = "data" if letter else "number"
    with open(f'{data_dir}/{saved_name}.pickle', 'wb') as dataset:
        pickle.dump({'data': data, 'labels': labels}, dataset)

    # Display all unique elements in labels sorted by length and alphabetically
    print(f"Labels: {sorted(set(labels), key=lambda item: (len(item), item))}")
    

if __name__ == '__main__':
    build_dataset(hands, data_dir='.')