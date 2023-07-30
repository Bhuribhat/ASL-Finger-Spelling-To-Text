import cv2
import tensorflow as tf


# Set length of frames to 128
FRAME_LENGTH = 128
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Colors RGB Format
COLOR = {
    "BLACK"  : (0, 0, 0),
    "RED"    : (255, 0, 0),
    "GREEN"  : (0, 255, 0),
    "BLUE"   : (0, 0, 255),
    "YELLOW" : (0, 255, 255),
    "WHITE"  : (255, 255, 255)
}


X = [f'x_right_hand_{i}' for i in range(21)] + [f'x_left_hand_{i}' for i in range(21)]
Y = [f'y_right_hand_{i}' for i in range(21)] + [f'y_left_hand_{i}' for i in range(21)]
Z = [f'z_right_hand_{i}' for i in range(21)] + [f'z_left_hand_{i}' for i in range(21)]

# Pose coordinates for hand movement
FEATURE_COLUMNS = X + Y + Z

X_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "x_" in col]
Y_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "y_" in col]
Z_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "z_" in col]

RHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "right" in col]
LHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if  "left" in col]


def resize_pad(x):
    if tf.shape(x)[0] < FRAME_LENGTH:
        x = tf.pad(x, ([[0, FRAME_LENGTH - tf.shape(x)[0]], [0, 0], [0, 0]]))
    else:
        x = tf.image.resize(x, (FRAME_LENGTH, tf.shape(x)[1]))
    return x


def pre_process(x):
    rhand = tf.gather(x, RHAND_IDX, axis=1)
    lhand = tf.gather(x, LHAND_IDX, axis=1)
    
    rnan_idx = tf.reduce_any(tf.math.is_nan(rhand), axis=1)
    lnan_idx = tf.reduce_any(tf.math.is_nan(lhand), axis=1)
    
    rnans = tf.math.count_nonzero(rnan_idx)
    lnans = tf.math.count_nonzero(lnan_idx)
    
    # For dominant hand
    if rnans > lnans:
        hand = lhand
        
        hand_x = hand[:, 0 * (len(LHAND_IDX) // 3) : 1 * (len(LHAND_IDX) // 3)]
        hand_y = hand[:, 1 * (len(LHAND_IDX) // 3) : 2 * (len(LHAND_IDX) // 3)]
        hand_z = hand[:, 2 * (len(LHAND_IDX) // 3) : 3 * (len(LHAND_IDX) // 3)]
        hand = tf.concat([1 - hand_x, hand_y, hand_z], axis=1)
    else:
        hand = rhand
    
    hand_x = hand[:, 0 * (len(LHAND_IDX) // 3) : 1 * (len(LHAND_IDX) // 3)]
    hand_y = hand[:, 1 * (len(LHAND_IDX) // 3) : 2 * (len(LHAND_IDX) // 3)]
    hand_z = hand[:, 2 * (len(LHAND_IDX) // 3) : 3 * (len(LHAND_IDX) // 3)]
    hand = tf.concat([hand_x[..., tf.newaxis], hand_y[..., tf.newaxis], hand_z[..., tf.newaxis]], axis=-1)
    
    mean = tf.math.reduce_mean(hand, axis=1)[:, tf.newaxis, :]
    std = tf.math.reduce_std(hand, axis=1)[:, tf.newaxis, :]
    hand = (hand - mean) / std
    
    # x = tf.concat([hand, pose], axis=1)
    x = tf.concat([hand], axis=1)
    x = resize_pad(x)
    x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
    
    # x = tf.reshape(x, (FRAME_LENGTH, len(LHAND_IDX) + len(LPOSE_IDX)))
    x = tf.reshape(x, (FRAME_LENGTH, len(LHAND_IDX)))
    return x


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), COLOR['BLACK'], 6)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), COLOR['WHITE'], 2)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), COLOR['BLACK'], 6)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), COLOR['WHITE'], 2)

        # Index finger
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), COLOR['BLACK'], 6)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), COLOR['WHITE'], 2)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), COLOR['BLACK'], 6)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), COLOR['WHITE'], 2)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), COLOR['BLACK'], 6)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), COLOR['WHITE'], 2)

        # Middle finger
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), COLOR['BLACK'], 6)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), COLOR['WHITE'], 2)
        cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), COLOR['BLACK'], 6)
        cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), COLOR['WHITE'], 2)
        cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), COLOR['BLACK'], 6)
        cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), COLOR['WHITE'], 2)

        # Ring finger
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), COLOR['BLACK'], 6)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), COLOR['WHITE'], 2)
        cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), COLOR['BLACK'], 6)
        cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), COLOR['WHITE'], 2)
        cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), COLOR['BLACK'], 6)
        cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), COLOR['WHITE'], 2)

        # Little finger
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), COLOR['BLACK'], 6)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), COLOR['WHITE'], 2)
        cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), COLOR['BLACK'], 6)
        cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), COLOR['WHITE'], 2)
        cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), COLOR['BLACK'], 6)
        cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), COLOR['WHITE'], 2)

        # Palm
        cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), COLOR['BLACK'], 6)
        cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), COLOR['WHITE'], 2)
        cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), COLOR['BLACK'], 6)
        cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), COLOR['WHITE'], 2)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), COLOR['BLACK'], 6)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), COLOR['WHITE'], 2)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), COLOR['BLACK'], 6)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), COLOR['WHITE'], 2)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), COLOR['BLACK'], 6)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), COLOR['WHITE'], 2)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), COLOR['BLACK'], 6)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), COLOR['WHITE'], 2)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), COLOR['BLACK'], 6)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), COLOR['WHITE'], 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:
            cv2.circle(image, (landmark[0], landmark[1]), 5, COLOR['WHITE'], -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, COLOR['BLACK'],  1)
        if index == 1:
            cv2.circle(image, (landmark[0], landmark[1]), 5, COLOR['WHITE'], -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, COLOR['BLACK'],  1)
        if index == 2:
            cv2.circle(image, (landmark[0], landmark[1]), 5, COLOR['WHITE'], -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, COLOR['BLACK'],  1)
        if index == 3:
            cv2.circle(image, (landmark[0], landmark[1]), 5, COLOR['WHITE'], -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, COLOR['BLACK'],  1)
        if index == 4:
            cv2.circle(image, (landmark[0], landmark[1]), 8, COLOR['WHITE'], -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, COLOR['BLACK'],  1)
        if index == 5:
            cv2.circle(image, (landmark[0], landmark[1]), 5, COLOR['WHITE'], -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, COLOR['BLACK'],  1)
        if index == 6:
            cv2.circle(image, (landmark[0], landmark[1]), 5, COLOR['WHITE'], -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, COLOR['BLACK'],  1)
        if index == 7:
            cv2.circle(image, (landmark[0], landmark[1]), 5, COLOR['WHITE'], -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, COLOR['BLACK'],  1)
        if index == 8:
            cv2.circle(image, (landmark[0], landmark[1]), 8, COLOR['WHITE'], -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, COLOR['BLACK'],  1)
        if index == 9:
            cv2.circle(image, (landmark[0], landmark[1]), 5, COLOR['WHITE'], -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, COLOR['BLACK'],  1)
        if index == 10:
            cv2.circle(image, (landmark[0], landmark[1]), 5, COLOR['WHITE'], -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, COLOR['BLACK'],  1)
        if index == 11:
            cv2.circle(image, (landmark[0], landmark[1]), 5, COLOR['WHITE'], -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, COLOR['BLACK'],  1)
        if index == 12:
            cv2.circle(image, (landmark[0], landmark[1]), 8, COLOR['WHITE'], -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, COLOR['BLACK'],  1)
        if index == 13:
            cv2.circle(image, (landmark[0], landmark[1]), 5, COLOR['WHITE'], -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, COLOR['BLACK'],  1)
        if index == 14:
            cv2.circle(image, (landmark[0], landmark[1]), 5, COLOR['WHITE'], -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, COLOR['BLACK'],  1)
        if index == 15:
            cv2.circle(image, (landmark[0], landmark[1]), 5, COLOR['WHITE'], -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, COLOR['BLACK'],  1)
        if index == 16:
            cv2.circle(image, (landmark[0], landmark[1]), 8, COLOR['WHITE'], -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, COLOR['BLACK'],  1)
        if index == 17:
            cv2.circle(image, (landmark[0], landmark[1]), 5, COLOR['WHITE'], -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, COLOR['BLACK'],  1)
        if index == 18:
            cv2.circle(image, (landmark[0], landmark[1]), 5, COLOR['WHITE'], -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, COLOR['BLACK'],  1)
        if index == 19:
            cv2.circle(image, (landmark[0], landmark[1]), 5, COLOR['WHITE'], -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, COLOR['BLACK'],  1)
        if index == 20:
            cv2.circle(image, (landmark[0], landmark[1]), 8, COLOR['WHITE'], -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, COLOR['BLACK'],  1)

    return image


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    # Keypoint
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width ), image_width  - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])
    return landmark_point


def draw_info_text(image, pos, hand_sign_text):
    cv2.rectangle(
        image, (pos[0] - 2, pos[1]), (pos[2] + 2, pos[1] - 20),
        COLOR['BLACK'], -1
    )

    info_text = ""
    if hand_sign_text != "":
        info_text = hand_sign_text
    cv2.putText(
        image, info_text, (pos[0] + 5, pos[1] - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR['WHITE'], 1, cv2.LINE_AA
    )
    return image