import os
import cv2

DATA_DIR = '.'
GREEN = (0, 255, 0)

def collect_images(class_labels, data_dir, dataset_size=100):
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    capture.set(cv2.CAP_PROP_FPS, 30)

    for label in class_labels:
        label_path = os.path.join(data_dir, label.upper())
        if not os.path.exists(label_path):
            os.makedirs(label_path)
        print('Collecting data for class {}'.format(label.upper()))

        # Press 'r' if you are ready
        while True:
            success, frame = capture.read()
            frame = cv2.flip(frame, 1)

            # setup text
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = 'Ready? Press "R"'

            # get boundary of this text
            textsize = cv2.getTextSize(text, font, 1.3, 3)[0]

            # get coords based on boundary
            textX = (frame.shape[1] - textsize[0]) // 2
            textY = (frame.shape[0] + textsize[1]) // 2

            cv2.putText(
                frame, text, (textX, textY), 
                font, 1.3, GREEN, 3,
                cv2.LINE_AA
            )

            cv2.imshow('frame', frame)
            if cv2.waitKey(10) == ord('r'):
                break

        for counter in range(dataset_size):
            success, frame = capture.read()
            cv2.imshow('frame', cv2.flip(frame, 1))
            cv2.waitKey(10)
            cv2.imwrite(
                os.path.join(label_path, f'{100+counter}.jpg'), 
                cv2.flip(frame, 1)
            )

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    classes = ['M', 'N']
    dataset_size = 200
    collect_images(classes, DATA_DIR, dataset_size)