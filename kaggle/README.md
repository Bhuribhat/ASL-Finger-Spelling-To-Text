# Description

American Sign Language Fingerspelling Recognition: [Google I/O 2023](https://blog.tensorflow.org/2023/05/american-sign-language-fingerspelling-recognition.html)

- Handle space
- Handle changing gesture
- Handle duplicate characters
- Handle numbers
- Handle !#$%&'()*+,-./:;=?@[_~


## Model

To support the full functionality of our demo. You need the following models located in these paths:

```bash
ASL-Finger-Spelling-To-Text
└── kaggle
    ├── character_to_prediction_index.json
    └── model
        └── model.tflite
```

To inference the model:

```bash
$ python inference.py
```


## Goal of the Competition

The goal of this competition is to detect and translate American Sign Language (ASL) fingerspelling into text. You will create a model trained on the largest dataset of its kind, released specifically for this competition. The data includes more than three million fingerspelled characters produced by over 100 Deaf signers captured via the selfie camera of a smartphone with a variety of backgrounds and lighting conditions.

Your work may help move sign language recognition forward, making AI more accessible for the Deaf and Hard of Hearing community.


## Getting Started Notebook

> To get started quickly, feel free to take advantage of [this starter notebook](https://github.com/Bhuribhat/ASL-Finger-Spelling-To-Text/blob/main/kaggle/ASL_Walkthrough.ipynb).


## Context

Voice-enabled assistants open the world of useful and sometimes life-changing features of modern devices. These revolutionary AI solutions include automated speech recognition (ASR) and machine translation. Unfortunately, these technologies are often not accessible to the more than 70 million Deaf people around the world who use sign language to communicate, nor to the 1.5+ billion people affected by hearing loss globally.

Fingerspelling uses hand shapes that represent individual letters to convey words. While fingerspelling is only a part of ASL, it is often used for communicating names, addresses, phone numbers, and other information commonly entered on a mobile phone. Many Deaf smartphone users can fingerspell words faster than they can type on mobile keyboards. In fact, ASL fingerspelling can be substantially faster than typing on a smartphone’s virtual keyboard (57 words/minute average versus 36 words/minute US average). But sign language recognition AI for text entry lags far behind voice-to-text or even gesture-based typing, as robust datasets didn't previously exist.

Technology that understands sign language fits squarely within Google's mission to organize the world's information and make it universally accessible and useful. Google’s AI principles also support this idea and encourage Google to make products that empower people, widely benefit current and future generations, and work for the common good. This collaboration between Google and the Deaf Professional Arts Network will explore AI solutions that can be scaled globally (such as other sign languages), and support individual user experience needs while interacting with products.

Your participation in this competition could help provide Deaf and Hard of Hearing users the option to fingerspell words instead of using a keyboard. Besides convenient text entry for web search, map directions, and texting, there is potential for an app that can then translate this input using sign language-to-speech technology to speak the words. Such an app would enable the Deaf and Hard of Hearing community to communicate with hearing non-signers more quickly and smoothly.


# Dataset Description

1. **[train/supplemental_metadata].csv**

- `path` - The path to the landmark file.
- `file_id` - A unique identifier for the data file.
- `participant_id` - A unique identifier for the data contributor.
- `sequence_id` - A unique identifier for the landmark sequence. Each data file may contain many sequences.
- `phrase` - The labels for the landmark sequence. The train and test datasets contain randomly generated addresses, phone numbers, and urls derived from components of real addresses/phone numbers/urls. Any overlap with real addresses, phone numbers, or urls is purely accidental. The supplemental dataset consists of fingerspelled sentences. Note that some of the urls include adult content. The intent of this competition is to support the Deaf and Hard of Hearing community in engaging with technology on an equal footing with other adults.

2. **character_to_prediction_index.json**

|  Character  |  Ordinal Encoding  |  Character  |  Ordinal Encoding  |
|:-----------:|:------------------:|:-----------:|:------------------:|
| !           | 1                  | [           | 30                 |
| #           | 2                  | _           | 31                 |
| $           | 3                  | a           | 32                 |
| %           | 4                  | b           | 33                 |
| &           | 5                  | c           | 34                 |
| '           | 6                  | d           | 35                 |
| (           | 7                  | e           | 36                 |
| )           | 8                  | f           | 37                 |
| *           | 9                  | g           | 38                 |
| +           | 10                 | h           | 39                 |
| ,           | 11                 | i           | 40                 |
| -           | 12                 | j           | 41                 |
| .           | 13                 | k           | 42                 |
| /           | 14                 | l           | 43                 |
| 0           | 15                 | m           | 44                 |
| 1           | 16                 | n           | 45                 |
| 2           | 17                 | o           | 46                 |
| 3           | 18                 | p           | 47                 |
| 4           | 19                 | q           | 48                 |
| 5           | 20                 | r           | 49                 |
| 6           | 21                 | s           | 50                 |
| 7           | 22                 | t           | 51                 |
| 8           | 23                 | u           | 52                 |
| 9           | 24                 | v           | 53                 |
| :           | 25                 | w           | 54                 |
| ;           | 26                 | x           | 55                 |
| =           | 27                 | y           | 56                 |
| ?           | 28                 | z           | 57                 |
| @           | 29                 | ~           | 58                 |


> **Note:** ordinal encoding 0 for `space`

3. **[train/supplemental]_landmarks/** The landmark data. The landmarks were extracted from raw videos with the [MediaPipe holistic model](https://github.com/google/mediapipe/blob/master/docs/solutions/holistic.md). Not all of the frames necessarily had visible hands or hands that could be detected by the model.

The landmark files contain the same data as in the ASL Signs competition (minus the row ID column) but reshaped into a wide format. This allows you to take advantage of the Parquet format to entirely skip loading landmarks that you aren't using.

- `sequence_id` - A unique identifier for the landmark sequence. landmark files contain approximately 1,000 sequences. The sequence ID is used as the dataframe index.
- `frame` - The frame number within a landmark sequence.
- `[x/y/z]_[type]_[landmark_index]` - There are now 1,629 spatial coordinate columns for the x, y and z coordinates for each of the 543 landmarks. The type of landmark is one of `['face', 'left_hand', 'pose', 'right_hand']`. Details of the [hand landmark locations can be found here](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker). The spatial coordinates have already been normalized by MediaPipe. Note that the MediaPipe model is not fully trained to predict depth so you may wish to ignore the z values. The landmarks have been converted to float32.

Landmark data should not be used to identify or re-identify an individual. Landmark data is not intended to enable any form of identity recognition or store any unique biometric identification.


# Evaluation

The evaluation metric for this contest is the normalized total [levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance). Let the total number of characters in all of the labels be `N` and the total levenshtein distance be `D`. The metric equals `(N - D) / N`.

Each video is loaded with the following function:

```py
def load_relevant_data_subset(pq_path):
    return pd.read_parquet(pq_path, columns=selected_columns)
```

If you want to load only a subset of the landmarks, include a file named `inference_args.json` in your `submission.zip` with the field `selected_columns` containing a list of the landmark columns you want to use. If that is not included we will load all columns.

Inference is performed (roughly) as follows, ignoring details like how we manage multiple videos:

```py
import tflite_runtime.interpreter as tflite
interpreter = tflite.Interpreter(model_path)

REQUIRED_SIGNATURE = "serving_default"
REQUIRED_OUTPUT = "outputs"

with open ("/kaggle/input/fingerspelling-character-map/character_to_prediction_index.json", "r") as f:
    character_map = json.load(f)
rev_character_map = {j:i for i, j in character_map.items()}

found_signatures = list(interpreter.get_signature_list().keys())

if REQUIRED_SIGNATURE not in found_signatures:
    raise KernelEvalException('Required input signature not found.')

prediction_fn = interpreter.get_signature_runner("serving_default")
output = prediction_fn(inputs=frames)
prediction_str = "".join([rev_character_map.get(s, "") for s in np.argmax(output[REQUIRED_OUTPUT], axis=1)])
```

# Acknowledgements

Thanks to the Deaf Professional Arts Network and their community of Deaf signers who made this dataset possible. Thanks also to the students of RIT/NTID who interned at Google to help set this project's agenda and create the data collection tools.
