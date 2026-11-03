# Video human action detection

Find people in a video and recognise what they are doing, using pretrained
deep-learning models.

## Task

Given a video file or a live video stream:

1. Detect people in each frame.
2. Recognise human actions, both for the whole scene and for each detected person.
3. Produce an annotated output video with boxes and action labels.

## Approach

Three approaches were tried, from simplest to most detailed:

| Approach | Model | Output | Code |
|---|---|---|---|
| Clip-level activity recognition | ResNet-34 3D CNN trained on Kinetics-400 (400 classes), ONNX, OpenCV DNN | one label per 16-frame clip | [src/clip_activity_recognition.py](src/clip_activity_recognition.py) |
| Person detection | YOLOv4 (COCO), OpenCV DNN | person boxes and people count per frame | [src/person_detection_yolo.py](src/person_detection_yolo.py) |
| Spatio-temporal action detection | Detectron2 Faster R-CNN for people + PyTorchVideo `slow_r50_detection` (Kinetics-400 → AVA v2.2, 80 actions) | action labels **per person** every second | [notebooks/spatiotemporal_action_detection.ipynb](notebooks/spatiotemporal_action_detection.ipynb) |

**Trade-offs**
- Clip-level recognition gives a single label for the whole frame, so it cannot tell
  apart several people in one scene. Kinetics-400 classes are mostly sports and hobbies.
- Person detection gives boxes and counts (occupancy, queues) but no actions.
- Per-person action detection combines both: AVA classes such as *stand, sit, walk,
  talk to, carry/hold, eat, drink, answer phone* describe everyday indoor scenes best,
  at the cost of running two models per clip.

## Project structure

```
video-human-action-detection/
├── README.md
├── requirements.txt
├── labels/
│   └── kinetics400_classes.txt            # Kinetics-400 class names
├── notebooks/
│   └── spatiotemporal_action_detection.ipynb
└── src/
    ├── clip_activity_recognition.py       # Kinetics-400 clip classification, batch or rolling window
    ├── person_detection_yolo.py           # YOLOv4 person detection
    ├── visualization.py                   # third-party video visualiser (PyTorchVideo)
    └── LICENSE-pytorchvideo.txt           # Apache License 2.0 for visualization.py
```

## Setup

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
The OpenCV scripts need only `opencv-python`, `imutils` and `numpy`; the rest is for the notebook.
Detectron2 (notebook only): install into the same environment following the
[installation guide](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md).

### Models (not included)
- **ResNet-34 Kinetics-400 ONNX** (`resnet-34_kinetics.onnx`): the model used in OpenCV's
  `samples/dnn/action_recognition.py`, converted from
  [kenshohara/3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch).
- **YOLOv4**: `yolov4.weights` and `yolov4.cfg` from
  [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet), `coco.names` from the same repo
  (`data/coco.names`). Put them in `models/`.
- **Notebook models** are downloaded automatically by Torch Hub and the Detectron2 model zoo.

## Usage

```sh
cd src

# one activity label per 16-frame block, write annotated video
python clip_activity_recognition.py -m ../models/resnet-34_kinetics.onnx \
    -c ../labels/kinetics400_classes.txt -i input.mp4 -o output.mp4

# rolling window of the last 16 frames, webcam
python clip_activity_recognition.py -m ../models/resnet-34_kinetics.onnx \
    -c ../labels/kinetics400_classes.txt --mode rolling

# person detection on a file or an IP camera
python person_detection_yolo.py --weights ../models/yolov4.weights \
    --cfg ../models/yolov4.cfg --names ../models/coco.names -i input.mp4
```
For the notebook, set `input_file_path` to your video and run all cells from `notebooks/` (`jupyter lab`).

## Data

No video is included. Use your own video or any public clip.

## Credits
- Spatio-temporal detection notebook adapted from the
  [PyTorchVideo](https://github.com/facebookresearch/pytorchvideo) detection inference tutorial;
  `src/visualization.py` is taken from the same tutorial (© Facebook, Apache License 2.0;
  licence text in [src/LICENSE-pytorchvideo.txt](src/LICENSE-pytorchvideo.txt)).
- Clip-level recognition based on the GeeksforGeeks article
  [Human Activity Recognition with OpenCV](https://www.geeksforgeeks.org/human-activity-recognition-with-opencv/)
  and the PyImageSearch tutorial on human activity recognition with OpenCV.
