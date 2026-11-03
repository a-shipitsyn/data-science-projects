# HUMAN ACTIVITY RECOGNITION on video clips
# ResNet-34 trained on Kinetics-400 (ONNX), run with OpenCV DNN.
# Based on https://www.geeksforgeeks.org/human-activity-recognition-with-opencv/
# and the PyImageSearch "Human activity recognition with OpenCV and deep learning" tutorial.
#
# Two sampling modes:
#   batch   - classify consecutive non-overlapping blocks of 16 frames, label all frames of the block
#   rolling - keep a rolling window of the last 16 frames, classify on every new frame
#
# Usage:
#   python clip_activity_recognition.py -m resnet-34_kinetics.onnx -c ../labels/kinetics400_classes.txt -i input.mp4 -o output.mp4
#   python clip_activity_recognition.py -m resnet-34_kinetics.onnx -c ../labels/kinetics400_classes.txt --mode rolling   # webcam

from collections import deque
import argparse

import cv2
import imutils
import numpy as np

SAMPLE_DURATION = 16  # number of frames per classified clip
SAMPLE_SIZE = 112     # spatial size expected by the model
MEAN = (114.7748, 107.7354, 99.4750)


def parse_args():
    ap = argparse.ArgumentParser(description="Human activity recognition on video with a Kinetics-400 model")
    ap.add_argument("-m", "--model", required=True, help="path to pre-trained ONNX model")
    ap.add_argument("-c", "--classes", required=True, help="path to class labels file")
    ap.add_argument("-i", "--input", type=str, default="", help="path to video file (default: webcam)")
    ap.add_argument("-o", "--output", type=str, default="", help="path to output video file")
    ap.add_argument("-d", "--display", type=int, default=1, help="display output frames or not")
    ap.add_argument("-g", "--gpu", type=int, default=0, help="use CUDA backend or not")
    ap.add_argument("--mode", choices=["batch", "rolling"], default="batch", help="frame sampling mode")
    return vars(ap.parse_args())


def load_model(model_path, gpu=False):
    print("[INFO] Loading the human activity recognition model...")
    net = cv2.dnn.readNet(model_path)
    if gpu:
        print("[INFO] Setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return net


def predict_label(net, frames, classes):
    # frames are stacked into a blob of shape (1, 3, SAMPLE_DURATION, SAMPLE_SIZE, SAMPLE_SIZE)
    blob = cv2.dnn.blobFromImages(
        list(frames), 1.0, (SAMPLE_SIZE, SAMPLE_SIZE), MEAN, swapRB=True, crop=True)
    blob = np.transpose(blob, (1, 0, 2, 3))
    blob = np.expand_dims(blob, axis=0)
    net.setInput(blob)
    outputs = net.forward()
    return classes[np.argmax(outputs)]


def draw_label(frame, label):
    cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
    cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


class Output:
    """Displays and/or writes annotated frames. show() returns False when 'q' is pressed."""

    def __init__(self, path, fps, display):
        self.path = path
        self.fps = fps if fps and fps > 0 else 25
        self.display = display
        self.writer = None

    def show(self, frame):
        if self.path and self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.path, fourcc, self.fps,
                                          (frame.shape[1], frame.shape[0]), True)
        if self.writer is not None:
            self.writer.write(frame)
        if self.display:
            cv2.imshow("Activity Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False
        return True

    def close(self):
        if self.writer is not None:
            self.writer.release()
        cv2.destroyAllWindows()


def run_batch(vs, net, classes, out):
    """Classify non-overlapping blocks of SAMPLE_DURATION frames."""
    while True:
        frames, originals = [], []
        for _ in range(SAMPLE_DURATION):
            grabbed, frame = vs.read()
            if not grabbed:
                return
            originals.append(frame)
            frames.append(imutils.resize(frame, width=400))

        label = predict_label(net, frames, classes)
        for frame in originals:
            if not out.show(draw_label(frame, label)):
                return


def run_rolling(vs, net, classes, out):
    """Classify a rolling window of the last SAMPLE_DURATION frames."""
    frames = deque(maxlen=SAMPLE_DURATION)
    while True:
        grabbed, frame = vs.read()
        if not grabbed:
            return
        frames.append(imutils.resize(frame, width=400))
        if len(frames) < SAMPLE_DURATION:
            continue
        label = predict_label(net, frames, classes)
        if not out.show(draw_label(frame, label)):
            return


def main():
    args = parse_args()
    classes = open(args["classes"]).read().strip().split("\n")
    net = load_model(args["model"], gpu=args["gpu"] > 0)

    print("[INFO] Accessing the video stream...")
    vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
    out = Output(args["output"], vs.get(cv2.CAP_PROP_FPS), args["display"] > 0)
    try:
        if args["mode"] == "batch":
            run_batch(vs, net, classes, out)
        else:
            run_rolling(vs, net, classes, out)
    finally:
        print("[INFO] Exiting...")
        vs.release()
        out.close()


if __name__ == "__main__":
    main()
