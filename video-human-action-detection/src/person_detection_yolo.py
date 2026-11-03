# PERSON DETECTION on video with YOLOv4 (Darknet weights, OpenCV DNN).
# First stage of a per-person activity pipeline: the detected boxes can be passed
# to an activity classifier (see clip_activity_recognition.py and the notebook).
#
# Usage:
#   python person_detection_yolo.py --weights models/yolov4.weights --cfg models/yolov4.cfg --names models/coco.names -i input.mp4
#   python person_detection_yolo.py ... -i rtsp://<user>:<password>@<host>:<port>/<path>   # IP camera stream

import argparse

import cv2
import numpy as np

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4


def parse_args():
    ap = argparse.ArgumentParser(description="Detect people in a video with YOLOv4")
    ap.add_argument("--weights", default="models/yolov4.weights", help="path to YOLOv4 weights")
    ap.add_argument("--cfg", default="models/yolov4.cfg", help="path to YOLOv4 config")
    ap.add_argument("--names", default="models/coco.names", help="path to COCO class names")
    ap.add_argument("-i", "--input", type=str, default="",
                    help="video file or stream URL (default: webcam)")
    return ap.parse_args()


# Load YOLOv4 model for human detection
def load_yolo(weights_path, cfg_path, names_path):
    net = cv2.dnn.readNet(weights_path, cfg_path)
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in np.array(net.getUnconnectedOutLayers()).flatten()]
    return net, classes, output_layers


# Run detector on a frame
def detect(frame, net, output_layers):
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    return net.forward(output_layers)


# Keep only "person" detections, apply non-maximum suppression
def person_boxes(detections, height, width, classes):
    boxes, confidences = [], []
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONF_THRESHOLD and classes[class_id] == "person":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                boxes.append([int(center_x - w / 2), int(center_y - h / 2), w, h])
                confidences.append(float(confidence))

    keep = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
    return [(boxes[i], confidences[i]) for i in np.array(keep).flatten()]


# Draw bounding boxes around detected people
def draw_boxes(frame, people):
    for (x, y, w, h), confidence in people:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"person {confidence:.2f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame


def main():
    args = parse_args()
    net, classes, output_layers = load_yolo(args.weights, args.cfg, args.names)
    cap = cv2.VideoCapture(args.input if args.input else 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        height, width = frame.shape[:2]
        detections = detect(frame, net, output_layers)
        people = person_boxes(detections, height, width, classes)
        frame = draw_boxes(frame, people)
        cv2.putText(frame, f"people: {len(people)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Person Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
