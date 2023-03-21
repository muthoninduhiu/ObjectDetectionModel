import cv2
import numpy as np
import argparse
import torch

# Define the classes to detect
classes = ['cell phone', 'laptop', 'satellite dish', 'USB stick', 'keyboard',
           'router', 'house keys', 'magnifying glass', 'server rack', 'mouse']

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

# Set the model to evaluation mode
model.eval()


def detect_objects(image_path):
    # Load the image
    img = cv2.imread(image_path)
    # Resize the image
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_CUBIC)

    # Detect objects in the image
    results = model(img)

    # Visualize the results on the image
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, class_idx = result.tolist()
        label = classes[int(class_idx)]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f"{label}: {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the image
    cv2.imshow('Detection Results', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    # Use argparse to allow the user to specify the image path as a command-line argument
    parser = argparse.ArgumentParser(description='Detect objects in an image.')
    parser.add_argument('image_path', type=str, help='Path to the input image.')
    args = parser.parse_args()

    detect_objects(args.image_path)
