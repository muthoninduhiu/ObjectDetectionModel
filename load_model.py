import cv2
import torch
import numpy as np
from yolov5.models.experimental import attempt_load
from yolov5.utils.torch_utils import select_device
from yolov5.utils.general import non_max_suppression


def detect_objects(image_path, model_path, conf_threshold=0.5, iou_threshold=0.5):
    # Load YOLOv5 model
    device = select_device('')
    model = attempt_load(model_path)
    labels = model.names
    print(labels)
    # Set model to evaluation mode
    model.to(device).eval()

    # Load image
    img = cv2.imread(image_path)

    # Convert image to RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize image to model input size
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)

    # Normalize image values
    img = img.astype(np.float32) / 255.0

    # Convert image to tensor
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    # Detect objects on image
    with torch.no_grad():
        outputs = model(img)
        outputs = non_max_suppression(outputs, conf_threshold, iou_threshold)

    # Display image with bounding boxes around detected objects
    img = cv2.imread(image_path)
    for output in outputs:
        if output is not None:
            for x1, y1, x2, y2, conf, cls in output:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f'{cls}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display image with bounding boxes
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image_path = 'images/test.jpg'
    model_path = 'models/custom_yolov5s/best.pt/weights/best.pt'
    detect_objects(image_path, model_path)
