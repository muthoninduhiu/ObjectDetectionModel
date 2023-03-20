import cv2
import torch

# Define the classes to detect
classes = ['phone', 'laptop', 'satellite dish', 'USB stick', 'keyboard',
           'router', 'keys', 'magnifying glass', 'server rack', 'mouse', "apple"]

# Load the YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

# Set the model to evaluation mode
model.eval()

# Define the image path
image_path = 'images/test.jpg'

# Load the image
img = cv2.imread(image_path)
# Resize the image
img = cv2.resize(img, (640, 640))
# print(img.shape)
# Detect objects in the image
results = model(img)

print("Results:{}".format(results))
# Iterate over the results object and display the image with bounding boxes around each detected object
for i in range(len(results.xyxy[0])):
    bbox = results.xyxy[0][i]
    label = results.names[int(bbox[5])]
    score = bbox[4]
    x1, y1, x2, y2 = map(int, bbox[:4])
    cv2.rectangle(img, (x1, y1), (x2, y2), (200, 21, 255), 3)
    cv2.putText(img, f"{label} {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 100), 2)

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


