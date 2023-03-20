import torch
import torchvision
import cv2

# Define the classes to detect
classes = ['person', 'car', 'dog']

# Load the Fast R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Set the model to evaluation mode
model.eval()

# Define the image path
image_path = 'images/test.jpg'

# Load the image
img = cv2.imread(image_path)

# Convert the image to RGB format
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Preprocess the image
transform = torchvision.transforms.ToTensor()
img = transform(img)

# Pass the image through the model to get the predicted bounding boxes, labels, and scores
outputs = model([img])

# Display the image with the predicted bounding boxes
boxes = outputs[0]['boxes'].detach().cpu().numpy()
labels = outputs[0]['labels'].detach().cpu().numpy()
scores = outputs[0]['scores'].detach().cpu().numpy()

for box, label, score in zip(boxes, labels, scores):
    if score > 0.5 and classes[label-1] in classes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'{classes[label-1]} {score:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow('Image', img.permute(1, 2, 0).numpy())
cv2.waitKey(0)
cv2.destroyAllWindows()
