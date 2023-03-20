import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image

# Define the labels for the objects we want to detect
labels = ['phone', 'laptop', 'satellite dish', 'USB stick', 'keyboard',
          'router', 'keys', 'magnifying glass', 'server rack', 'mouse']

# Load the pre-trained Faster R-CNN model from torchvision
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Set the model to evaluation mode
model.eval()

# Define the transformations to be applied to the image
transform = T.Compose([
    T.Resize((800, 800)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define the image to be processed
image_path = 'images/test.jpg'
image = Image.open(image_path).convert('RGB')

# Apply the transformations to the image and add a batch dimension
image_tensor = transform(image).unsqueeze(0)

# Pass the image tensor through the model's forward() method
with torch.no_grad():
    predictions = model(image_tensor)

# Extract the predicted bounding boxes, labels, and scores
boxes = predictions[0]['boxes'].cpu().numpy()
labels = predictions[0]['labels'].cpu().numpy()
scores = predictions[0]['scores'].cpu().numpy()

# Create a list to store the detected objects
detected_objects = []

# Loop through the predicted boxes and append the label of each detected object to the list
for i in range(len(boxes)):
    if scores[i] > 0.5:  # Only consider boxes with confidence score > 0.5
        label = labels[i]
        detected_objects.append(labels[label - 1])  # Subtract 1 from the label to match the index of the labels list

# Print the list of detected objects
print(detected_objects)
