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
image_path = 'images/mice.jpg'

# Load the image
img = cv2.imread(image_path)
# Resize the image
img = cv2.resize(img, (640, 640))
# print(img.shape)
# Detect objects in the image
results = model(img)

print("Results:{}".format(results))
# Create a DataFrame from the results
df = results.pandas().xyxy[0]

# Filter the DataFrame by the classes of interest
df = df[df['name'].isin(classes)]

# Count the occurrences of each item
counts = df['name'].value_counts()

# Print the counts
print(counts)
