import cv2
import torch
import matplotlib.pyplot as plt
# Define the classes to detect
classes = ['phone', 'laptop', 'satellite dish', 'USB stick', 'keyboard',
           'router', 'keys', 'magnifying glass', 'server rack', 'mouse']

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set the model to evaluation mode
model.eval()

# Define the image path
image_path = 'images/mice.jpg'

# Load the image
img = cv2.imread(image_path)
# Resize the image
img = cv2.resize(img, (640, 640))
print(img.shape)
# Detect objects in the image
results = model(img)

# Get the predicted bounding boxes, labels, and scores
print("Results:\n{}".format(results))
boxes = results.xyxy[0].cpu().numpy().tolist()
labels = results.names
scores = results.xyxy[0][:, 4].cpu().numpy().tolist()
print("Boxes:{}\nLabels:{}\nScores:{}".format(boxes, labels, scores))
# # Plot the image and the predicted boxes
# # Loop through the boxes and draw them on the image
# for box, label, score in zip(boxes, labels, scores):
#     if score > 0.1 and label in classes:
#         # Only display boxes with confidence score > 0.1 and class in the defined classes
#         x1, y1, x2, y2 = map(int, box[:4])
#         cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
#         cv2.putText(img, f'{label} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
#
# # Show the image
# cv2.imshow('Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# Loop through the boxes and draw them on the image


# Display the image
fig, ax = plt.subplots()
ax.imshow(img[:, :, ::-1])

# Loop through the boxes and draw them on the image
for box, label, score in zip(boxes, labels, scores):
    if score > 0.1 and label in classes:
        # Only display boxes with confidence score > 0.1 and class in the defined classes
        x1, y1, x2, y2 = map(int, box[:4])
        width, height = x2 - x1, y2 - y1
        rect = plt.Rectangle((x1, y1), width, height, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f'{label} {score:.2f}', bbox=dict(facecolor='red', alpha=0.5),
                fontsize=8, color='white')

# Show the plot
plt.show()
