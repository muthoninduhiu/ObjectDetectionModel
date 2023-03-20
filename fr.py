import cv2
import torchvision
from torchvision.transforms import transforms

# Load the Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=True)

# Set the model to evaluation mode
model.eval()

# Define the image path
image_path = 'images/test.jpg'

# Load the image
img = cv2.imread(image_path)

# Define the classes to detect
classes = ['phone', 'laptop', 'satellite dish', 'USB stick', 'keyboard',
           'router', 'keys', 'magnifying glass', 'server rack', 'mouse']

# Define the transforms to apply to the image
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Apply the transforms to the image
img = transform(img)

# Get the predictions for the image
predictions = model([img])
print("Predictions:\n".format(predictions))
# Loop through the predicted boxes and draw them on the image
for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
    if score > 0.1 and label in classes:
        # Only display boxes with confidence score > 0.1 and class in the defined classes
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, f'{classes[label]} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

# Show the image
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
