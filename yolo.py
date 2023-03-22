import cv2
import torch


# Load the YOLOv5 model this is the smaller version with 213 layers
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# Load the YOLOv5 model
# Download and load the YOLOv5 model from Ultralytics
# 'yolov5x' is the model architecture to use, and 'pretrained=True' means to download the pre-trained weights

def load_model(model_architecture):
    model = torch.hub.load('ultralytics/yolov5', model_architecture, pretrained=True)
    # Set the model to evaluation mode, so it doesn't train while detecting objects in the image
    model.eval()
    return model


def resize_image(img, target_size):
    # Resize the image
    return cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)


def detect_objects(img, model):
    # Detect objects in the image
    results = model(img)
    # print("Results:{}".format(results))
    return results


# print(img.shape)
def filter_results(results, classes):
    # Create a DataFrame from the results, so we can be able to count the objects detected
    df = results.pandas().xyxy[0]

    # Filter the DataFrame by the classes of interest
    df = df[df['name'].isin(classes)]
    return df


def count_objects(df):
    # Count the occurrences of each item with value_counts()

    counts = df['name'].value_counts()

    # Print the counts
    # print(counts)
    return counts
