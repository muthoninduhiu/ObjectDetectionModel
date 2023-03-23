from pprint import pprint

import cv2
import torch


def load_model(model_architecture):
    """
    Resizes image to required target size
    Parameters:
       model_architecture: takes the name of the architecture we use e.g 'yolov5s' but we use 'yolov5x'
       # 'yolov5s' model has with 213 layers and 'yolov5x has 444 layers
    Returns:
      pre-trained model we use in evaluation mode
    """
    # Load the YOLOv5 model
    # Download and load the YOLOv5 model from Ultralytics
    # 'yolov5x' is the model architecture to use, and 'pretrained=True'
    # means to download the pre-trained weights
    model = torch.hub.load('ultralytics/yolov5', model_architecture, pretrained=True)
    # Set the model to evaluation mode, so it doesn't train while detecting objects in the imageoo
    model.eval()
    labels = model.names
    pprint(labels)
    return model


def resize_image(img, target_size):
    """
        Resizes image to required target size

        Parameters:
             img: input image to perform detection on.
             target_size: size model works with to detect objects e.g 640x640 for YOLOv5.

        Returns:
             an image that has been resized maintaining its aspect ratio
    """
    # Get the current width and height of the image
    height, width = img.shape[:2]

    # Calculate the aspect ratio of the original image
    aspect_ratio = float(width) / height

    # Calculate the new width and height of the resized image while preserving aspect ratio
    target_width, target_height = target_size
    if (target_width / target_height) > aspect_ratio:
        new_width = int(target_height * aspect_ratio)
        new_height = target_height
    else:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)

    # Resize the image using the calculated new width and height
    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)


def detect_objects(img, model):
    """
    Detects objects in the given image using the specified object detection model.

    Parameters:
        img: A numpy.ndarray representing the input image to perform detection on.
        model: A pre-trained object detection model that can detect objects in images.

    Returns:
        An object containing the detection results, including
        the bounding boxes, object classes, and confidence scores.
    """
    # Detect objects in the image
    results = model(img)
    # print("Results:{}".format(results))
    return results


# print(img.shape)
def filter_results(results, classes):
    """
        Filters the detected objects in the given results by the specified classes.

        Parameters:
            results: The results from an object detection model.
            classes (list): A list of strings representing the classes of interest.

        Returns:
            A Pandas DataFrame containing the filtered results.
    """
    # Create a DataFrame from the results, so we can be able to count the objects detected
    df = results.pandas().xyxy[0]

    # Filter the DataFrame by the classes of interest
    df = df[df['name'].isin(classes)]
    return df


def count_objects(df):
    """
       Counts the occurrences of each object class in the given DataFrame.

       Parameters:
           df: A Pandas DataFrame containing the detected objects.

       Returns:
          A Pandas Series object containing the counts of each object class.
    """
    # Count the occurrences of each item with value_counts()

    counts = df['name'].value_counts()

    # Print the counts
    # print(counts)
    return counts
