import cv2
import torch


# Load the YOLOv5 model this is the smaller version with 213 layers
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# Load the YOLOv5 model
# Download and load the YOLOv5 model from Ultralytics
# 'yolov5x' is the model architecture to use, and 'pretrained=True' means to download the pre-trained weights

def load_model(model_architecture):
    """
    Resizes image to required target size
    Parameters:
       model_architecture: takes the name of the architecture we use
    Returns:
      pre-trained model we use in evaluation mode
    """
    model = torch.hub.load('ultralytics/yolov5', model_architecture, pretrained=True)
    # Set the model to evaluation mode, so it doesn't train while detecting objects in the image
    model.eval()
    return model


def resize_image(img, target_size):
    """
        Resizes image to required target size

        Parameters:
             img: input image to perform detection on.
             target_size: size model works with to detect objects e.g 640x640 for YOLOv5.

        Returns:
             an image that has been resized
    """
    # Resize the image
    return cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)


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
