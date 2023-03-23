import os
import cv2
from prediction_results import extract_predictions
from visualization import visualize
from pre_trained_model import load_model, resize_image, detect_objects, filter_results, count_objects


def main():
    """
        Loads an image and performs object detection using a pre-trained YOLOv5 model.
        Filters the detected objects based on a pre-defined list of classes.
        Counts the number of objects of each class.
        Displays the object detection results.
        Displays predictions on the image.

        Parameters:
        None

        Returns:
        None
        """
    # Define the classes to detect
    classes = ['cell phone', 'laptop', 'satellite dish', 'USB stick', 'keyboard',
               'router', 'house keys', 'magnifying glass', 'server rack', 'mouse']
    model = load_model('yolov5x')
    # Define the image path
    folder_path = 'images/'

    # define resizing image size
    target_size = (640, 640)
    # Loop through all the images in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is an image file
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            # Load the image
            image_path = os.path.join(folder_path, filename)
            img = cv2.imread(image_path)
            # Resize image
            img = resize_image(img, target_size)

            # detect objects
            results = detect_objects(img, model)
            print(results)

            # filter detected objects
            df = filter_results(results, classes)
            # count objects
            counts = count_objects(df)
            print(counts)
            # print predictions

            # define empty list
            detections = extract_predictions(results)
            # print out detections
            for detection in detections:
                print(f"Label: {detection['label']}, Score: {detection['score']:.2f}")
            # visualize the predictions
            visualize(results, img, detections)


# call the main method
if __name__ == '__main__':
    main()
