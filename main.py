import cv2
from prediction_results import extract_predictions
from visualization import visualize
from yolo import load_model, resize_image, detect_objects, filter_results, count_objects


def main():
    # Define the classes to detect
    classes = ['cell phone', 'laptop', 'satellite dish', 'USB stick', 'keyboard',
               'router', 'house keys', 'magnifying glass', 'server rack', 'mouse']
   

if __name__ == '__main__':
    main()
