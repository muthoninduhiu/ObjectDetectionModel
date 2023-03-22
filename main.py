import cv2
from prediction_results import extract_predictions
from visualization import visualize
from yolo import load_model, resize_image, detect_objects, filter_results, count_objects


def main():
    # Define the classes to detect
    classes = ['cell phone', 'laptop', 'satellite dish', 'USB stick', 'keyboard',
               'router', 'house keys', 'magnifying glass', 'server rack', 'mouse']
    model = load_model('yolov5x')

    # Define the image path
    image_path = 'images/several.jpg'

    # define resizing image size comment this when using test1.jpg
    # as we don't want to resize it for better results
    target_size = (640, 640)

    # Load the image
    img = cv2.imread(image_path)

    # Resize image
    img = resize_image(img, target_size)

    # detect objects
    results = detect_objects(img, model)
    print(results)
    
if __name__ == '__main__':
    main()
