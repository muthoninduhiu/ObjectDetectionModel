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


if __name__ == '__main__':
    main()
