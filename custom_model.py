def detect_custom_objects(pretrained_weights, img_size, conf_threshold, input_path):
    """Detect objects in an image using a custom YOLOv5 model and draw bounding boxes on the predictions.
    It is still has some kinks but i am working on it
      Args:
          pretrained_weights (str): Path to the trained model weights file.
          img_size (int): The input size of the image (width and height).
          conf_threshold (float): Confidence threshold for object detection.
          input_path (str): Path to the input image directory or file.

      Returns:
          None.
      """
    d_command = ["python",
                 "detect.py",
                 "- -weights",
                 pretrained_weights,
                 "- -img",
                str(img_size),
                 "- -conf",
                 str(conf_threshold),
                 "- -source",
                 str(input_path)]
    print(d_command)
    # subprocess.run(d_command, check=True)


if __name__ == '__main__':
    detect_custom_objects('models/custom_yolov5s/best.pt/weights/best.pt', img_size=640, conf_threshold=0.5,
                          input_path="yolov5/datasets/test/images")
