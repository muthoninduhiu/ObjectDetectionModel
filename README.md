# Object Detection Model using yoloV5x
In this project, we will use YOLOv5x (You Only Look Once version 5 extra-large) to detect objects in images.

## YOLOv5x

YOLOv5x is an object detection model that is based on the You Only Look Once (YOLO) algorithm. The YOLO algorithm is a one-stage object detection model that uses a single neural network to predict bounding boxes and class probabilities directly from full images in one evaluation. This makes YOLO faster than two-stage detection models like Faster R-CNN and SSD.

YOLOv5x is the largest version of the YOLOv5 models, with 177.2 million parameters. It has a larger number of layers and filters than the smaller models, which makes it more accurate but slower. YOLOv5x is a good choice for object detection tasks where accuracy is important, but where speed is still a consideration.

## Getting Started

To get started with YOLOv5x, you will need to:

* Install the required dependencies and libraries

* Load the model into your Python code from the github repository

* Run the model on your input image

* Visualize the results

## Installing Dependencies

To use YOLOv5x, you will need to have Python 3.8 or later installed, as well as the following dependencies:

* PyTorch

* OpenCV

* Pandas

You can install these dependencies using pip:

* pip install torch opencv-python numpy pandas psutil pyyaml seaborn

* pip install --force-reinstall --no-cache -U opencv-python==4.5.5.62

## Load the Model

To load the YOLOv5x model in Python, you can use the torch.hub module:

* import torch

* model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

## Running the Model

We start with a pre-trained model and fine-tune it on your own dataset.

* We set the model to evaluation mode so we can use it with our input image:
  - model.eval()
* get the image file path
* preproccess the image
* use this to find the objects detected on the image :
  - results = model(img)
  
## Visualize Predictions:

* Here we view the image with bounding boxes of the predictions that were found with their scores if they are above 0.4
* We print all the predicitions to the console

## Conclusion
YOLOv5x is a powerful object detection model that can be used to detect objects in images with high accuracy. By following the steps outlined in this tutorial, you can get started with using YOLOv5x for your own object detection tasks.

## Project Structure
objectDetectionModel/
    ├── README.md
    ├── images/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── prediction_results.py
    ├── visualization.py
    ├── pre_trained_model.py
    └── main.py
