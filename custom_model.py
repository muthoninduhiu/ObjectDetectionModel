import os
import json
import glob
import subprocess


def detect_custom_objects(pretrained_weights, img_size, conf_threshold, input_path):
    detect_command = [
        "python",
        "yolov5/detect.py",
        "--weights",
        pretrained_weights,
        "--img-size",
        str(img_size),
        "--conf-thres",
        str(conf_threshold),
        "--source",
        str(input_path),
        "--save-txt",
        "--save-conf",
        "--exist-ok"
    ]
    print(detect_command)
    subprocess.run(detect_command, check=True)

# image_ size
    img_size = 640
    # model path
    model_path = 'models/custom_yolov5s/best.pt/weights/best.pt'
    # define threshold
    conf_threshold = 0.4
