import subprocess


def train_yolov5(data, cfg_path, pretrained_weights, num_epochs, batch_size, img_size, model_name, save_dir):
    # Define the YOLOv5 training command as a list of strings
    train_command = [
        "python",
        "yolov5/train.py",
        "--img",
        str(img_size),
        "--batch",
        str(batch_size),
        "--epochs",
        str(num_epochs),
        "--data",
        str(data),
        "--cfg",
        cfg_path,
        "--weights",
        pretrained_weights,
        "--name",
        model_name,
        "--weights",
        save_dir
    ]

    # Run the YOLOv5 training command using subprocess.Popen
    process = subprocess.Popen(train_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for the process to complete and capture its output and error messages
    stdout, stderr = process.communicate()

    # Print the captured output and error messages
    print(stdout.decode())
    print(stderr.decode())


if __name__ == "__main__":
    # Set the path to your custom dataset and configuration file
    data_path = "yolov5/datasets/data.yaml"
    cfg_path = "yolov5/models/yolov5x.yaml"
    # Set the path to the pre-trained YOLOv5 weights
    pretrained_weights = "yolov5x.pt"

    # Set the number of training epochs and batch size
    num_epochs = 8
    batch_size = 2

    # Set the input image size
    img_size = 640

    # Set the name of your custom model
    model_name = "best.pt"

    # Set the directory to save the trained model weights
    save_dir = "models/custom_yolov5s"

    # Train the YOLOv5 model
    train_yolov5(data_path, cfg_path, pretrained_weights, num_epochs, batch_size, img_size, model_name, save_dir)
