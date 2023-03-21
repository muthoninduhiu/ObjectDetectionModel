import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.utils import img_to_array

# Define the filename of the image to load
filename = 'images/test1.jpg'


# Load the ResNet50 model
def load_model():
    try:
        get_model = ResNet50(weights='imagenet')
    except ValueError:
        print("Error: Could not load ResNet50 model. Please make sure you have an internet connection.")
        exit()
    return get_model


# Load the image using PIL
def load_image(file):
    try:
        input_image = Image.open(file)
    except IOError:
        print(f"Error: Could not open image file {file}")
        exit()

    # Resize the image to 224x224 pixels
    input_image = input_image.resize((224, 224))
    # Change image Shape from RGBA to RGB as needed by resNet50
    input_image = input_image.convert('RGB')
    print(input_image)
    # Convert the PIL image to a numpy array
    numpy_image = img_to_array(input_image)
    # Scale the pixel values to be between -1 and 1
    scaled_image = preprocess_input(numpy_image)
    # Add a batch dimension to the array
    batch = np.expand_dims(scaled_image, axis=0)
    return batch, input_image


# Make a prediction on the image
def make_prediction(use_model, batches):
    my_predictions = use_model.predict(batches)
    my_predictions = decode_predictions(my_predictions, top=10)[0]
    print(my_predictions)
    return my_predictions


# Define the filename of the image to load
file_name = 'images/test.jpg'

# Load the ResNet50 model
model = load_model()

# Compile the model with appropriate optimizer, loss function, and metrics
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Load the image using PIL
image_batch, image = load_image(file_name)

# Make a prediction on the image
decoded_predictions = make_prediction(model, image_batch)
print(len(decoded_predictions))

model.save('object_recognition.h5')
