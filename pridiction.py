import keras
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image


model = keras.models.load_model('./Model/fun80')

# Define the class labels in the order of the model's prediction output



def predict(image_path='./testimages/images.jpeg'):
    class_labels = ['Actinic keratoses','Basal cell carcinoma', 'Benign keratosis-like lesions', 'Dermatofibroma', 'Melanocytic nevi', 'Melanoma', 'Vascular lesions', 'Vascular lesions']
    input_dims = (299,299)
    image = Image.open(image_path).resize(input_dims)

    # ms = 299
    # im = Image.open(image_path)
    # (a, b) = im.size
    # image = None
    # if a > b:
    #     c = a / ms
    #     input_dims = (ms, int(b/c))
    #     image = im.resize(input_dims)
    # else:
    #     c = b / ms
    #     input_dims = (int(a/c), ms)
    #     image = im.resize(input_dims)

    image_array = np.asarray(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_array)
    predicted_class = class_labels[np.argmax(prediction)]
    return predicted_class

print("name",__name__) 

if __name__ == "__main__":
    print(predict())